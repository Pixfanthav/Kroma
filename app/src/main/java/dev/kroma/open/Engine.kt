package dev.kroma.open

import java.io.File

/**
 * The Kroma engine: the blocks on the Hexagon DSP, through libkengine.so.
 * Its files (engine/ beside the bundle: weights_b<i>.f16, norms_b<i>.f32, embed.f16, final.f16) are what kconv makes from a
 * safetensors checkpoint. The DSP code is libkengine_skel_<arch>.so, one per HVX architecture.
 */
class Engine private constructor(private var h: Long) : AutoCloseable {
    /** prediction [tokens x 64] for the patches, or an exception with the engine's error code */
    fun step(patches: FloatArray, ctx: FloatArray, cos: FloatArray, sin: FloatArray, kvalid: Int, mods: FloatArray, fmod: FloatArray, res: FloatArray) {
        val e = nStep(h, patches, ctx, cos, sin, kvalid, mods, fmod, res)
        if (e == -99) throw InterruptedException("stopped after a block")
        if (e != 0) error("engine step failed: ${Engine.why(e)}")
    }
    override fun close() { if (h != 0L) { nClose(h); h = 0L } }
    /** blocks finished in the step running now */
    fun progress() = if (h != 0L) nProgress(h) else 0

    companion object {
        /** how a LoRA set reaches the blocks, one path or the other and never the other behind the user's back: "run" (as they go,
         *  nothing written), "bake" (baked into a copy kept for the next picture) or "bake_once" (baked, the copy removed when the run
         *  of pictures ends). What the chosen path cannot do stops the picture with the reason ([loraError]). */
        private val loraMode get() = Advanced.loraMode
        private var bakedNow: File? = null
        // only "bake" keeps the copy. Called when a run of pictures ends, not per picture: the pictures of one run share the copy
        // instead of baking it each time.
        fun dropBaked() { if (loraMode != "bake") bakedNow?.deleteRecursively(); bakedNow = null }
        private fun runs(set: File) = loraMode == "run"
        /** why the LoRAs could not go the chosen way, for the screen; null when they did */
        @Volatile var loraError: String? = null
        private fun runError(code: Int) = when (code) {
            -6 -> "this DoRA cannot run yet (input axis or Klein): choose baking"
            -8 -> "more than 4096 ranks in one layer: choose baking"
            else -> "the LoRAs could not be loaded ($code)"
        }
        /** Runs [step] while reporting each finished block to [onBlock] (polled: the engine counts, the DSP calls back to no one). */
        fun <T> withProgress(progress: () -> Int, blocks: Int, onBlock: (Int) -> Unit, step: () -> T): T {
            val poll = Thread {
                var last = 0
                try { while (true) { val p = progress(); if (p in (last + 1) until blocks) { last = p; onBlock(p) }; Thread.sleep(100) } } catch (_: InterruptedException) {}
            }.apply { isDaemon = true; start() }
            try { return step() } finally { poll.interrupt(); poll.join(500) }
        }
        /** (blocks baked, blocks) while a LoRA is baked into the weights, from the opening thread; set by the screen */
        @Volatile var onBake: ((Int, Int) -> Unit)? = null
        private fun <T> baking(nb: Int, bake: () -> T): T {
            onBake?.invoke(0, nb)
            try { return withProgress({ nBakeProgress() }, nb, { b -> onBake?.invoke(b, nb) }, bake) } finally { onBake?.invoke(nb, nb) }
        }
        private val loaded = runCatching { System.loadLibrary("kengine") }.isSuccess
        /** the screen's part while the DSP is waited for ([dspRetry]): the seconds of the next wait (0 when the wait is over); false: stop waiting */
        @Volatile var onWait: ((Int) -> Boolean)? = null
        /** An open that the DSP refused is tried again after 2, 4, 8, 15 and 20 s. On an 8 Gen 2 Krea's, Klein's and Anima's
         *  first try after a download failed at once three times in a row and the same try went through 20 to 45 s later; SDXL's single
         *  tries failed the same way. The first refusal also puts the process's own fastrpc lines in the log, for the cause. */
        private val WAITS = intArrayOf(2, 4, 8, 15, 20)
        internal fun dspRetry(what: String, open: () -> Long): Long {
            var h = open(); var i = 0
            while (h == 0L && i < WAITS.size) {
                if (i == 0) { Logs.add("  $what: the DSP did not open"); Logs.ownLog() }
                if (onWait?.invoke(WAITS[i]) == false) break
                Thread.sleep(WAITS[i] * 1000L); h = open(); i++
            }
            if (i > 0) { onWait?.invoke(0); Logs.add("  $what: " + if (h != 0L) "opened at try ${i + 1}" else "still refused after ${i + 1} tries") }
            return h
        }
        private fun arch(chip: Chip?) = chip?.arch
        /** the engine can be tried here with these files (the DSP may still refuse when it opens) */
        fun usable(dir: File) = loaded && arch(Chip.detect()) != null && File(dir, "embed.f16").exists()
        /** null when the engine cannot run here (no library, an unknown chip, no files, or the DSP refused) */
        fun open(dir: File, blocks: Int, tokens: Int, ctxLen: Int, lora: File? = null, log: (String) -> Unit = {}, set: File? = null): Engine? {
            // the per-block diagnosis (Advanced.diagLog): the engine looks for this empty marker file of the app's own each step
            runCatching { File(dir, "kdebug").let { if (Advanced.diagLog) it.createNewFile() else if (it.length() == 0L) it.delete() } }
            val a = arch(Chip.detect()) ?: return null
            if (!loaded || !File(dir, "embed.f16").exists()) return null
            // the chosen LoRAs baked exactly, one after another ([set] lists them, LoraSet), instead of the 64-slot copy in [lora]
            if (set != null) {
                loraError = null
                if (runs(set)) {
                    val h0 = dspRetry("engine") { nOpen("", a, dir.absolutePath, "", blocks, tokens, ctxLen) }
                    if (h0 == 0L) return null
                    val lr = nLora(h0, set.absolutePath)
                    if (lr >= 0) { log(if (lr > 0) "  engine: LoRAs as they run · $lr rank slots, no baking" else "  engine: no LoRA for the blocks"); return Engine(h0) }
                    nClose(h0); loraError = runError(lr); log("  engine: ${loraError}"); return null
                }
                val baked = bakeSet(dir, set, blocks, 0, log) ?: run { loraError = "the LoRAs could not be baked"; return null }
                val h = dspRetry("engine") { nOpen("", a, dir.absolutePath, baked, blocks, tokens, ctxLen) }
                if (h != 0L) bakedNow = File(baked)
                return if (h == 0L) null else Engine(h)
            }
            // a LoRA is baked into a copy of the block weights once (engine_lora beside engine), kept while its tag stays the same
            var wdir = ""
            // the LoRA and the base it is baked into: another checkpoint in engine/ bakes anew
            val tag = lora?.let { runCatching { File(it, "tag.txt").readText() + "|" + File(dir, "weights_b0.f16").lastModified() }.getOrNull() }
            if (lora != null && tag != null) {
                val baked = File(dir.parentFile, "engine_lora")
                if (runCatching { File(baked, "tag.txt").readText() }.getOrNull() != tag) {
                    baked.deleteRecursively(); baked.mkdirs(); val t = System.currentTimeMillis()
                    val used = baking(blocks) { nBake(dir.absolutePath, lora.absolutePath, baked.absolutePath, blocks) }
                    if (used < 0) { baked.deleteRecursively(); log("  engine: LoRA bake failed ($used)"); return null }
                    File(baked, "tag.txt").writeText(tag)
                    log("  engine: LoRA baked into the weights · $used pairs · ${(System.currentTimeMillis() - t) / 1000.0}s")
                }
                wdir = baked.absolutePath
            }
            val h = dspRetry("engine") { nOpen("", a, dir.absolutePath, wdir, blocks, tokens, ctxLen) }
            return if (h == 0L) null else Engine(h)
        }
        @JvmStatic private external fun nOpen(lib: String, arch: String, dir: String, wdir: String, nb: Int, tokens: Int, ctx: Int): Long
        @JvmStatic private external fun nBake(base: String, lora: String, out: String, nb: Int): Int
        @JvmStatic private external fun nLora(h: Long, set: String): Int
        @JvmStatic private external fun nBakeSet(base: String, set: String, out: String, nb: Int, krea: Int): Int
        /** a zeroed direct buffer in native memory (not counted against the Java heap), and its release */
        fun direct(bytes: Long): java.nio.ByteBuffer = (nDirect(bytes) ?: throw OutOfMemoryError("native $bytes")).order(java.nio.ByteOrder.nativeOrder())
        fun freeDirect(b: java.nio.ByteBuffer) = nFreeDirect(b)
        @JvmStatic private external fun nDirect(bytes: Long): java.nio.ByteBuffer?
        @JvmStatic private external fun nFreeDirect(b: java.nio.ByteBuffer)
        @JvmStatic private external fun nTimeBake(tpath: String, set: String, sig: FloatArray, t: FloatArray, tvec: FloatArray): Int
        /** the timestep path with [set] in it, for these sigmas: t [n x 6144] and tvec [n x 36864] (kengine's time_bake); null on failure */
        fun timePath(tpath: File, set: File, sig: FloatArray, log: (String) -> Unit): Pair<FloatArray, FloatArray>? {
            if (!loaded) return null
            val t = FloatArray(sig.size * 6144); val tv = FloatArray(sig.size * 36864); val t0 = System.currentTimeMillis()
            val used = nTimeBake(tpath.absolutePath, set.absolutePath, sig, t, tv)
            if (used < 0) { log("  time LoRA: failed ($used)"); return null }
            log("  time LoRA: the timestep path computed with it · $used Linears · ${(System.currentTimeMillis() - t0) / 1000.0}s")
            return t to tv
        }
        @JvmStatic private external fun nTxtBake(out: String, set: String, keys: Array<String>, offs: LongArray, rows: IntArray, cols: IntArray, tr: IntArray): Int

        /** [set] baked into a copy of the blocks (engine_lora beside [dir], kept while the set, its folders and the base stay the
         *  same; a block no LoRA touches is a link to the base file). The copy's folder, or null when it failed. */
        private fun bakeSet(dir: File, set: File, nb: Int, krea: Int, log: (String) -> Unit): String? {
            val tag = LoraSet.tag(set) + "|" + File(dir, "weights_b0.f16").lastModified()
            val baked = File(dir.parentFile, "engine_lora")
            if (runCatching { File(baked, "tag.txt").readText() }.getOrNull() == tag) return baked.absolutePath
            baked.deleteRecursively(); baked.mkdirs(); val t = System.currentTimeMillis()
            val used = baking(nb) { nBakeSet(dir.absolutePath, set.absolutePath, baked.absolutePath, nb, krea) }
            if (used < 0) { baked.deleteRecursively(); log("  engine: LoRA bake failed ($used)"); return null }
            File(baked, "tag.txt").writeText(tag)
            log("  engine: LoRAs baked into the weights exactly · $used Linears · ${(System.currentTimeMillis() - t) / 1000.0}s")
            return baked.absolutePath
        }

        /**
         * Krea 2's text path with [set] baked in: a copy of txt2.onnx (one file, float32 inside) rewritten at each parameter's
         * offset (assets/txt2_offsets.json, made for that file's size). Kept beside it as txt2_lora.onnx while the set stays the
         * same. Null when the model is not the one the offsets are for, or the bake failed.
         */
        fun bakeText(kdir: File, set: File, log: (String) -> Unit): File? {
            val js = runCatching { org.json.JSONObject(Lang.res().assets.open("txt2_offsets.json").use { it.readBytes().toString(Charsets.UTF_8) }) }.getOrNull() ?: return null
            return bakeOnnx(kdir, "txt2", js, set, log)
        }
        /** first.onnx / last.onnx the same way (assets/cpu_offsets.json) */
        fun bakeOuter(kdir: File, name: String, set: File, log: (String) -> Unit): File? {
            val js = runCatching { org.json.JSONObject(Lang.res().assets.open("cpu_offsets.json").use { it.readBytes().toString(Charsets.UTF_8) }).getJSONObject("$name.onnx") }.getOrNull() ?: return null
            return bakeOnnx(kdir, name, js, set, log)
        }
        private fun bakeOnnx(kdir: File, name: String, js: org.json.JSONObject, set: File, log: (String) -> Unit): File? {
            val src = File(kdir, "$name.onnx"); val out = File(kdir, "${name}_lora.onnx"); val tagF = File(kdir, "${name}_lora.tag")
            val tag = LoraSet.tag(set) + "|" + src.length() + "|" + src.lastModified()
            if (out.isFile && runCatching { tagF.readText() }.getOrNull() == tag) return out
            if (js.optLong("size") != src.length()) { log("  $name LoRA: $name.onnx is not the file the offsets are for"); return null }
            val ps = js.getJSONObject("params"); val keys = ArrayList<String>(); val offs = ArrayList<Long>(); val rows = ArrayList<Int>(); val cols = ArrayList<Int>(); val tr = ArrayList<Int>()
            for (k in ps.keys()) {
                val e = ps.getJSONArray(k)
                keys.add(k.removeSuffix(".weight")); offs.add(e.getLong(0)); rows.add(e.getInt(1)); cols.add(e.getInt(2)); tr.add(e.getInt(3))
            }
            val t = System.currentTimeMillis()
            tagF.delete(); out.delete()
            val part = File(kdir, "${name}_lora.onnx.part")
            runCatching { java.io.FileInputStream(src).channel.use { i -> java.io.FileOutputStream(part).channel.use { o -> var p = 0L; val n = i.size(); while (p < n) p += i.transferTo(p, n - p, o) } } }
                .onFailure { part.delete(); log("  $name LoRA: no room for a copy of $name.onnx"); return null }
            val used = nTxtBake(part.absolutePath, set.absolutePath, keys.toTypedArray(), offs.toLongArray(), rows.toIntArray(), cols.toIntArray(), tr.toIntArray())
            if (used < 0) { part.delete(); log("  $name LoRA: bake failed ($used)"); return null }
            part.renameTo(out); tagF.writeText(tag)
            log("  $name LoRA: baked in exactly · $used parameters · ${(System.currentTimeMillis() - t) / 1000.0}s")
            return out
        }
        @JvmStatic private external fun nStep(h: Long, patches: FloatArray, ctx: FloatArray, cos: FloatArray, sin: FloatArray, kvalid: Int,
                                               mods: FloatArray, fmod: FloatArray, res: FloatArray): Int
        @JvmStatic private external fun nClose(h: Long)
        @JvmStatic private external fun nProgress(h: Long): Int
        @JvmStatic private external fun nBakeProgress(): Int
        /** the block loops stop before their next block (true), or run on (false: cleared as a run starts) */
        fun abort(on: Boolean) { if (loaded) runCatching { nAbort(if (on) 1 else 0) } }
        /** the VAE decoder's kept DSP session let go at the end of a run of pictures (kept, the DSP process lived as long as the app and
         *  filled up: SDXL's VAE failed with 0xe after Anima and SDXL in one app process) */
        fun vdecDrop() { if (loaded) runCatching { nVdecDrop() } }
        @JvmStatic private external fun nVdecDrop()
        @JvmStatic private external fun nAbort(on: Int)
        @JvmStatic private external fun nK2Progress(h: Long): Int

        /** Krea 2's 28 blocks on the DSP (kconv's files in [dir]), for a sequence of [tokens] rows; null when it cannot run here */
        fun openK2(dir: File, tokens: Int, lora: File? = null, strength: Float = 1f, log: (String) -> Unit = {}, set: File? = null): K2? {
            val a = arch(Chip.detect()) ?: return null
            if (!loaded || !File(dir, "weights_b0.f16").exists() || !File(dir, "params_b0.f32").exists()) return null
            val nb = generateSequence(0) { it + 1 }.takeWhile { File(dir, "weights_b$it.f16").exists() }.count()
            // blocks kept in memory: what MemAvailable leaves after 3.5GB for the rest of the picture (VAE, text encoder, the
            // engine's own buffers); each one saves a 0.87GB read every step
            val availMb = runCatching { File("/proc/meminfo").readLines().first { it.startsWith("MemAvailable") }.filter { it.isDigit() }.toLong() / 1024 }.getOrDefault(0L)
            val blockMb = File(dir, "weights_b0.f16").length() shr 20
            // and what the DSP can map: its address space is 32-bit. With fp16 blocks (870MB) at 832x1216 two kept blocks, the two
            // read buffers and the scratch came to 4GB and the fourth block's call failed with 0xe "Bad address" (on an 8 Gen 2
            // with 5.9GB free); three big buffers and the scratch (3.1GB) ran. So all of it stays under 3.3GB.
            val scrMb = (tokens.toLong() * (6144 * 4 + 1536 * 2 + 16384 * 2) * 2) shr 20
            val dspRoom = if (blockMb <= 0) 0 else ((3300 - scrMb - 2 * blockMb) / blockMb).toInt().coerceAtLeast(0)
            val nres = if (blockMb <= 0) 0 else minOf(((availMb - 3500) / blockMb).toInt(), dspRoom).coerceIn(0, nb)
            log("  engine: $nres of $nb blocks kept in memory (${availMb}MB available)")
            // a LoRA baked into a copy of the 28 blocks (23GB): only with room for it, else the picture is made without the LoRA.
            // A set ([LoraSet]) is baked exactly, the blocks it does not touch linked, not copied; one that only changes the text
            // path needs no copy at all.
            var wdir = ""
            if (set != null) {
                loraError = null
                val blocks = LoraImport.hasBlocks(LoraSet.dirs(set))
                // MOSS and Klein have no baked path (bake_set knows Anima's and Krea's blocks): they always run
                val runOnly = File(dir, "moss.txt").exists() || File(dir, "klein.txt").exists()
                if (blocks && (runs(set) || runOnly)) {
                    val h0 = dspRetry("engine") { nK2Open(a, dir.absolutePath, "", nb, tokens, nres, set.absolutePath) }
                    if (h0 == 0L) return null
                    val lr = nK2LoraSlots(h0)
                    if (lr < 0) { nK2Close(h0); loraError = runError(lr); log("  engine: ${loraError}"); return null }
                    log(if (lr > 0) "  engine: LoRAs as they run · $lr rank slots, no baking" else "  engine: no LoRA for the blocks")
                    File(dir.parentFile, "engine_lora").deleteRecursively()      // a baked copy from before is not needed now
                    return K2(h0)
                }
                if (blocks) {
                    val need = (0 until nb).sumOf { File(dir, "weights_b$it.f16").length() } + (2L shl 30)
                    val free = android.os.StatFs(dir.absolutePath).availableBytes + (File(dir.parentFile, "engine_lora").takeIf { it.isDirectory }?.listFiles()?.sumOf { it.length() } ?: 0L)
                    if (free < need) { loraError = "no room to bake the LoRAs (${free shr 30}GB free, ${need shr 30}GB needed)"; log("  engine: ${loraError}"); return null }
                    wdir = bakeSet(dir, set, nb, 1, log)?.also { bakedNow = File(it) } ?: run { loraError = "the LoRAs could not be baked"; return null }
                }
            } else if (lora != null) {
                val tag = "${lora.absolutePath}|${lora.lastModified()}|$strength|${File(dir, "weights_b0.f16").lastModified()}"
                val baked = File(dir.parentFile, "engine_lora")
                if (runCatching { File(baked, "tag.txt").readText() }.getOrNull() == tag) wdir = baked.absolutePath
                else {
                    baked.deleteRecursively()
                    val need = (0 until nb).sumOf { File(dir, "weights_b$it.f16").length() } + (2L shl 30)
                    val free = android.os.StatFs(dir.absolutePath).availableBytes
                    if (free < need) log("  engine: no room to bake the LoRA (${free shr 30}GB free, ${need shr 30}GB needed): made without it")
                    else {
                        baked.mkdirs(); val t = System.currentTimeMillis()
                        val used = baking(nb) { nK2Bake(dir.absolutePath, lora.absolutePath, baked.absolutePath, nb, strength) }
                        if (used < 0) { baked.deleteRecursively(); log("  engine: LoRA bake failed ($used)") }
                        else { File(baked, "tag.txt").writeText(tag); wdir = baked.absolutePath
                            log("  engine: LoRA baked into the weights · $used Linears · ${(System.currentTimeMillis() - t) / 1000.0}s") }
                    }
                }
            }
            val h = dspRetry("engine") { nK2Open(a, dir.absolutePath, wdir, nb, tokens, nres, "") }
            return if (h == 0L) null else K2(h)
        }
        @JvmStatic private external fun nK2Open(arch: String, dir: String, wdir: String, nb: Int, tokens: Int, nres: Int, set: String): Long
        @JvmStatic private external fun nK2Linear(h: Long, aux: Long, woff: Long, n: Int, k: Int, x: FloatArray, m: Int, out: FloatArray): Int
        @JvmStatic private external fun nK2LoraSlots(h: Long): Int
        @JvmStatic private external fun nK2Step(h: Long, seq: java.nio.FloatBuffer, tvec: java.nio.FloatBuffer, cos: java.nio.FloatBuffer, sin: java.nio.FloatBuffer,
                                                 ka: Int, kb: Int, ka2: Int, kb2: Int): Int
        @JvmStatic private external fun nK2Close(h: Long)
        /** MOSS-Image's text encoder (Qwen3.5-9B) on the DSP from the converted encoder's folder; null when it cannot open */
        fun openQte(dir: File): Qte? { val a = arch(Chip.detect()) ?: return null
            return if (!loaded || !File(dir, "qte_l0.w").exists()) null else dspRetry("text encoder") { nQteOpen(a, dir.absolutePath) }.takeIf { it != 0L }?.let { Qte(it) } }
        @JvmStatic private external fun nQteOpen(arch: String, dir: String): Long
        @JvmStatic private external fun nQteEncode(h: Long, ids: IntArray, out: FloatArray): Int
        @JvmStatic private external fun nQteProgress(h: Long): Int
        @JvmStatic private external fun nQteClose(h: Long)
        /** Krea 2's text encoder (Qwen3-VL-4B text layers) on the DSP from the converted encoder's folder; null when it cannot open */
        fun openKte(dir: File): Kte? { val a = arch(Chip.detect()) ?: return null
            return if (!loaded || !File(dir, "kte_l0.w").exists()) null else dspRetry("text encoder") { nKteOpen(a, dir.absolutePath) }.takeIf { it != 0L }?.let { Kte(it) } }
        /** the VAE decoder on the DSP: z [32 x H x W] -> ARGB [8 H x 8 W], from vdec.w and vdec.prm in [dir]; null where it can't run. times: kblock_vdec's */
        fun vdec(dir: File, z: FloatArray, H: Int, W: Int, times: IntArray = IntArray(32), qwen: Boolean = false): IntArray? {   // qwen: the Qwen-Image VAE, 16 channels
            val a = arch(Chip.detect()) ?: return null
            if (!loaded || !File(dir, "vdec.w").isFile) return null
            val px = IntArray(64 * H * W); return px.takeIf { nVdec(a, dir.absolutePath, z, H, W, if (qwen) 1 else 0, px, times) == 0 } }
        /** vdec at any size: past [VDEC_WHOLE] pixels a side (the DSP's address space) overlapping tiles, blended with linear ramps. z [C x H x W] */
        const val VDEC_WHOLE = 1408
        fun vdecAny(dir: File, z: FloatArray, C: Int, H: Int, W: Int, times: IntArray = IntArray(32), qwen: Boolean = false, whole: Int = VDEC_WHOLE): IntArray? {
            if (8 * maxOf(H, W) <= whole) return vdec(dir, z, H, W, times, qwen)
            // the fewest tiles that fit [whole] (each side on its own), the same size, in 8s: 2048 x 2048 is 2 x 2 tiles of 1088 (it was
            // 3 x 3 of 1024, 29 s of VAE)
            val OV = 16; val T = whole / 8
            fun tile(n: Int): Int { if (n <= T) return n; val k = (n - OV + T - OV - 1) / (T - OV); return minOf(T, ((n + (k - 1) * OV + k - 1) / k + 7) / 8 * 8) }
            val Th = tile(H); val Tw = tile(W)
            fun starts(n: Int, t: Int) = buildList { var p = 0; while (true) { if (p + t >= n) { add(n - t); break }; add(p); p += t - OV } }
            fun ramp(n: Int, first: Boolean, last: Boolean) = FloatArray(n) { i -> var a = 1f; val o = OV * 8f
                if (!first) a = minOf(a, minOf(i + 0.5f, o) / o); if (!last) a = minOf(a, minOf(n - 1 - i + 0.5f, o) / o); a }
            val PH = 8 * H; val PW = 8 * W; val acc = FloatArray(3 * PH * PW); val sum = FloatArray(PH * PW); val tz = FloatArray(C * Th * Tw)
            val ys = starts(H, Th); val xs = starts(W, Tw)
            for (y0 in ys) for (x0 in xs) {
                for (c in 0 until C) for (y in 0 until Th) System.arraycopy(z, (c * H + y0 + y) * W + x0, tz, (c * Th + y) * Tw, Tw)
                val p = vdec(dir, tz, Th, Tw, times, qwen) ?: return null
                val wy = ramp(8 * Th, y0 == 0, y0 + Th >= H); val wx = ramp(8 * Tw, x0 == 0, x0 + Tw >= W)
                for (y in 0 until 8 * Th) for (x in 0 until 8 * Tw) { val a = wy[y] * wx[x]; val v = p[y * 8 * Tw + x]; val i = (y0 * 8 + y) * PW + x0 * 8 + x
                    sum[i] += a; acc[3 * i] += a * ((v shr 16) and 255); acc[3 * i + 1] += a * ((v shr 8) and 255); acc[3 * i + 2] += a * (v and 255) }
            }
            return IntArray(PH * PW) { i -> val s = sum[i]; (0xFF shl 24) or ((acc[3 * i] / s + 0.5f).toInt() shl 16) or ((acc[3 * i + 1] / s + 0.5f).toInt() shl 8) or (acc[3 * i + 2] / s + 0.5f).toInt() }
        }
        /** Real-ESRGAN on the DSP: rgb [3 x H x W] 0..1 -> [3 x 4H x 4W], from esr.w and esr.prm in [dir]; null where it can't run */
        fun esr(dir: File, rgb: FloatArray, H: Int, W: Int): FloatArray? {
            val a = arch(Chip.detect()) ?: return null
            if (!loaded || !File(dir, "esr.w").isFile) return null
            val out = FloatArray(48 * H * W); return out.takeIf { nEsr(a, dir.absolutePath, rgb, H, W, it, IntArray(4)) == 0 } }
        @JvmStatic private external fun nEsr(arch: String, dir: String, rgb: FloatArray, H: Int, W: Int, out: FloatArray, times: IntArray): Int
        /** the Qwen-Image VAE's encoder on the DSP: img [3 x H x W] -1..1 -> the latent's mean [16 x H / 8 x W / 8], from venc.w and venc.prm in [dir]; null where it can't run */
        fun qenc(dir: File, img: FloatArray, H: Int, W: Int): FloatArray? {
            val a = arch(Chip.detect()) ?: return null
            if (!loaded || !File(dir, "venc.w").isFile) return null
            val z = FloatArray(16 * (H / 8) * (W / 8)); return z.takeIf { nQenc(a, dir.absolutePath, img, H, W, it, IntArray(4)) == 0 } }
        @JvmStatic private external fun nQenc(arch: String, dir: String, img: FloatArray, H: Int, W: Int, z: FloatArray, times: IntArray): Int
        @JvmStatic private external fun nVdec(arch: String, dir: String, z: FloatArray, H: Int, W: Int, qw: Int, px: IntArray, times: IntArray): Int
        /** SDXL (SdxlGen): the Euler loop over the UNet's folder, z the noise in and the latent out; step(i, n) true stops it (-99) */
        fun xlGen(dir: String, ctx: FloatArray, y: FloatArray, z: FloatArray, lh: Int, lw: Int, steps: Int, cfg: Float, step: (Int, Int) -> Boolean): Int {
            val a = arch(Chip.detect()) ?: return -1
            return nXlGen(a, dir, ctx, y, z, lh, lw, steps, cfg, object : XlStep { override fun step(i: Int, n: Int) = step(i, n) })
        }
        fun xlVae(dir: String, z: FloatArray, lh: Int, lw: Int, rgb: FloatArray): Int { val a = arch(Chip.detect()) ?: return -1; return nXlVae(a, dir, z, lh, lw, rgb) }
        interface XlStep { fun step(i: Int, n: Int): Boolean }
        /** FLUX.2's VAE encoder on the DSP (fvenc.w in [dir]): rgb [3 x H x W] in -1..1, W a multiple of 256, H of 16 -> mean [32 x H/8 x W/8]; 0 when done */
        fun fvEnc(dir: String, rgb: FloatArray, H: Int, W: Int, mean: FloatArray): Int { if (!loaded) return -1; val a = arch(Chip.detect()) ?: return -1; return runCatching { nFvEnc(a, dir, rgb, H, W, mean) }.getOrDefault(-1) }
        @JvmStatic private external fun nFvEnc(arch: String, dir: String, rgb: FloatArray, H: Int, W: Int, mean: FloatArray): Int
        /** the SDXL runtime LoRA (LoraImport.buildSdxl's folder) for the next generations, "" for none */
        fun xlLora(dir: String) { if (loaded) runCatching { nXlLora(dir) } }
        /** where the SDXL UNet is within its step: (step, units done, units) -- the runs of layers, or the layers unbatched */
        fun xlProgress(): Triple<Int, Int, Int> { val p = runCatching { nXlProgress() }.getOrDefault(0); return Triple(p ushr 20, (p ushr 10) and 1023, p and 1023) }
        @JvmStatic private external fun nXlProgress(): Int
        @JvmStatic private external fun nXlLora(dir: String)
        @JvmStatic private external fun nXlGen(arch: String, dir: String, ctx: FloatArray, y: FloatArray, z: FloatArray, H: Int, W: Int, steps: Int, cfg: Float, cb: XlStep): Int
        @JvmStatic private external fun nXlVae(arch: String, dir: String, z: FloatArray, H: Int, W: Int, rgb: FloatArray): Int
        /** what the DSP says of itself, one line for a bug report (Logs.report): its VTCM, the engine's HVX threads, the HMX lock and clock vote,
         *  whether the engine could start (and why not) and a one-tile fp16 HMX self-test's error -- a few hundred ms, the DSP opened for it */
        fun hwInfo(c: Chip): String {
            if (!loaded) return "engine library not loaded"
            val o = IntArray(16); val e = nHwInfo(c.arch, o)
            if (e != 0) return "the DSP did not answer: ${why(e)}"
            return "v${o[8]} · VTCM ${o[0] shr 20}MB total, ${o[1] shr 20}MB free, the engine asks ${o[2] shr 20}MB · HVX threads ${o[3]} · " +
                "HMX lock4 ${o[4]}, clock vote ${o[5]} · engine start ${why(o[6])} · fp16 HMX tile " + (if (o[7] < 0) "not run" else "error %.4f%%".format(o[7] / 1e4))
        }
        /** an engine error code in words where it has a known meaning (0x4000 | MB: the chip's VTCM is too small), else in hex */
        fun why(err: Int): String = when {
            err == 0 -> "ok"
            err and 0xFF00 == 0x4000 -> "this chip's VTCM is ${err and 0xFF} MB, the engine needs 8 MB"
            else -> "0x" + Integer.toHexString(err)
        }
        @JvmStatic private external fun nHwInfo(arch: String, out: IntArray): Int
        @JvmStatic private external fun nKteOpen(arch: String, dir: String): Long
        @JvmStatic private external fun nKteEncode(h: Long, emb: String, ids: IntArray, out: FloatArray): Int
        @JvmStatic private external fun nKteClose(h: Long)
        fun mossAux(f: File): MossAux? = if (!loaded || !f.isFile) null else nMossAux(f.absolutePath).takeIf { it != 0L }?.let { MossAux(it) }
        @JvmStatic private external fun nMossAux(path: String): Long
        @JvmStatic private external fun nMossAuxClose(h: Long)
        @JvmStatic private external fun nMossLinear(h: Long, woff: Long, boff: Long, n: Int, k: Int, x: FloatArray, m: Int, out: FloatArray): Int
        @JvmStatic private external fun nK2Bake(base: String, lora: String, out: String, nb: Int, strength: Float): Int
    }

    /** MOSS-Image's text encoder: token ids -> the last hidden state [T x 4096]; [layers] gets the layers done (0..32) while it runs */
    class Qte internal constructor(private var h: Long) : AutoCloseable {
        fun encode(ids: IntArray, layers: (Int) -> Unit = {}): FloatArray {
            val out = FloatArray(ids.size * 4096)
            val r = withProgress({ nQteProgress(h) }, 32, layers) { nQteEncode(h, ids, out) }
            if (r != 0) error("text encoder failed ($r)"); return out
        }
        override fun close() { if (h != 0L) { nQteClose(h); h = 0L } }
    }

    /** Krea 2's text encoder: token ids -> the twelve tapped hidden states [12 x T x 2560]; emb is krea/embed.f16 */
    class Kte internal constructor(private var h: Long) : AutoCloseable {
        fun encode(emb: File, ids: IntArray, outputs: Int = 12, d: Int = 2560): FloatArray {
            val out = FloatArray(outputs * ids.size * d)
            val r = nKteEncode(h, emb.absolutePath, ids, out)
            if (r != 0) error("text encoder failed ($r)"); return out
        }
        override fun close() { if (h != 0L) { nKteClose(h); h = 0L } }
    }

    /** MOSS-Image's parts outside the blocks (kconv's moss_aux.f16, mapped): products out = x W^T + b on the CPU, by byte offsets
     *  from moss_aux.json ([MossGen] reads it) */
    class MossAux internal constructor(private var h: Long) : AutoCloseable {
        internal val handle get() = h
        /** [M x N] from x [M x K]; W [N x K] at [woff], b [N] at [boff] (-1: none) */
        fun linear(woff: Long, boff: Long, n: Int, k: Int, x: FloatArray, m: Int): FloatArray {
            val out = FloatArray(m * n); val r = nMossLinear(h, woff, boff, n, k, x, m, out)
            if (r != 0) error("moss linear failed ($r)"); return out
        }
        override fun close() { if (h != 0L) { nMossAuxClose(h); h = 0L } }
    }

    /** One pass through the blocks: seq (direct, [tokens x 6144]) in and out; keys [ka, kb) and [ka2, kb2) masked */
    class K2 internal constructor(private var h: Long) : AutoCloseable {
        /** [M x N] from x [M x K] on the HMX; W [N x K] at [woff] of [aux] */
        fun linear(aux: MossAux, woff: Long, n: Int, k: Int, x: FloatArray, m: Int): FloatArray {
            val out = FloatArray(m * n); val r = nK2Linear(h, aux.handle, woff, n, k, x, m, out)
            if (r != 0) error("engine linear failed: "+r); return out
        }
        fun step(seq: java.nio.FloatBuffer, tvec: java.nio.FloatBuffer, cos: java.nio.FloatBuffer, sin: java.nio.FloatBuffer, ka: Int, kb: Int, ka2: Int, kb2: Int) {
            val e = nK2Step(h, seq, tvec, cos, sin, ka, kb, ka2, kb2)
            if (e == -99) throw InterruptedException("stopped after a block")
            if (e != 0) error("engine step failed: ${Engine.why(e)}")
        }
        override fun close() { if (h != 0L) { nK2Close(h); h = 0L } }
        fun progress() = if (h != 0L) nK2Progress(h) else 0
    }
}
