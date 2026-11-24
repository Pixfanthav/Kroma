package dev.kroma.open

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Anima Turbo (a 2B DiT on Cosmos-Predict2), fp16 or a fine-tune converted for it, on the phone's DSP ([Engine]).
 *
 *  text  : Qwen3-0.6B (CPU) -> llm_adapter with the T5 token ids (CPU) -> context [1,512,1024]
 *  blocks: engine/ beside the bundle (kconv's files), or a fine-tune's ([Custom]); the modulations on the CPU ([Adaln])
 *  shape : every aspect ratio by its RoPE table (rope_<W>x<H>.bin), a picture using the first gh*gw of the NB token rows
 *  LoRA  : [loraSet], run or baked by the engine; a LoRA's modulation Linears through [lora]'s adaln.txt
 *  sample: ER-SDE over the "simple" schedule at shift 3, CFG 1 (the Turbo distillation needs no guidance).
 *  end   : the same Wan21 latent and VAE decoder as Krea ([KreaGen.decode]).
 *
 * Files: anima/te/qwen3_te.onnx (+ .data), anima/llm_adapter.onnx, anima/tok/t5_vocab.tsv, anima/tok/t5_charsmap.bin,
 * tok/qwen_*.txt (shared with Krea), anima_bundle/anima.txt ("<NB> <steps> <cfg>").
 */
class AnimaGen(
    private val root: File, private val bundle: File,
    private val log: (String) -> Unit, private val status: (String) -> Unit, private val stop: () -> Boolean,
    private val W: Int, private val H: Int,
    private val lora: File? = null,          // the active folder ([LoraImport.activateAnima]): adaln.txt names a LoRA with modulation Linears
    private val loraSet: File? = null,       // the chosen LoRAs for the engine ([LoraSet])
) {
    private val env = OrtEnvironment.getEnvironment()
    private val adir = File(root, "anima")
    private val NB: Int = File(bundle, "anima.txt").readText().trim().split(Regex("\\s+"))[0].toInt()
    private val LH = H / 8; private val LW = W / 8; private val gh = LH / 2; private val gw = LW / 2
    private val LAT = 16 * LH * LW
    var lastX0: FloatArray? = null; private set
    var doneSteps = 0; private set

    // One AnimaGen can draw several pictures in a row ([generate] again, then [close]): the encoded prompt is kept when the text is the same (5-13 s).
    private var heldPrompt: String? = null; private var heldCtx: FloatArray? = null
    fun close() {}

    init { require(gh * gw <= NB) { "${W}x$H is over this bundle's token budget" } }

    // no prepacking: ORT would otherwise copy the 2.4GB of encoder weights into packed form before the first run, which is most
    // of what made the text step take 5-13 s instead of about 3 s
    private fun cpuOpts() = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4); setMemoryPatternOptimization(false); setCPUArenaAllocator(false)
        addConfigEntry("session.disable_prepacking", "1") }
    private fun floats(t: Any): FloatArray { val b = (t as OnnxTensor).floatBuffer; return FloatArray(b.remaining()).also { b.get(it) } }

    /**
     * The encoded prompt, from a small cache on the phone when this text was encoded before (the last 16 prompts, 2MB each, under
     * anima/ctx_cache; named by a hash of the text and of the encoder file, so nothing readable is kept and a new encoder is not
     * mixed with an old one).
     */
    private fun encodeCached(prompt: String, qtok: QwenTok): FloatArray {
        val te = File(adir, "te/qwen3_te.onnx"); val dir = File(adir, "ctx_cache")
        val key = java.security.MessageDigest.getInstance("SHA-256").digest("$prompt|${te.length()}|${te.lastModified()}".toByteArray()).joinToString("") { "%02x".format(it) }.take(32)
        val f = File(dir, "$key.f32")
        if (f.length() == CTX * 1024 * 4L) runCatching {
            val a = FloatArray(CTX * 1024); ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(a)
            f.setLastModified(System.currentTimeMillis()); log("  text from the cache"); return a
        }
        val a = encode(prompt, qtok)
        runCatching {
            dir.mkdirs(); val bb = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN); bb.asFloatBuffer().put(a); f.writeBytes(bb.array())
            dir.listFiles()?.sortedByDescending { it.lastModified() }?.drop(16)?.forEach { it.delete() }
        }
        return a
    }

    /** Prompt -> context [512 * 1024], zero-padded. */
    private fun encode(prompt: String, qtok: QwenTok): FloatArray {
        status(tr(R.string.preparing))
        var q = qtok.encode(prompt); if (q.isEmpty()) q = longArrayOf(PAD)
        var t5 = T5Tok(File(adir, "tok")).encode(prompt); if (t5.isEmpty()) t5 = longArrayOf(1L)
        // the encoder on the DSP (the converted encoder's folder), or its ONNX on the CPU where only that is there
        val kte = File(adir, "kte")
        val hidden = Engine.openKte(kte)?.use { it.encode(File(kte, "embed.f16"), IntArray(q.size) { i -> q[i].toInt() }, 1, 1024) }
            ?: env.createSession(File(adir, "te/qwen3_te.onnx").absolutePath, cpuOpts()).use { te ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(q), longArrayOf(1, q.size.toLong())).use { ids ->
                    te.run(mapOf(te.inputNames.first() to ids)).use { floats(it[0]) } }
            }
        val tAd = System.currentTimeMillis()
        val adapted = env.createSession(File(adir, "llm_adapter.onnx").absolutePath, cpuOpts()).use { ad ->
            val names = ad.inputNames.toList()
            OnnxTensor.createTensor(env, FloatBuffer.wrap(hidden), longArrayOf(1, q.size.toLong(), 1024)).use { src ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(t5), longArrayOf(1, t5.size.toLong())).use { tgt ->
                    ad.run(mapOf(names[0] to src, names[1] to tgt)).use { floats(it[0]) } } }
        }
        log("  text qwen ${q.size} t5 ${t5.size} tokens · adapter ${(System.currentTimeMillis() - tAd) / 1000.0}s")
        return FloatArray(CTX * 1024).also { System.arraycopy(adapted, 0, it, 0, minOf(t5.size, CTX) * 1024) }
    }

    // CONST flow matching with a time shift (3 is the model's own). Schedulers as ComfyUI has them: "simple" (the default, with
    // its offset first sigma), "karras" (rho 7 between the model's smallest and largest sigma) and "linear" (even steps in time).
    // The last entry is 0.
    private fun sigmas(steps: Int, shift: Double = 3.0, scheduler: String = "simple") = Sched.sigmas(steps, shift, scheduler)
    private fun ns(x: Double) = x * (Math.exp(Math.pow(x, 0.3)) + 10.0)
    private fun halfLogSnr(s: Double) = -(Math.log(s) - Math.log1p(-s))

    fun generate(prompt: String, qtok: QwenTok, steps: Int, seed: Long,
                 onPreview: ((FloatArray) -> Unit)? = null, onUnit: ((Int, Int, Int) -> Unit)? = null,
                 tune: Advanced.Tuning = Advanced.DEFAULT, init: FloatArray? = null, from: Int = 0, mask: FloatArray? = null, onStep: (Int, Long) -> Unit): FloatArray {
        val tEnc = System.currentTimeMillis()
        // the text beside the modulations and the blocks' opening: none of them needs another
        // with guidance, every step also runs the model on the negative prompt (twice the time)
        val textJob = java.util.concurrent.FutureTask {
            val c = heldCtx?.takeIf { prompt == heldPrompt } ?: encodeCached(prompt, qtok).also { heldCtx = it; heldPrompt = prompt }
            val n = if (tune.usesCfg) encodeCached(tune.negative, qtok) else null
            log("  text · ${(System.currentTimeMillis() - tEnc) / 1000}s")
            c to n
        }.also { Thread(it, "text").start() }
        if (tune !== Advanced.DEFAULT) log("  advanced: $tune")
        val ropeFile = File(bundle, "rope_${W}x$H.bin")
        val rope = if (ropeFile.exists()) ByteBuffer.wrap(ropeFile.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().let { b -> FloatArray(b.remaining()).also { b.get(it) } }
                   else ropeTable(W, H, NB)
        val cos = rope.copyOfRange(0, NB * 64); val sin = rope.copyOfRange(NB * 64, NB * 128)

        // patch (c, m, n) -> c*4 + m*2 + n over 16 latent channels + the padding-mask channel (0)
        fun patches(x: FloatArray) = FloatArray(NB * 68).also { o ->
            for (y in 0 until gh) for (xx in 0 until gw) for (c in 0 until 16) for (m in 0..1) for (n in 0..1)
                o[(y * gw + xx) * 68 + c * 4 + m * 2 + n] = x[(c * LH + 2 * y + m) * LW + 2 * xx + n]
        }
        // token (p1 p2 c) -> latent [16, 2y+p1, 2x+p2]
        fun unpatch(o: FloatArray) = FloatArray(LAT).also { v ->
            for (y in 0 until gh) for (xx in 0 until gw) for (p1 in 0..1) for (p2 in 0..1) for (c in 0 until 16)
                v[(c * LH + 2 * y + p1) * LW + 2 * xx + p2] = o[(y * gw + xx) * 64 + p1 * 32 + p2 * 16 + c]
        }

        // The modulations ([Adaln]) take 2-3 s on the CPU: they are computed while the blocks open, and
        // kept for the next picture with the same steps, LoRA and strength.
        val sig = sigmas(steps, tune.shift, tune.scheduler)
        // a checkpoint converted for the engine (kconv) brings its own modulation weights: a fine-tune changes them too
        // the engine's files: a fine-tune the user converted (Custom, extended mode), or the bundle's own
        val engDir = Custom.engineDir(bundle.parentFile!!, "anima")?.also { log("  model: ${it.parentFile?.name}") } ?: File(bundle, "engine")
        val adDir = engDir.takeIf { Adaln.present(it) && File(it, "embed.f16").exists() } ?: bundle
        val modsJob = if (!Adaln.present(adDir)) null else {
            // the LoRA in use, if it has modulation Linears: its own folder and the strength, as activateAnima noted them
            val al = lora?.let { File(it, "adaln.txt") }?.takeIf { it.exists() }?.readLines()
            val loraLib = al?.getOrNull(0)?.let { File(it) }; val st = al?.getOrNull(1)?.toFloatOrNull() ?: 1f
            val key = modsKey(adDir, steps, tune, loraLib, st)
            java.util.concurrent.FutureTask {
                modsCache?.takeIf { it.first == key }?.let { return@FutureTask it.second to "from the cache" }
                val t = System.currentTimeMillis()
                // without a LoRA the same schedule gives the same vectors: kept on disk, computed once
                val file = if (loraLib != null) null else modsFile(adir, key)
                file?.let { readMods(it, steps) }?.let { m -> modsCache = key to m; return@FutureTask m to "from the disk ${(System.currentTimeMillis() - t) / 1000.0}s" }
                val m = Adaln(adDir, loraLib, st).compute(FloatArray(steps) { sig[it].toFloat() })
                modsCache = key to m
                file?.let { runCatching { writeMods(it, m) } }
                m to "${(System.currentTimeMillis() - t) / 1000.0}s${if (loraLib != null) " · with the LoRA" else ""}"
            }.also { Thread(it, "adaln").start() }
        }

        val eng = (if (Adaln.present(adDir)) Engine.open(engDir, Adaln(adDir).blocks, NB, CTX, lora?.takeIf { File(it, "tag.txt").exists() }, log, loraSet) else null)
            ?: error(Engine.loraError ?: "the engine could not open ${engDir.name}")
        val (ctx, negCtx) = try { textJob.get() } catch (e: java.util.concurrent.ExecutionException) { eng.close(); throw e.cause ?: e }
        var tExec = 0L
        // the modulations of every step, computed beside the blocks' opening
        var mods: List<Map<String, FloatArray>>? = null
        try {
            /** One pass through the blocks: the model's prediction for x at sigma -> the denoised latent. */
            val passes = if (negCtx == null) 1 else 2
            fun denoise(x: FloatArray, sigma: Double, step: Int, cT: FloatArray = ctx): FloatArray {
                if (stop()) throw InterruptedException()
                val m = mods?.get(step) ?: error("the engine needs the modulations")
                val bm = FloatArray((m.size - 1) * 9 * 2048)                      // b<i>_mod one after another
                for (i in 0 until m.size - 1) System.arraycopy(m["b${i}_mod"]!!, 0, bm, i * 9 * 2048, 9 * 2048)
                val res = FloatArray(NB * 64); val t0 = System.currentTimeMillis()
                val nbk = m.size - 1                                              // the bar moves with every block
                // with guidance a step is two passes (the prompt, then the negative): counted as one unit, so the percent does not
                // fall back when the second pass starts over at block 0
                val nu = nbk * passes; val at = if (cT === negCtx) nbk else 0
                Engine.withProgress({ eng.progress() }, nbk, { b -> onUnit?.invoke(step, at + b, nu) }) {
                    eng.step(patches(x), cT, cos, sin, gh * gw, bm, m["final_mod"]!!, res) }
                tExec += System.currentTimeMillis() - t0
                if (Advanced.diagLog) log("  HASH step $step x %08x ctx %08x mod %08x f %08x res %08x".format(java.util.Arrays.hashCode(x), java.util.Arrays.hashCode(cT), java.util.Arrays.hashCode(bm), java.util.Arrays.hashCode(m["final_mod"]!!), java.util.Arrays.hashCode(res)))
                onUnit?.invoke(step, at + nbk, nu)
                val v = unpatch(res); val sf = sigma.toFloat()
                return FloatArray(LAT) { x[it] - v[it] * sf }
            }

            if (modsJob != null) {
                val t = System.currentTimeMillis()
                val (m, how) = try { var r: Pair<List<Map<String, FloatArray>>, String>? = null
                    while (r == null) { if (stop()) throw InterruptedException("stopped"); r = runCatching { modsJob.get(200, java.util.concurrent.TimeUnit.MILLISECONDS) }
                        .getOrElse { if (it is java.util.concurrent.TimeoutException) null else throw it } }
                    r } catch (e: java.util.concurrent.ExecutionException) { throw e.cause ?: e }
                mods = m
                log("  modulations on the CPU · $how · waited ${(System.currentTimeMillis() - t) / 1000.0}s")
            }
            val rng = java.util.Random(seed)
            fun noise() = FloatArray(LAT) { rng.nextGaussian().toFloat() }
            val s0 = sig[0].toFloat()
            val x = noise().also { for (j in it.indices) it[j] *= s0 }
            // img2img (the start picture's latent): noised to the sigma of the step it starts from, as the flow does
            val i0 = if (init != null) from else 0
            val n0 = if (init != null && mask != null) FloatArray(LAT) { x[it] / s0 } else null
            if (init != null) { val sf = sig[from].toFloat(); for (j in 0 until LAT) x[j] = x[j] / s0 * sf + (1 - sf) * init[j] }
            // inpaint: outside the mask the picture is put back, noised to the step's sigma, before every model call; and at the end
            // (Differential Diffusion, as in KreaGen: a cell is redrawn once its mask reaches the step's sigma)
            fun keepOut(s: Float) { if (n0 != null) { val hw = mask!!.size; for (j in 0 until LAT) { val m = mask[j % hw]; if (m <= 0f || m < s) x[j] = s * n0[j] + (1 - s) * init!![j] } } }
            val sampler = FlowSampler(tune.sampler, sig, i0, ::noise)
            for (i in i0 until steps) {
                if (stop()) break
                val t0 = System.currentTimeMillis(); val e0 = tExec
                keepOut(sig[i].toFloat())
                val den = try {
                    val dp = denoise(x, sig[i], i)
                    // classifier-free guidance: the prediction pushed away from the negative prompt's by the scale
                    if (negCtx == null) dp else { val dn = denoise(x, sig[i], i, negCtx); val g = tune.cfg.toFloat(); FloatArray(LAT) { dn[it] + g * (dp[it] - dn[it]) } }
                } catch (e: InterruptedException) { break }      // a stop mid-step abandons that step
                sampler.step(x, den, i)
                lastX0 = den; doneSteps = i + 1
                log("  step ${i + 1}: DSP ${(tExec - e0) / 1000.0}s · other ${(System.currentTimeMillis() - t0 - (tExec - e0)) / 1000.0}s" + Logs.stats(x))
                if (i + 1 < steps) onPreview?.invoke(den)
                onStep(i + 1, System.currentTimeMillis() - t0)
            }
            keepOut(0f)
            return x
        } finally { eng.close() }
    }

    companion object {
        const val CTX = 512
        private const val PAD = 151643L
        /** The last picture's modulations (about 21 MB for ten steps), by what they depend on. */
        @Volatile private var modsCache: Pair<String, List<Map<String, FloatArray>>>? = null
        private fun modNames(blocks: Int) = List(blocks) { "b${it}_mod" } + "final_mod"
        /** every step's vectors one after another, in [modNames]' order; the newest few schedules kept */
        private fun writeMods(f: File, m: List<Map<String, FloatArray>>) {
            f.parentFile!!.mkdirs(); val names = modNames(m[0].size - 1); val n = names.sumOf { m[0][it]!!.size }
            val bb = ByteBuffer.allocate(4 + n * m.size * 4).order(ByteOrder.LITTLE_ENDIAN); bb.putInt(m[0].size - 1)
            for (s in m) for (k in names) for (v in s[k]!!) bb.putFloat(v)
            val tmp = File(f.path + ".part"); tmp.writeBytes(bb.array()); tmp.renameTo(f)
            f.parentFile!!.listFiles()?.sortedByDescending { it.lastModified() }?.drop(8)?.forEach { it.delete() }
        }
        private fun readMods(f: File, steps: Int): List<Map<String, FloatArray>>? = runCatching {
            if (!f.isFile) return null
            val bb = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN); val blocks = bb.getInt(); val fb = bb.asFloatBuffer()
            val d = 2048; val rows = blocks * 9 * d + 2 * d
            if (fb.remaining() != rows * steps) return null
            f.setLastModified(System.currentTimeMillis())
            List(steps) { LinkedHashMap<String, FloatArray>().apply { for (k in modNames(blocks)) put(k, FloatArray(if (k == "final_mod") 2 * d else 9 * d).also { fb.get(it) }) } as Map<String, FloatArray> }
        }.getOrNull()
        private fun modsKey(adDir: File, steps: Int, tune: Advanced.Tuning, loraLib: File?, st: Float) =
            "${adDir.absolutePath}|${File(adDir, "adaln.f16").lastModified()}|$steps|${tune.shift}|${tune.scheduler}|${loraLib?.absolutePath}|${loraLib?.let { File(it, "info.json").lastModified() }}|$st"
        private fun modsFile(adir: File, key: String) = File(File(adir, "mods_cache"), java.security.MessageDigest.getInstance("SHA-256").digest(key.toByteArray()).joinToString("") { "%02x".format(it) } + ".f32")
        private var warming: Thread? = null
        /** The modulations of [steps] at the chosen schedule computed ahead and kept on disk, from the moment Anima is chosen or its steps or
         *  schedule change (the first picture after a change used to wait about 7 s for them). Without a LoRA only: a LoRA's depend on
         *  its strength, set at the picture. A second call while one runs waits for it, the newest setting wins. */
        fun warmMods(root: File, steps: Int, tune: Advanced.Tuning, log: (String) -> Unit) {
            val bundle = File(root, "anima_bundle"); val adir = File(root, "anima")
            val engDir = Custom.engineDir(root, "anima") ?: File(bundle, "engine")
            val adDir = engDir.takeIf { Adaln.present(it) && File(it, "embed.f16").exists() } ?: bundle
            if (!Adaln.present(adDir)) return
            val key = modsKey(adDir, steps, tune, null, 1f); val file = modsFile(adir, key)
            if (file.isFile || modsCache?.first == key) return
            synchronized(this) { val prev = warming
                warming = Thread({ prev?.join(); if (file.isFile) return@Thread
                    val t = System.currentTimeMillis()
                    runCatching { val sig = Sched.sigmas(steps, tune.shift, tune.scheduler); val m = Adaln(adDir, null, 1f).compute(FloatArray(steps) { sig[it].toFloat() })
                        modsCache = key to m; writeMods(file, m) }
                        .onSuccess { log("modulations ahead: $steps steps, ${tune.scheduler} shift ${tune.shift} · ${(System.currentTimeMillis() - t) / 1000.0}s") }
                        .onFailure { log("modulations ahead failed: ${it.message}") } }, "adaln-ahead").apply { isDaemon = true; start() } }
        }
        /** The token rows a bundle's blocks were built for (anima.txt's first number): a picture of gh*gw tokens needs gh*gw <= this. */
        fun budget(bundle: File): Int = runCatching { File(bundle, "anima.txt").readText().trim().split(Regex("\\s+"))[0].toInt() }.getOrDefault(4096)
        /** Any size whose sides are multiples of 16 (one token = 16x16 pixels) and whose tokens fit the bundle's rows. */
        fun accepts(bundle: File, w: Int, h: Int) = w % 16 == 0 && h % 16 == 0 && w >= 256 && h >= 256 && (w / 16) * (h / 16) <= budget(bundle)
        /** The RoPE table of a size, cos then sin, [NB, 64] each, spare rows identity: ComfyUI's Cosmos RoPE3D for Anima (h and w
         *  extrapolation ratio 4, one frame) -- the same numbers as the rope_<W>x<H>.bin files (checked to 3e-6 against 832x1216). */
        fun ropeTable(w: Int, h: Int, nb: Int): FloatArray {
            val gh = h / 16; val gw = w / 16; val dimH = 128 / 6 * 2; val dimT = 128 - 2 * dimH; val nh = dimH / 2; val nt = dimT / 2
            val ntk = Math.pow(4.0, dimH.toDouble() / (dimH - 2)); val theta = 10000.0 * ntk
            val freq = DoubleArray(nh) { 1.0 / Math.pow(theta, (2.0 * it) / dimH) }
            val out = FloatArray(nb * 128); for (i in 0 until nb * 64) out[i] = 1f       // cos 1, sin 0 for the spare rows and the time part
            for (y in 0 until gh) for (x in 0 until gw) {
                val r = y * gw + x
                for (k in 0 until nh) {
                    val ay = y * freq[k]; val ax = x * freq[k]
                    out[r * 64 + nt + k] = Math.cos(ay).toFloat(); out[nb * 64 + r * 64 + nt + k] = Math.sin(ay).toFloat()
                    out[r * 64 + nt + nh + k] = Math.cos(ax).toFloat(); out[nb * 64 + r * 64 + nt + nh + k] = Math.sin(ax).toFloat()
                }
            }
            return out
        }
    }
}
