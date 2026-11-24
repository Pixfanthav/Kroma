package dev.kroma.open

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.concurrent.thread

/**
 * Krea 2 Turbo (12.8B DiT, fp16) or a fine-tune converted for it, on the phone's DSP ([Engine]).
 *
 *  text    : Qwen3-VL-4B text layers on the DSP (krea/kte); the 12 tapped hidden
 *            states -> txt2.onnx (txtfusion + txtmlp, CPU) -> [S,6144]
 *  per step: patchify -> first.onnx -> [text padded to T | image tokens | spare rows] -> 28 blocks on the DSP
 *            -> last.onnx -> velocity -> the sampler
 *  end     : Wan21 process_out -> the qwen_image VAE decoder
 *
 * krea.txt "W H T R 1": W x H the token budget, T the text rows. RoPE is an input, so every aspect ratio whose token
 * grid fits runs.
 */
class KreaGen(
    root: File,
    private val bundle: File,
    private val log: (String) -> Unit,
    private val status: (String) -> Unit,
    private val stop: () -> Boolean,
    private val W: Int, private val H: Int,
    private val lora: File? = null, private val loraStrength: Float = 1f,     // a folder written by LoraImport.convert
    private val loraSet: File? = null,                                         // the chosen LoRAs for the exact bakes ([LoraSet])
) {
    private val env = OrtEnvironment.getEnvironment()
    private val kdir = File(root, "krea")
    private val T: Int; private val R: Int; private val budget: Int
    init {
        val c = File(bundle, "krea.txt").readText().trim().split(Regex("\\s+")).map { it.toInt() }
        require(c.size > 4 && c[4] == 1) { "not a RoPE-input bundle" }
        T = c[2]; R = c[3]; budget = (c[0] / 16) * (c[1] / 16)
        require((W / 16) * (H / 16) <= budget) { "${W}x$H is over this bundle's token budget" }
    }
    private val LH = H / 8; private val LW = W / 8          // latent
    private val gh = H / 16; private val gw = W / 16         // token grid
    private val NI = gh * gw                                 // image tokens (this run)
    private val L = T + budget                               // compiled sequence length
    private val D = 6144
    // Text is not tied to the T text rows: every row is fed its own RoPE and key mask, and text sits at position 0 like a spare image
    // slot does, so the image rows this picture's shape leaves unused can carry more text (none at 1:1, 144 at 2:3).
    private val TT = textRows(T, budget, W, H)      // text rows this run can use

    /** After generate(): the last step's denoised estimate, and how many steps completed. */
    var lastX0: FloatArray? = null; private set
    var doneSteps = 0; private set

    private fun cpuOpts(threads: Int = 6) = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads); setMemoryPatternOptimization(false)
        setCPUArenaAllocator(false); addConfigEntry("session.disable_prepacking", "1")
    }
    private fun t(a: FloatArray, vararg shape: Long) = OnnxTensor.createTensor(env, FloatBuffer.wrap(a), shape)
    private fun floats(o: Any): FloatArray { val b = (o as OnnxTensor).floatBuffer; val a = FloatArray(b.remaining()); b.get(a); return a }
    /** a .f16 file as floats (the LoRA pairs LoraImport writes) */
    private fun readF16(f: File): FloatArray {
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        return FloatArray(b.limit()) { val h = b.get(it).toInt() and 0xFFFF
            val e = (h shr 10) and 0x1F; val m = h and 0x3FF; val sg = if (h and 0x8000 != 0) -1f else 1f
            sg * when (e) { 0 -> m * 5.9604645e-8f; 31 -> if (m == 0) Float.POSITIVE_INFINITY else Float.NaN
                            else -> Math.scalb((1024 + m).toFloat(), e - 25) } }
    }
    private fun readF32(f: File): FloatArray {
        val bb = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(bb.remaining()).also { bb.get(it) }
    }

    // ---------------------------------------------------------------- text
    /** Prompt -> the 12 tapped Qwen3 hidden states per token, cond [S*12*2560], and S. */
    private fun encodeCond(prompt: String, qtok: QwenTok): Pair<FloatArray, Int> {
        val tpl = org.json.JSONObject(File(kdir, "template.json").readText())
        fun ints(k: String) = tpl.getJSONArray(k).let { a -> IntArray(a.length()) { a.getInt(it) } }
        val pre = ints("prefix"); val suf = ints("suffix"); val body = qtok.encode(prompt).map { it.toInt() }.toIntArray()
        val hid = 2560
        // the model keeps what follows "user\n" of the 2nd <|im_start|>
        var te = pre.withIndex().filter { it.value == 151644 }[1].index
        if (pre[te + 1] == 872 && pre[te + 2] == 198) te += 3
        // too long: the prompt is cut, never the template's end -- a text left without it broke the picture (colours)
        val room = TT - (pre.size - te) - suf.size; val whole = pre.size - te + body.size + suf.size
        if (whole > TT) log(tr(dev.kroma.open.R.string.prompt_too_long_only_the, TT, whole))
        val ids = pre + body.copyOf(minOf(body.size, room)) + suf
        // the encoder on the DSP (the converted encoder's folder)
        val dsp = Engine.openKte(File(kdir, "kte")) ?: error("the text encoder could not open krea/kte")
        status(tr(dev.kroma.open.R.string.preparing))
        val all = dsp.use { it.encode(File(kdir, "embed.f16"), ids) }
        val n = ids.size
        val S = n - te
        val cond = FloatArray(S * 12 * hid)
        for (s in 0 until S) for (l in 0 until 12)
            System.arraycopy(all, (l * n + te + s) * hid, cond, (s * 12 + l) * hid, hid)
        return cond to S
    }

    /** cond -> txt [max(S,T)*D] (zero-padded to T rows): txtfusion + txtmlp. Its LoRA inputs get rank-1 zeros. */
    private fun fuse(cond: FloatArray, S: Int): FloatArray {
        val hid = 2560
        val meta = org.json.JSONObject(File(kdir, "txt2_meta.json").readText()).getJSONArray("lora_inputs")
        val txt = FloatArray(maxOf(S, T) * D); var textLo = 0
        // a change the LoRA inputs cannot carry (a LoKr, DoRA, a norm scale, a full rank) is baked into a copy of the text path,
        // and then the inputs stay zero
        val baked = loraSet?.takeIf { LoraImport.textNeedsBake(LoraSet.dirs(it)) }?.let { status(tr(dev.kroma.open.R.string.preparing_the_lora)); Engine.bakeText(kdir, it, log) }
        val lora = if (baked != null) null else lora
        env.createSession((baked ?: File(kdir, "txt2.onnx")).absolutePath, cpuOpts()).use { s ->
            val feeds = LinkedHashMap<String, OnnxTensor>()
            feeds["cond"] = t(cond, 1, S.toLong(), 12L * hid)
            for (i in 0 until meta.length()) {
                val name = meta.getString(i).replace('.', '_')
                val a = s.inputInfo["${name}_A"]!!.info as ai.onnxruntime.TensorInfo
                val b = s.inputInfo["${name}_B"]!!.info as ai.onnxruntime.TensorInfo
                // the chosen LoRA's pair for this Linear if it has one (txtfusion.* / txtmlp.*), else a rank-1 zero
                val key = meta.getString(i)
                val fa = lora?.let { File(it, "$key.A.f16") }?.takeIf { it.isFile }
                val fb = lora?.let { File(it, "$key.B.f16") }?.takeIf { it.isFile }
                val inD = a.shape[1].toInt(); val outD = b.shape[0].toInt()
                val r = if (fa != null && fb != null) (fa.length() / 2 / inD).toInt() else 0
                if (r in 1..4096 && fa!!.length() == r.toLong() * inD * 2 && fb!!.length() == outD.toLong() * r * 2) {
                    val av = readF16(fa); val bv = readF16(fb)
                    if (loraStrength != 1f) for (k in bv.indices) bv[k] *= loraStrength
                    feeds["${name}_A"] = t(av, r.toLong(), a.shape[1]); feeds["${name}_B"] = t(bv, b.shape[0], r.toLong())
                    textLo++
                } else {
                    feeds["${name}_A"] = t(FloatArray(inD), 1, a.shape[1])
                    feeds["${name}_B"] = t(FloatArray(outD), b.shape[0], 1)
                }
            }
            s.run(feeds).use { r -> System.arraycopy(floats(r[0]), 0, txt, 0, S * D) }
            if (textLo > 0) log("  text LoRA: $textLo Linears")
            feeds.values.forEach { it.close() }
        }
        return txt
    }

    // ---------------------------------------------------------------- sampling
    /** [init]: img2img's start picture as a model-space latent ([encode]), noised to sigma[from] and denoised from step [from] on. */
    fun generate(prompt: String, qtok: QwenTok, steps: Int, seed: Long,
                 onPreview: ((FloatArray) -> Unit)? = null, onUnit: ((Int, Int, Int) -> Unit)? = null,
                 init: FloatArray? = null, from: Int = 0, mask: FloatArray? = null, tune: Advanced.Tuning? = null, onStep: (Int, Long) -> Unit): FloatArray {
        val tEnc = System.currentTimeMillis()
        val (cond, S) = encodeCond(prompt, qtok)
        val txt = fuse(cond, S)
        log("  text $S tokens · ${(System.currentTimeMillis() - tEnc) / 1000}s")
        // tmlp/tproj only ever see sigma: with the model's own schedule their outputs come precomputed per step count; another
        // schedule (extended mode) has them computed for its sigmas from tpath.bf16
        val own = tune == null || tune.scheduler == "krea"
        val sig = if (own) org.json.JSONArray(File(kdir, "sigmas_$steps.json").readText()).let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
                  else Sched.sigmas(steps, tune!!.shift, tune.scheduler)
        val st = sig.size - 1                                             // the steps that run (the "beta" schedule can drop one)
        var tvecAll: FloatArray; var tAll: FloatArray
        if (own) { tvecAll = readF32(File(kdir, "tvec_$steps.bin")); tAll = readF32(File(kdir, "t_$steps.bin")) }
        else {
            val tp = File(kdir, "tpath.bf16"); require(tp.isFile) { "this Krea bundle has no tpath.bf16: only its own schedule runs" }
            val none = File(System.getProperty("java.io.tmpdir") ?: "/data/local/tmp", "krea_no_lora.txt").apply { writeText("") }
            val (t1, tv1) = Engine.timePath(tp, loraSet ?: none, FloatArray(st) { sig[it].toFloat() }, log) ?: error("the time path could not be computed")
            tAll = t1; tvecAll = tv1
            log("  schedule ${tune.scheduler} shift ${tune.shift} · sampler ${tune.sampler} · cfg ${tune.cfg}")
        }
        // a LoRA on tmlp / tproj: the path computed for these sigmas with it (from tpath.bf16, which a bundle may leave out)
        val setDirs = loraSet?.let { LoraSet.dirs(it) } ?: emptyList()
        if (loraSet != null && LoraImport.hasTime(setDirs)) {
            val tp = File(kdir, "tpath.bf16")
            if (!tp.isFile) log("  time LoRA: left out (no tpath.bf16 in the bundle)")
            else Engine.timePath(tp, loraSet, FloatArray(st) { sig[it].toFloat() }, log)?.let { (t, tv) -> tAll = t; tvecAll = tv }
        }
        require(T + NI <= L)
        // key mask: text padding and the spare image slots never get attention -- except the first `extra` spare
        // slots, which hold the text rows beyond T
        val extra = maxOf(0, S - T)
        require(T + NI + extra <= L)
        val rng = java.util.Random(seed)
        val LAT = 16 * LH * LW
        val x = FloatArray(LAT) { rng.nextGaussian().toFloat() }        // sigma[0] = 1
        // img2img: the flow's noising, sigma noise + (1 - sigma) picture, at the step it starts from
        val n0 = if (init != null && mask != null) x.copyOf() else null
        if (init != null) { val sf = sig[from].toFloat(); for (j in 0 until LAT) x[j] = sf * x[j] + (1 - sf) * init[j] }
        // inpaint ([mask] [LH*LW], 1 = redraw), as ComfyUI's Differential Diffusion: before every step a cell is redrawn only once its mask
        // reaches the step's sigma, else the picture is put back noised to that sigma -- a soft edge is redrawn in the last steps only,
        // so it joins what stays instead of being a blend of two pictures. At the end (sigma 0) every masked cell keeps what was drawn.
        fun keepOut(s: Float) { if (n0 != null) { val hw = LH * LW; for (j in 0 until LAT) { val m = mask!![j % hw]; if (m <= 0f || m < s) x[j] = s * n0[j] + (1 - s) * init!![j] } } }
        // Every big tensor lives in native (direct) memory: one [L,6144] fp32 activation is 104MB and the Java heap is capped.
        fun direct(n: Int): FloatBuffer = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        val seq = direct(L * D)
        val tvecBuf = direct(36864)
        // RoPE for this run's grid: text at position 0, image (0,y,x); axes 32/48/48 dims -> 16/24/24
        // frequencies, theta 1000. Spare slots: cos 1, sin 0.
        val cosBuf = direct(L * 64); val sinBuf = direct(L * 64)
        run {
            val om1 = DoubleArray(24) { 1.0 / Math.pow(1000.0, 2.0 * it / 48) }
            val row = FloatArray(64)
            for (p in 0 until L) {
                val cell = (p - T).takeIf { it in 0 until NI } ?: -1
                val yy = if (cell < 0) -1 else cell / gw; val xx = if (cell < 0) -1 else cell % gw
                fun ang(j: Int) = when { cell < 0 || j < 16 -> 0.0; j < 40 -> yy * om1[j - 16]; else -> xx * om1[j - 40] }
                for (j in 0 until 64) row[j] = Math.cos(ang(j)).toFloat()
                cosBuf.put(row)
                for (j in 0 until 64) row[j] = Math.sin(ang(j)).toFloat()
                sinBuf.put(row)
            }
        }
        val padRows = FloatArray((L - T - NI - extra) * D)
        fun outer(n: String) = (if (loraSet != null && LoraImport.hasOuter(setDirs, "$n.")) Engine.bakeOuter(kdir, n, loraSet, log) else null) ?: File(kdir, "$n.onnx")
        val first = env.createSession(outer("first").absolutePath, cpuOpts())
        val last = env.createSession(outer("last").absolutePath, cpuOpts())
        val keng = Engine.openK2(Custom.engineDir(bundle.parentFile!!, "krea")?.also { log("  model: ${it.parentFile?.name}") } ?: File(bundle, "engine"), L, lora, loraStrength, log, loraSet)
            ?: error(Engine.loraError ?: "the engine could not open the Krea 2 blocks")
        try {
            // a stop mid-step abandons that step and returns: the caller decides by doneSteps what to keep
            // extended mode: another sampler, or guidance (the model run again on the negative prompt, engine only)
            val i0 = if (init != null) from else 0
            val plain = tune == null || (tune.sampler == "euler" && !tune.usesCfg)
            val sampler = if (plain) null else FlowSampler(tune!!.sampler, sig, i0) { FloatArray(LAT) { rng.nextGaussian().toFloat() } }
            val neg = if (tune != null && tune.usesCfg) encodeCond(tune.negative, qtok).let { (c, n) -> fuse(c, n) to n } else null
            /** the engine's velocity for this step's patches with another text (the negative prompt) */
            fun velocityWith(p: FloatArray, tx: FloatArray, sx: Int, i: Int): FloatArray {
                val ex = maxOf(0, sx - T); require(T + NI + ex <= L)
                seq.clear(); seq.put(tx, 0, T * D)
                t(p, 1, NI.toLong(), 64).use { pt -> first.run(mapOf("p" to pt)).use { r -> seq.put((r[0] as OnnxTensor).floatBuffer) } }
                if (ex > 0) seq.put(tx, T * D, ex * D)
                val pad = FloatArray((L - T - NI - ex) * D); if (pad.isNotEmpty()) seq.put(pad)
                tvecBuf.clear(); tvecBuf.put(tvecAll, i * 36864, 36864)
                Engine.withProgress({ keng.progress() }, 28, { b -> onUnit?.invoke(i, 28 + b, 56) }) {
                    keng.step(seq, tvecBuf, cosBuf, sinBuf, minOf(sx, T), T, T + NI + ex, L) }
                onUnit?.invoke(i, 56, 56)
                seq.position(T * D); val iv = seq.slice().apply { limit(NI * D) }; seq.rewind()
                return OnnxTensor.createTensor(env, iv, longArrayOf(1, NI.toLong(), D.toLong())).use { xi ->
                    t(tAll.copyOfRange(i * D, (i + 1) * D), 1, 1, D.toLong()).use { ti -> last.run(mapOf("x" to xi, "t" to ti)).use { r -> floats(r[0]) } } }
            }
            stepLoop@ for (i in i0 until st) {
                if (stop()) break
                val t0 = System.currentTimeMillis()
                keepOut(sig[i].toFloat())
                // patchify: token (y,x) feature c*4+ph*2+pw = lat[c, 2y+ph, 2x+pw]
                val p = FloatArray(NI * 64)
                for (yy in 0 until gh) for (xx in 0 until gw) for (c in 0 until 16) for (ph in 0..1) for (pw in 0..1)
                    p[(yy * gw + xx) * 64 + c * 4 + ph * 2 + pw] = x[(c * LH + 2 * yy + ph) * LW + 2 * xx + pw]
                seq.clear(); seq.put(txt, 0, T * D)
                t(p, 1, NI.toLong(), 64).use { pt -> first.run(mapOf("p" to pt)).use { r -> seq.put((r[0] as OnnxTensor).floatBuffer) } }
                if (extra > 0) seq.put(txt, T * D, extra * D)    // the text beyond T, in the first spare slots
                if (padRows.isNotEmpty()) seq.put(padRows)       // the rest of the spare slots start each step at 0
                tvecBuf.clear(); tvecBuf.put(tvecAll, i * 36864, 36864)
                if (stop()) break@stepLoop
                // with guidance the negative pass follows (velocityWith): the step counts 56 blocks, the second 28 there
                val nu = if (neg != null) 56 else 28
                val te0 = System.currentTimeMillis()
                Engine.withProgress({ keng.progress() }, 28, { b -> onUnit?.invoke(i, b, nu) }) {     // the bar moves with every block
                    keng.step(seq, tvecBuf, cosBuf, sinBuf, minOf(S, T), T, T + NI + extra, L) }
                val tExec = System.currentTimeMillis() - te0
                onUnit?.invoke(i, 28, nu)
                log("  step ${i + 1}: DSP ${tExec / 1000.0}s · other ${(System.currentTimeMillis() - t0 - tExec) / 1000.0}s")
                // last layer on this run's image rows -> velocity, unpatchify, flow Euler
                seq.position(T * D); val imgView = seq.slice().apply { limit(NI * D) }; seq.rewind()
                val tt = tAll.copyOfRange(i * D, (i + 1) * D)
                var v = OnnxTensor.createTensor(env, imgView, longArrayOf(1, NI.toLong(), D.toLong())).use { xi ->
                    t(tt, 1, 1, D.toLong()).use { ti -> last.run(mapOf("x" to xi, "t" to ti)).use { r -> floats(r[0]) } } }
                if (neg != null) {   // classifier-free guidance on the velocity
                    val vn = velocityWith(p, neg.first, neg.second, i); val g = tune!!.cfg.toFloat()
                    v = FloatArray(v.size) { vn[it] + g * (v[it] - vn[it]) }
                }
                run { var a = 0.0; var m = 0f; for (f in v) { a += f * f; if (Math.abs(f) > m) m = Math.abs(f) }
                    log("  v ${i + 1}: rms %.5g max %.5g".format(Math.sqrt(a / v.size), m) + Logs.stats(v)) }
                if (sampler != null) {
                    // the denoised estimate x0 = x - sigma v, then the chosen sampler's step
                    val si = sig[i].toFloat(); val den = FloatArray(LAT)
                    for (yy in 0 until gh) for (xx in 0 until gw) for (c in 0 until 16) for (ph in 0..1) for (pw in 0..1) {
                        val li = (c * LH + 2 * yy + ph) * LW + 2 * xx + pw
                        den[li] = x[li] - si * v[(yy * gw + xx) * 64 + c * 4 + ph * 2 + pw]
                    }
                    sampler.step(x, den, i)
                    lastX0 = if (i + 1 < st) den else x
                    if (i + 1 < st) onPreview?.invoke(den)
                    doneSteps = i + 1
                    onStep(i + 1, System.currentTimeMillis() - t0)
                    continue@stepLoop
                }
                val dts = (sig[i + 1] - sig[i]).toFloat()
                for (yy in 0 until gh) for (xx in 0 until gw) for (c in 0 until 16) for (ph in 0..1) for (pw in 0..1) {
                    val li = (c * LH + 2 * yy + ph) * LW + 2 * xx + pw
                    x[li] += dts * v[(yy * gw + xx) * 64 + c * 4 + ph * 2 + pw]
                }
                // the denoised estimate x0 = x_{i+1} - sigma_{i+1} v (the noisy x is mostly grain): the live
                // preview, and what a stopped run keeps instead of the half-noised latent
                if (i + 1 < st) {
                    val s1 = sig[i + 1].toFloat(); val x0 = FloatArray(LAT)
                    for (yy in 0 until gh) for (xx in 0 until gw) for (c in 0 until 16) for (ph in 0..1) for (pw in 0..1) {
                        val li = (c * LH + 2 * yy + ph) * LW + 2 * xx + pw
                        x0[li] = x[li] - s1 * v[(yy * gw + xx) * 64 + c * 4 + ph * 2 + pw]
                    }
                    lastX0 = x0
                    onPreview?.invoke(x0)
                } else lastX0 = x
                doneSteps = i + 1
                onStep(i + 1, System.currentTimeMillis() - t0)
            }
        } finally { first.close(); last.close(); keng.close() }
        if (doneSteps == st) keepOut(0f)
        return x
    }

    companion object {
        // Wan 2.1 latent format (the qwen_image VAE)
        private val MEAN = floatArrayOf(-0.7571f,-0.7089f,-0.9113f,0.1075f,-0.1745f,0.9653f,-0.1517f,1.5508f,
            0.4134f,-0.0715f,0.5517f,-0.3632f,-0.1922f,-0.9497f,0.2503f,-0.2921f)
        private val STD = floatArrayOf(2.8184f,1.4541f,2.3275f,2.6558f,1.2196f,1.7708f,2.6052f,2.0743f,
            3.2687f,2.1526f,2.8652f,1.5579f,1.6382f,1.1253f,2.8251f,1.9160f)
        // latent -> RGB in one matrix product, for the per-step preview (a real decode takes ~a minute)
        private val RGBF = floatArrayOf(
            -0.1299f,-0.1692f, 0.2932f,  0.0671f, 0.0406f, 0.0442f,  0.3568f, 0.2548f, 0.1747f,  0.0372f, 0.2344f, 0.1420f,
             0.0313f, 0.0189f,-0.0328f,  0.0296f,-0.0956f,-0.0665f, -0.3477f,-0.4059f,-0.2925f,  0.0166f, 0.1902f, 0.1975f,
            -0.0412f, 0.0267f,-0.1364f, -0.1293f, 0.0740f, 0.1636f,  0.0680f, 0.3019f, 0.1128f,  0.0032f, 0.0581f, 0.0639f,
            -0.1251f, 0.0927f, 0.1699f,  0.0060f,-0.0633f, 0.0005f,  0.3477f, 0.2275f, 0.2950f,  0.1984f, 0.0913f, 0.1861f)
        private val RGBB = floatArrayOf(-0.1835f, -0.0868f, -0.3360f)

        fun preview(x0: FloatArray, lh: Int, lw: Int): Bitmap {
            val n = lh * lw
            val px = IntArray(n) { p ->
                var r = RGBB[0]; var g = RGBB[1]; var b = RGBB[2]
                for (c in 0 until 16) { val v = x0[c * n + p]; r += v * RGBF[c * 3]; g += v * RGBF[c * 3 + 1]; b += v * RGBF[c * 3 + 2] }
                fun u(f: Float) = ((f + 1f) * 127.5f).toInt().coerceIn(0, 255)
                (0xFF shl 24) or (u(r) shl 16) or (u(g) shl 8) or u(b)
            }
            return Bitmap.createScaledBitmap(Bitmap.createBitmap(px, lw, lh, Bitmap.Config.ARGB_8888), lw * 8, lh * 8, true)
        }

        /** Text rows a picture of this size can use: the T text rows plus its unused image rows, as far as the text encoder reaches (its length minus the 34-token template prefix). */
        fun textRows(T: Int, budget: Int, W: Int, H: Int) = T + budget - (W / 16) * (H / 16)

        /** whether img2img can run at this size: the encoder is one fixed-size NPU graph per size */
        fun canEncode(root: File, W: Int, H: Int) = W % 8 == 0 && H % 8 == 0 && maxOf(W, H) <= Engine.VDEC_WHOLE && File(root, "onnx/qvae/venc.w").isFile

        /**
         * img2img's start picture -> the model-space latent [16,H/8,W/8]: scaled to cover W x H and cut to it in the middle, -1..1,
         * the VAE encoder on the NPU (its mean), Wan21 process_in. Krea 2 and Anima share the VAE.
         */
        fun encode(root: File, src: Bitmap, W: Int, H: Int): FloatArray {
            val k = maxOf(W.toFloat() / src.width, H.toFloat() / src.height)
            val sw = Math.round(src.width * k).coerceAtLeast(W); val sh = Math.round(src.height * k).coerceAtLeast(H)
            val sc = Bitmap.createScaledBitmap(src, sw, sh, true)
            val px = IntArray(W * H).also { sc.getPixels(it, 0, W, (sw - W) / 2, (sh - H) / 2, W, H) }
            if (sc !== src) sc.recycle()
            val img = FloatArray(3 * W * H)
            for (i in 0 until W * H) { val p = px[i]
                img[i] = ((p shr 16) and 255) / 127.5f - 1; img[W * H + i] = ((p shr 8) and 255) / 127.5f - 1; img[2 * W * H + i] = (p and 255) / 127.5f - 1 }
            val t0 = System.currentTimeMillis()
            val z = Engine.qenc(File(root, "onnx/qvae"), img, H, W) ?: throw IllegalStateException("VAE encoder on the DSP failed or its files are missing")
            val n = (H / 8) * (W / 8)
            for (c in 0 until 16) for (p in 0 until n) { val i = c * n + p; z[i] = (z[i] - MEAN[c]) / STD[c] }
            Logs.add("vae encode on the DSP ${System.currentTimeMillis() - t0}ms")
            return z
        }

        /**
         * Model-space latent [16,LH,LW] -> Wan21 process_out -> the VAE decoder -> bitmap.
         * On the DSP, any size (the files in onnx/qvae).
         */
        fun decode(root: File, x: FloatArray, LH: Int, LW: Int): Bitmap {
            val proc = FloatArray(x.size)
            for (c in 0 until 16) for (p in 0 until LH * LW) { val i = c * LH * LW + p; proc[i] = x[i] * STD[c] + MEAN[c] }
            val env = OrtEnvironment.getEnvironment()
            val t0 = System.currentTimeMillis()
            // the DSP's decoder, any size, when it is there (the files in onnx/qvae)
            val tm = IntArray(32); Engine.vdecAny(File(root, "onnx/qvae"), proc, 16, LH, LW, tm, true)?.let { px -> Logs.add("vae on the DSP ${System.currentTimeMillis() - t0}ms: product ${tm[24] / 1000} (fill ${tm[29] / 1000}) norm ${tm[27] / 1000} add ${tm[28] / 1000} ms")
                return Bitmap.createBitmap(px, 8 * LW, 8 * LH, Bitmap.Config.ARGB_8888) }
            throw IllegalStateException("VAE on the DSP failed or its files (onnx/qvae) are missing")
        }
    }
}
