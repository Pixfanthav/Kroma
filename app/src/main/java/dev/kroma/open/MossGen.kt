package dev.kroma.open

import java.io.File

/**
 * MOSS-Image (OpenMOSS, Apache 2.0): the 42 blocks on the DSP through the Krea path in MOSS
 * mode (kconv's weights_b<i>.f16 / params_b<i>.f32 with moss.txt), everything else here on the CPU from moss_aux.f16.
 *
 * The text encoder's last hidden state [T x 4096] comes in from outside. Then, once per prompt: the text refiner (a Linear and two
 * plain transformer blocks with biases), and per step: the time embedding and the modulations (shared by all blocks), the input layer on
 * the latent, the blocks, the output head. The sampler is [generateFlow]'s.
 */
class MossGen(private val dir: File, private val log: (String) -> Unit = {}) : AutoCloseable {
    companion object {
        const val D = 3072; const val H = 24; const val DH = 128; const val F = 8192; const val C = 128; const val TD = 4096
        /**
         * The prompt as MOSS-Image's pipeline gives it to its text encoder: Qwen's chat template with the generation prompt and thinking on
         * ("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"), the four markers as their own ids. [tok] is a QwenTok
         * over Qwen3.5's vocabulary (qwen_vocab.txt / qwen_merges.txt from its tokenizer.json).
         */
        fun promptIds(tok: QwenTok, prompt: String): IntArray {
            val r = ArrayList<Int>()
            fun add(s: String) { tok.encode(s).forEach { r.add(it.toInt()) } }
            r.add(248045); add("user\n$prompt"); r.add(248046); add("\n"); r.add(248045); add("assistant\n"); r.add(248068); add("\n")
            return r.toIntArray()
        }
        /** the text encoder's hidden state for [prompt] (MossText: the tokenizer loaded ahead, the last prompt's result kept) */
        fun encode(qteDir: File, tokDir: File, prompt: String, log: (String) -> Unit = {}): Pair<FloatArray, Int> =
            MossText.hidden(qteDir, MossText.ids(tokDir, prompt), log)
        /** the release's latent step (batch-norm undone, 2 x 2 unpatchify: channel c * 4 + 2 py + px) and the DSP's decoder */
        fun decode(x0: FloatArray, gh: Int, gw: Int, vae: File): android.graphics.Bitmap {
            val bn = org.json.JSONObject(File(vae, "vae_bn.json").readText()); val mean = bn.getJSONArray("mean"); val std = bn.getJSONArray("std")
            val n = gh * gw; val H = 2 * gh; val W = 2 * gw; val z = FloatArray(32 * H * W)
            for (c in 0 until 128) { val mu = mean.getDouble(c).toFloat(); val sd = std.getDouble(c).toFloat(); val c32 = c / 4; val py = (c / 2) % 2; val px = c % 2
                for (i in 0 until n) { val y = i / gw; val x = i % gw; z[(c32 * H + 2 * y + py) * W + 2 * x + px] = x0[c * n + i] * sd + mu } }
            // the DSP's decoder, any size, when the bundle has it
            val tm = IntArray(32); Engine.vdecAny(vae, z, 32, H, W, tm)?.let { px -> Logs.add("  vae on the DSP: product ${tm[24] / 1000} (fill ${tm[29] / 1000}) norm ${tm[27] / 1000} add ${tm[28] / 1000} ms")
                return android.graphics.Bitmap.createBitmap(px, 8 * W, 8 * H, android.graphics.Bitmap.Config.ARGB_8888) }
            throw IllegalStateException("VAE on the DSP failed or its files (vdec.w) are missing")
        }
        /** FLUX.2's latent to RGB (ComfyUI's factors, in the model's batch-normalised space) */
        private val RGBF = floatArrayOf(
            0.0058f, 0.0113f, 0.0073f, 0.0495f, 0.0443f, 0.0836f, -0.0099f, 0.0096f, 0.0644f, 0.2144f, 0.3009f, 0.3652f,
            0.0166f, -0.0039f, -0.0054f, 0.0157f, 0.0103f, -0.0160f, -0.0398f, 0.0902f, -0.0235f, -0.0052f, 0.0095f, 0.0109f,
            -0.3527f, -0.2712f, -0.1666f, -0.0301f, -0.0356f, -0.0180f, -0.0107f, 0.0078f, 0.0013f, 0.0746f, 0.0090f, -0.0941f,
            0.0156f, 0.0169f, 0.0070f, -0.0034f, -0.0040f, -0.0114f, 0.0032f, 0.0181f, 0.0080f, -0.0939f, -0.0008f, 0.0186f,
            0.0018f, 0.0043f, 0.0104f, 0.0284f, 0.0056f, -0.0127f, -0.0024f, -0.0022f, -0.0030f, 0.1207f, -0.0026f, 0.0065f,
            0.0128f, 0.0101f, 0.0142f, 0.0137f, -0.0072f, -0.0007f, 0.0095f, 0.0092f, -0.0059f, 0.0000f, -0.0077f, -0.0049f,
            -0.0465f, -0.0204f, -0.0312f, 0.0095f, 0.0012f, -0.0066f, 0.0290f, -0.0034f, 0.0025f, 0.0220f, 0.0169f, -0.0048f,
            -0.0332f, -0.0457f, -0.0468f, -0.0085f, 0.0389f, 0.0609f, -0.0076f, 0.0003f, -0.0043f, -0.0111f, -0.0460f, -0.0614f)
        /** the live preview from a denoised estimate [128 x gh x gw]: per 32-channel latent cell, stretched to the picture's size */
        fun preview(x0: FloatArray, gh: Int, gw: Int): android.graphics.Bitmap {
            val n = gh * gw; val H = 2 * gh; val W = 2 * gw; val px = IntArray(H * W)
            fun u(f: Float) = ((f + 1f) * 127.5f).toInt().coerceIn(0, 255)
            for (i in 0 until n) for (py in 0..1) for (qx in 0..1) {
                var r = -0.0329f; var g = -0.0718f; var b = -0.0851f
                for (c in 0 until 32) { val v = x0[(c * 4 + 2 * py + qx) * n + i]; r += v * RGBF[c * 3]; g += v * RGBF[c * 3 + 1]; b += v * RGBF[c * 3 + 2] }
                px[(2 * (i / gw) + py) * W + 2 * (i % gw) + qx] = (0xFF shl 24) or (u(r) shl 16) or (u(g) shl 8) or u(b)
            }
            return android.graphics.Bitmap.createScaledBitmap(android.graphics.Bitmap.createBitmap(px, W, H, android.graphics.Bitmap.Config.ARGB_8888), 16 * gw, 16 * gh, true)
        }
        /** MOSS-Image's files on the phone (moss_bundle), by a download or put there */
        fun present(root: File) = File(root, "moss_bundle/engine/moss.txt").isFile && File(root, "moss_bundle/qte/qte_l0.w").isFile
        /** the rows the engine runs: its attention takes keys in chunks of the largest count of 32-row blocks up to 16 that divides
         *  the rows' blocks, and an odd chunk comes out wrong (4192 rows, 131 blocks: 80 % off) -- so the first length from n up
         *  whose chunk is even */
        fun rows(n: Int): Int {
            var l = (n + 63) / 64 * 64
            // a small even chunk fails too: 1088 rows (chunk 2) put one 32-row tile in three 10-17 % off; 1152 (chunk 12) is right.
            // And the softmax takes its blocks four at a time past the last one: 4032 rows (126 blocks, chunk 14) gave noise at 832x1216
            while (true) { val b = l / 32; val c = (16 downTo 1).first { b % it == 0 }; if (b % 4 == 0 && c >= 8) return l; l += 64 }
        }
    }
    private val aux = Engine.mossAux(File(dir, "moss_aux.f16")) ?: error("moss_aux.f16 missing in $dir")
    private val shapes = HashMap<String, Pair<IntArray, Long>>()
    init {
        val js = org.json.JSONObject(File(dir, "moss_aux.json").readText()).getJSONObject("tensors")
        for (k in js.keys()) { val e = js.getJSONObject(k); val sh = e.getJSONArray("shape"); shapes[k] = IntArray(sh.length()) { sh.getInt(it) } to e.getLong("offset") }
    }
    /** a vector (norm weight, bias) as floats */
    private val vecs = HashMap<String, FloatArray>()
    private fun vec(name: String): FloatArray = vecs.getOrPut(name) {
        val (sh, off) = shapes[name] ?: error("no $name"); val n = sh.fold(1) { a, b -> a * b }
        java.io.RandomAccessFile(File(dir, "moss_aux.f16"), "r").use { f ->
            val b = ByteArray(n * 2); f.seek(off); f.readFully(b)
            val bb = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN); FloatArray(n) { half(bb.getShort(it * 2)) } }
    }
    private fun half(h: Short): Float = android.util.Half.toFloat(h)
    /** x [m x K] through the Linear [name] (its bias when there is one) */
    private fun lin(name: String, x: FloatArray, m: Int): FloatArray {
        val (sh, off) = shapes["$name.weight"] ?: error("no $name.weight")
        val boff = shapes["$name.bias"]?.second ?: -1L
        val N = sh[0]; val K = sh[1]
        return aux.linear(off, boff, N, K, x, m)
    }

    private fun rms(x: FloatArray, m: Int, n: Int, w: FloatArray?): FloatArray {
        val o = FloatArray(m * n)
        for (r in 0 until m) { var s = 0.0; for (c in 0 until n) { val v = x[r * n + c]; s += v * v }
            val k = (1.0 / Math.sqrt(s / n + 1e-6)).toFloat(); for (c in 0 until n) o[r * n + c] = x[r * n + c] * k * (w?.get(c) ?: 1f) }
        return o
    }
    private fun sigmoid(v: Float) = 1f / (1f + Math.exp(-v.toDouble()).toFloat())
    private fun situ(h: Float, g: Float) = (4f * Math.tanh(g / 4.0).toFloat() * sigmoid(g)) * (25f * Math.tanh(h / 25.0).toFloat())

    /** RoPE phases [n x 64] (axis * 16 + band) for 4-axis coordinates (r, f, h, w) per row */
    private fun phases(coords: Array<FloatArray>): FloatArray {
        val bands = FloatArray(16) { (1.0 / Math.pow(10000.0, it / 16.0)).toFloat() }
        val p = FloatArray(coords.size * 64)
        for ((i, c) in coords.withIndex()) for (a in 0 until 4) for (b in 0 until 16) p[i * 64 + a * 16 + b] = c[a] * bands[b]
        return p
    }
    /** the picture's rows (h, w) then the text's: the release's _RoPE.forward */
    private fun coords(gh: Int, gw: Int, t: Int): Array<FloatArray> =
        Array(gh * gw + t) { i -> if (i < gh * gw) floatArrayOf(0f, t.toFloat(), i / gw - (gh - 1) / 2f, i % gw - (gw - 1) / 2f) else floatArrayOf(0f, (i - gh * gw).toFloat(), 0f, 0f) }

    /** split-half rotation of each head of q [n x D] in place by phases [n x 64] */
    private fun rotate(q: FloatArray, n: Int, ph: FloatArray) {
        for (r in 0 until n) for (hh in 0 until H) { val o = r * D + hh * DH
            for (d in 0 until 64) { val a = q[o + d]; val b = q[o + d + 64]; val c = Math.cos(ph[r * 64 + d].toDouble()).toFloat(); val s = Math.sin(ph[r * 64 + d].toDouble()).toFloat()
                q[o + d] = a * c - b * s; q[o + d + 64] = b * c + a * s } }
    }

    /** the text refiner: hidden [t x 4096] -> [t x 3072] (text_embedder: proj, then two blocks with RoPE on the text rows alone) */
    fun refine(hidden: FloatArray, t: Int, gh: Int, gw: Int): FloatArray {
        var x = lin("text_embedder.proj", hidden, t)
        val ph = phases(coords(gh, gw, t).copyOfRange(gh * gw, gh * gw + t))
        for (b in 0 until 2) {
            val p = "text_embedder.blocks.$b"
            val qkv = lin("$p.qkv", rms(x, t, D, vec("$p.norm1.weight")), t)
            val q = FloatArray(t * D); val k = FloatArray(t * D); val v = FloatArray(t * D)
            for (r in 0 until t) { System.arraycopy(qkv, r * 3 * D, q, r * D, D); System.arraycopy(qkv, r * 3 * D + D, k, r * D, D); System.arraycopy(qkv, r * 3 * D + 2 * D, v, r * D, D) }
            val qn = vec("$p.q_norm.weight"); val kn = vec("$p.k_norm.weight")
            fun headNorm(a: FloatArray, w: FloatArray) { for (r in 0 until t) for (hh in 0 until H) { val o = r * D + hh * DH; var s = 0.0
                for (d in 0 until DH) s += a[o + d] * a[o + d]; val m = (1.0 / Math.sqrt(s / DH + 1e-6)).toFloat(); for (d in 0 until DH) a[o + d] = a[o + d] * m * w[d] } }
            headNorm(q, qn); headNorm(k, kn); rotate(q, t, ph); rotate(k, t, ph)
            val att = FloatArray(t * D); val sc = FloatArray(t); val scale = 1f / Math.sqrt(DH.toDouble()).toFloat()
            for (hh in 0 until H) for (i in 0 until t) {
                var mx = -Float.MAX_VALUE
                for (j in 0 until t) { var s = 0f; for (d in 0 until DH) s += q[i * D + hh * DH + d] * k[j * D + hh * DH + d]; sc[j] = s * scale; if (sc[j] > mx) mx = sc[j] }
                var z = 0f; for (j in 0 until t) { sc[j] = Math.exp((sc[j] - mx).toDouble()).toFloat(); z += sc[j] }
                for (d in 0 until DH) { var s = 0f; for (j in 0 until t) s += sc[j] * v[j * D + hh * DH + d]; att[i * D + hh * DH + d] = s / z }
            }
            val o = lin("$p.proj", att, t); for (i in x.indices) x[i] += o[i]
            val y = rms(x, t, D, vec("$p.norm2.weight"))
            val up = lin("$p.ffn_up", y, t); val gate = lin("$p.ffn_gate", y, t)
            for (i in up.indices) up[i] = situ(up[i], gate[i])
            val dn = lin("$p.ffn_down", up, t); for (i in x.indices) x[i] += dn[i]
        }
        return x
    }

    /** the time embedding and both modulations for [t]: the blocks' [6 x D] (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f) and
     *  the head's [2 x D] (shift, scale). */
    fun modulations(t: Float): Pair<FloatArray, FloatArray> {
        val half = 128; val e = FloatArray(256)
        for (i in 0 until half) { val ph = t.toDouble() * Math.pow(10000.0, -i.toDouble() / half); e[i] = Math.cos(ph).toFloat(); e[half + i] = Math.sin(ph).toFloat() }
        fun linO(name: String, x: FloatArray): FloatArray = lin(name, x, 1)
        val h1 = linO("time_embedding.0", e); for (i in h1.indices) h1[i] = h1[i] * sigmoid(h1[i])
        val time = linO("time_embedding.2", h1)
        return linO("shared_block_modulation", time) to linO("video_head_modulation", time)
    }

    private var eng: Engine.K2? = null; private var engRows = 0; private var ropeFor = ""
    /** the image blocks' engine for [l] rows (its blocks kept in memory read, the LoRAs built): called ahead, while the text encoder runs,
     *  it takes that time off the first step */
    @Synchronized fun open(l: Int) {
        if (eng != null && engRows == l) return
        eng?.close(); eng = Engine.openK2(dir, l, log = log, set = loraSet) ?: error(Engine.loraError ?: "the engine could not open $dir"); engRows = l
        ropeFor = ""; bufs.forEach { Engine.freeDirect(it) }; bufs.clear(); seqB = buf(l.toLong() * D * 4); cosBB = buf(l.toLong() * 64 * 4); sinBB = buf(l.toLong() * 64 * 4); tvB = buf(6L * D * 4)
    }
    /** a LoRA set (LoraSet's file) the engine runs as it goes: the turbo's pair, and the user's own after it */
    var loraSet: File? = null
    private val bufs = ArrayList<java.nio.ByteBuffer>()
    private fun buf(bytes: Long) = Engine.direct(bytes).also { bufs.add(it) }
    /** the velocity for latent x [128 x gh x gw] at time [t] with the refined text [txt] (t rows); [mods] from [modulations] */
    fun velocity(x: FloatArray, gh: Int, gw: Int, txt: FloatArray, t: Int, mods: Pair<FloatArray, FloatArray>, onBlock: (Int) -> Unit = {},
                 rowsFor: Int = rows(gh * gw + t)): FloatArray {
        // [rowsFor]: the engine's length -- for CFG the longer text's, so both passes share one engine; the rows past this text are masked
        val n = gh * gw; val l = rowsFor
        open(l)
        val tok = FloatArray(n * C); for (c in 0 until C) for (i in 0 until n) tok[i * C + c] = x[c * n + i]
        val vid = lin("video_embedder", tok, n)
        val seq = seqB!!.asFloatBuffer(); seq.put(vid); seq.put(txt, 0, t * D); while (seq.hasRemaining()) seq.put(0f); seq.rewind()
        // the RoPE tables are the same every step: filled once per size and text length
        val ropeKey = "$gh x $gw x $t x $l"
        if (ropeKey != ropeFor) {
            val ph = phases(coords(gh, gw, t))
            val cosB = cosBB!!.asFloatBuffer(); val sinB = sinBB!!.asFloatBuffer()
            for (i in 0 until (n + t) * 64) { cosB.put(Math.cos(ph[i].toDouble()).toFloat()); sinB.put(Math.sin(ph[i].toDouble()).toFloat()) }
            while (cosB.hasRemaining()) { cosB.put(1f); sinB.put(0f) }
            ropeFor = ropeKey
        }
        val cosB = cosBB!!.asFloatBuffer(); val sinB = sinBB!!.asFloatBuffer()
        val m = mods.first; val tv = tvB!!.asFloatBuffer()
        for (k in intArrayOf(1, 0, 2, 4, 3, 5)) tv.put(m, k * D, D); tv.rewind()          // the engine's order: scale, shift, gate (pre, post)
        Engine.withProgress({ eng!!.progress() }, 42, onBlock) { eng!!.step(seq, tv, cosB, sinB, n + t, l, 0, 0) }
        val out = FloatArray(n * D); seq.rewind(); seq.get(out)
        val hd = mods.second; val nw = vec("final_norm_video.weight")
        val y = rms(out, n, D, nw); for (i in 0 until n) for (c in 0 until D) y[i * D + c] = y[i * D + c] * (1f + hd[D + c]) + hd[c]
        val v = lin("final_linear_video", y, n)
        val r = FloatArray(C * n); for (i in 0 until n) for (c in 0 until C) r[c * n + i] = v[i * C + c]
        return r
    }

    private var seqB: java.nio.ByteBuffer? = null; private var cosBB: java.nio.ByteBuffer? = null; private var sinBB: java.nio.ByteBuffer? = null; private var tvB: java.nio.ByteBuffer? = null

    /**
     * MOSS-Image as released (SFT or a fine-tune), flow matching x = (1 - s) x0 + s z with the model's output
     * v = z - x0 at time s * 1000, through the shared samplers ([FlowSampler], the denoised estimate x - s v) on [sig] (Sched.sigmas).
     * CFG above 1 runs the negative text too (its own hidden state [negHidden], [negT] rows): v = vu + cfg (v - vu).
     */
    fun generateFlow(hidden: FloatArray, t: Int, negHidden: FloatArray?, negT: Int, gh: Int, gw: Int, seed: Long, sig: DoubleArray, sampler: String,
                     cfg: Float, stop: () -> Boolean = { false }, onPreview: (FloatArray) -> Unit = {}, onStep: (Int, Int) -> Unit = { _, _ -> }): FloatArray {
        val n = gh * gw; val useCfg = cfg > 1.0001f && negHidden != null
        var t0 = System.currentTimeMillis()
        val txt = refine(hidden, t, gh, gw); val ntxt = if (useCfg) refine(negHidden!!, negT, gh, gw) else null
        log("  text refiner · ${(System.currentTimeMillis() - t0) / 1000.0}s")
        val l = rows(n + maxOf(t, if (useCfg) negT else 0))
        val rng = java.util.Random(seed)
        val x = FloatArray(C * n) { (rng.nextGaussian() * sig[0]).toFloat() }
        val fs = FlowSampler(sampler, sig, 0) { FloatArray(C * n) { rng.nextGaussian().toFloat() } }
        var x0 = x
        for (i in 0 until sig.size - 1) {
            if (stop()) break
            t0 = System.currentTimeMillis()
            val s = sig[i].toFloat(); val mods = modulations(s * 1000f)
            val v = velocity(x, gh, gw, txt, t, mods, { b -> onStep(i, if (useCfg) b / 2 else b) }, l)
            if (useCfg) { val vu = velocity(x, gh, gw, ntxt!!, negT, mods, { b -> onStep(i, 21 + b / 2) }, l)
                for (k in v.indices) v[k] = vu[k] + cfg * (v[k] - vu[k]) }
            val den = FloatArray(C * n) { x[it] - s * v[it] }
            fs.step(x, den, i); x0 = if (i + 1 < sig.size - 1) den else x
            if (i + 1 < sig.size - 1) onPreview(den)
            log("  step ${i + 1}: ${(System.currentTimeMillis() - t0) / 1000.0}s" + Logs.stats(x))
        }
        return x0
    }

    override fun close() { eng?.close(); eng = null; bufs.forEach { Engine.freeDirect(it) }; bufs.clear(); aux.close() }
}

/**
 * MOSS-Image's text side kept between pictures: the Qwen3.5 tokenizer (250k entries, seconds to read) loaded from the moment the model
 * is chosen ([warm]), and the last prompt's hidden state, so the same prompt again costs nothing.
 */
object MossText {
    private var tokDir: File? = null; private var tok: QwenTok? = null; private var loading: Thread? = null
    private var lastKey = ""; private var last: FloatArray? = null
    /** starts reading the tokenizer in the background (the model was just chosen) */
    @Synchronized fun warm(dir: File) {
        if (tokDir == dir && (tok != null || loading != null) || !File(dir, "qwen_vocab.txt").isFile) return
        tokDir = dir; tok = null
        loading = Thread { val t = runCatching { QwenTok(dir) }.getOrNull(); synchronized(this) { if (tokDir == dir) tok = t; loading = null } }.apply { isDaemon = true; start() }
    }
    private fun tokenizer(dir: File): QwenTok {
        warm(dir); loading?.join()
        return synchronized(this) { tok } ?: QwenTok(dir).also { synchronized(this) { tok = it } }
    }
    fun ids(dir: File, prompt: String): IntArray = MossGen.promptIds(tokenizer(dir), prompt)
    /** the encoder's output for [ids] (the same ids and folder as last time: the kept one) */
    fun hidden(qteDir: File, ids: IntArray, log: (String) -> Unit = {}): Pair<FloatArray, Int> {
        val key = qteDir.absolutePath + ":" + ids.joinToString(",")
        synchronized(this) { if (key == lastKey) last?.let { log("  text encoder: the same prompt, kept"); return it to ids.size } }
        val t0 = System.currentTimeMillis()
        val h = (Engine.openQte(qteDir) ?: error("the text encoder could not open $qteDir")).use { it.encode(ids) }
        log("  text encoder: ${ids.size} tokens · ${(System.currentTimeMillis() - t0) / 1000.0}s")
        synchronized(this) { lastKey = key; last = h }
        return h to ids.size
    }
}
