package dev.kroma.open

import java.io.File

/**
 * FLUX.2 Klein on the engine: klein_bundle/ holds engine/ (kconv's blocks and moss_aux), kte/ (the text encoder on the DSP, the inputs of its
 * layers 9, 18, 27), and shares MOSS-Image's VAE format ([MossGen.decode]). The rows the engine runs: the text (padded), the image, spare rows.
 */
class KleinGen(private val dir: File, private val log: (String) -> Unit = {}) : AutoCloseable {
    companion object {
        /** a picture as a reference latent: scaled to about [area] pixels (sides in 16s), through FLUX.2's VAE encoder on the CPU (vae/fvae_enc.onnx,
         *  the mean), then 2 x 2 patched to 128 channels and batch-normalised (vae_bn.json) as the model's latents are */
        fun encodeRef(src: android.graphics.Bitmap, vae: File, area: Int = 1024 * 1024, log: (String) -> Unit = {}): Ref? {
            // on the DSP when the bundle has the DSP encoder (about 2 s at 1 MP; its width in 256s), else on the CPU through ONNX (about 20 s at 0.6 MP)
            val dsp = File(vae, "fvenc.w").isFile; val onnx = File(vae, "fvae_enc.onnx"); if (!dsp && !onnx.isFile) { log("reference: no encoder (vae/fvenc.w or fvae_enc.onnx)"); return null }
            val t0 = System.currentTimeMillis(); val s = Math.sqrt((if (dsp) area else 768 * 768).toDouble() / (src.width.toDouble() * src.height)).coerceAtMost(1.0)
            val W = if (dsp) maxOf(256, Math.round(src.width * s / 256).toInt() * 256) else maxOf(16, (src.width * s / 16).toInt() * 16)
            val H = if (dsp) maxOf(16, Math.round(src.height * (W.toDouble() / src.width) / 16).toInt() * 16) else maxOf(16, (src.height * s / 16).toInt() * 16)
            val bm = android.graphics.Bitmap.createScaledBitmap(src, W, H, true); val px = IntArray(W * H); bm.getPixels(px, 0, W, 0, 0, W, H)
            val inp = FloatArray(3 * H * W); for (i in 0 until H * W) { val v = px[i]; inp[i] = ((v shr 16) and 255) / 127.5f - 1f; inp[H * W + i] = ((v shr 8) and 255) / 127.5f - 1f; inp[2 * H * W + i] = (v and 255) / 127.5f - 1f }
            val z = if (dsp) FloatArray(32 * (H / 8) * (W / 8)).takeIf { Engine.fvEnc(vae.absolutePath, inp, H, W, it) == 0 } ?: run { log("reference: the DSP encoder failed"); return null }
            else { val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
            env.createSession(onnx.absolutePath, ai.onnxruntime.OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }).use { se ->
                ai.onnxruntime.OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(inp), longArrayOf(1, 3, H.toLong(), W.toLong())).use { t ->
                    se.run(mapOf("px" to t)).use { r -> val b = (r[0] as ai.onnxruntime.OnnxTensor).floatBuffer; FloatArray(b.remaining()).also { b.get(it) } } } } }
            val bn = org.json.JSONObject(File(vae, "vae_bn.json").readText()); val mean = bn.getJSONArray("mean"); val std = bn.getJSONArray("std")
            val lh = H / 8; val lw = W / 8; val rh = lh / 2; val rw = lw / 2; val n = rh * rw; val lat = FloatArray(128 * n)
            for (c in 0 until 128) { val mu = mean.getDouble(c).toFloat(); val sd = std.getDouble(c).toFloat(); val c32 = c / 4; val py = (c / 2) % 2; val qx = c % 2
                for (i in 0 until n) { val y = i / rw; val x = i % rw; lat[c * n + i] = (z[(c32 * lh + 2 * y + py) * lw + 2 * x + qx] - mu) / sd } }
            log("  reference ${W}x$H encoded on the ${if (dsp) "DSP" else "CPU"} · ${System.currentTimeMillis() - t0} ms")
            return Ref(lat, rh, rw)
        }
        const val C = 128
        /** [bundle]: klein_bundle (4B), klein9_bundle (9B) */
        fun present(root: File, bundle: String = "klein_bundle") = File(root, "$bundle/engine/klein.txt").isFile && File(root, "$bundle/kte/kte_l0.w").isFile
        /** the chat template with thinking off, as the release gives it to its encoder */
        fun promptIds(tok: QwenTok, prompt: String): IntArray {
            val r = ArrayList<Int>()
            fun add(s: String) { tok.encode(s).forEach { r.add(it.toInt()) } }
            r.add(151644); add("user\n$prompt"); r.add(151645); add("\n"); r.add(151644); add("assistant\n"); r.add(151667); add("\n\n"); r.add(151668); add("\n\n")
            return r.toIntArray()
        }
        /** the release's flow schedule for this size and step count */
        fun sigmas(n: Int, steps: Int): DoubleArray {
            val a1 = 8.73809524e-05; val b1 = 1.89833333; val a2 = 0.00016927; val b2 = 0.45666666
            val mu = if (n > 4300) a2 * n + b2 else { val m200 = a2 * n + b2; val m10 = a1 * n + b1; val a = (m200 - m10) / 190.0
                val b = m200 - 200.0 * a; if (steps > 200) m200 else a * steps + b }
            val e = Math.exp(mu)
            return DoubleArray(steps + 1) { i -> if (i == steps) 0.0 else { val s = 1.0 - i * (1.0 - 1.0 / steps) / maxOf(steps - 1, 1); e / (e + (1.0 / s - 1.0)) } }
        }
    }
    /** the blocks' width: 4B 3072, 9B 4096 */
    private val D = File(dir, "klein.txt").readText().trim().split(Regex("\\s+")).getOrNull(3)?.toIntOrNull() ?: 3072
    val blocks: Int get() = org.json.JSONObject(File(dir, "moss_aux.json").readText()).getInt("blocks")
    private val aux = Engine.mossAux(File(dir, "moss_aux.f16")) ?: error("moss_aux.f16 missing in $dir")
    private val shapes = HashMap<String, Pair<IntArray, Long>>()
    init {
        val js = org.json.JSONObject(File(dir, "moss_aux.json").readText()).getJSONObject("tensors")
        for (k in js.keys()) { val e = js.getJSONObject(k); val sh = e.getJSONArray("shape"); shapes[k] = IntArray(sh.length()) { sh.getInt(it) } to e.getLong("offset") }
    }
    private fun lin(name: String, x: FloatArray, m: Int): FloatArray {
        val (sh, off) = shapes["$name.weight"] ?: error("no $name"); val N = sh[0]; val K = sh[1]
        val y = aux.linear(off, -1L, N, K, x, m)
        for ((a, b, st) in outer(name)) {      // y += strength (x A^T) B^T
            val R = a.size / K
            for (r in 0 until m) { val h = FloatArray(R) { q -> var s = 0f; for (c in 0 until K) s += a[q * K + c] * x[r * K + c]; s * st }
                for (n in 0 until N) { var s = 0f; for (q in 0 until R) s += b[n * R + q] * h[q]; y[r * N + n] += s } }
        }
        return y
    }
    /** a LoRA set's pairs for one Linear outside the blocks (LoraImport.convertKlein): A [R x K], B [N x R], the strength */
    private val outers = HashMap<String, List<Triple<FloatArray, FloatArray, Float>>>()
    private fun outer(name: String) = outers.getOrPut(name) {
        val set = loraSet ?: return@getOrPut emptyList()
        fun f16(f: File): FloatArray { val bb = java.nio.ByteBuffer.wrap(f.readBytes()).order(java.nio.ByteOrder.LITTLE_ENDIAN).asShortBuffer(); return FloatArray(bb.limit()) { android.util.Half.toFloat(bb.get(it)) } }
        set.readLines().mapNotNull { l -> val p = l.split('\t'); val d = File(p[0]); val a = File(d, "outer.$name.A.f16"); val b = File(d, "outer.$name.B.f16")
            if (a.isFile && b.isFile) Triple(f16(a), f16(b), p.getOrNull(1)?.toFloatOrNull() ?: 1f) else null }
    }
    private fun silu(x: FloatArray) = FloatArray(x.size) { val v = x[it]; (v / (1.0 + Math.exp(-v.toDouble()))).toFloat() }

    /** a LoRA set (LoraSet's file) the engine runs as it goes */
    var loraSet: File? = null
    private var eng: Engine.K2? = null; private val bufs = ArrayList<java.nio.ByteBuffer>()
    private fun buf(bytes: Long) = Engine.direct(bytes).also { bufs.add(it) }

    /** the text encoder's three layers for [ids] -> context [S x 3 hidden] */
    fun context(kte: File, ids: IntArray): FloatArray {
        val hid = File(kte, "kte.txt").readText().trim().split(Regex("\\s+"))[0].toInt()
        val out = (Engine.openKte(kte) ?: error("the text encoder could not open $kte")).use { it.encode(File(kte, "embed.f16"), ids, 3, hid) }
        val S = ids.size; val c = FloatArray(S * 3 * hid)
        for (s in 0 until S) for (l in 0 until 3) System.arraycopy(out, (l * S + s) * hid, c, (s * 3 + l) * hid, hid)
        return c
    }

    /** A reference picture for editing (FLUX.2's way, ComfyUI's "index" method): its latent [128 x rh x rw] (encodeRef) goes in as more image
     *  tokens after the picture's, RoPE time index 10 for the first, 20 for the next, its own rows and columns from 0; the result is the picture's
     *  tokens only. They take the image stream's modulations and the same timestep. */
    class Ref(val lat: FloatArray, val rh: Int, val rw: Int)
    /** [ctx] [S x 7680] -> the denoised latent [128 x gh x gw] (the VAE's batch-normalised, 2 x 2 patched space). With [cfg] > 1 and a
     *  negative context [negCtx] [negS x 7680] each step runs the blocks twice, v = v_neg + cfg (v - v_neg) (a fine-tune that is not
     *  distilled); the built-in models are distilled to one pass (cfg 1). */
    fun generate(ctx: FloatArray, S: Int, gh: Int, gw: Int, seed: Long, steps: Int, stop: () -> Boolean, onPreview: (FloatArray) -> Unit,
                 refs: List<Ref> = emptyList(), negCtx: FloatArray? = null, negS: Int = 0, cfg: Float = 1f, onStep: (Int, Int) -> Unit): FloatArray {
        val guided = cfg > 1.0001f && negCtx != null; val Sm = if (guided) maxOf(S, negS) else S
        val N = gh * gw; val NR = refs.sumOf { it.rh * it.rw }; val N32 = (N + NR + 31) / 32 * 32; val S32 = (Sm + 31) / 32 * 32
        val L = MossGen.rows(S32 + N32); val T = L - N32                  // what the rows' rule adds goes to the text's padding
        val txt = lin("context_embedder", ctx, S); val ntxt = if (guided) lin("context_embedder", negCtx!!, negS) else null
        eng = Engine.openK2(dir, L, log = log, set = loraSet) ?: error(Engine.loraError ?: "the engine could not open $dir")
        val nb = blocks
        val seqB = buf(L.toLong() * D * 4); val cosB = buf(L.toLong() * 64 * 4); val sinB = buf(L.toLong() * 64 * 4); val tvB = buf(15L * D * 4)
        // RoPE: 4 axes (t, h, w, l) of 16 bands at theta 2000; the text along l, the image along h and w
        val bands = DoubleArray(16) { 1.0 / Math.pow(2000.0, 2.0 * it / 32) }
        val cosF = cosB.asFloatBuffer(); val sinF = sinB.asFloatBuffer()
        for (r in 0 until L) {
            val cell = r - T; var tq = 0; var h = 0; var w = 0
            if (cell in 0 until N) { h = cell / gw; w = cell % gw }
            else if (cell in N until N + NR) { var q = cell - N; for ((k, f) in refs.withIndex()) { if (q < f.rh * f.rw) { tq = 10 * (k + 1); h = q / f.rw; w = q % f.rw; break }; q -= f.rh * f.rw } }
            val l = if (r < Sm) r else 0
            for (a in 0 until 4) for (b in 0 until 16) { val ph = (when (a) { 0 -> tq; 1 -> h; 2 -> w; else -> l }) * bands[b]; cosF.put(Math.cos(ph).toFloat()); sinF.put(Math.sin(ph).toFloat()) }
        }
        // the references' tokens through x_embedder once (they do not change between steps)
        val refEmb = if (NR == 0) FloatArray(0) else FloatArray(NR * D).also { out -> var at = 0
            for (f in refs) { val n = f.rh * f.rw; val tok = FloatArray(n * C); for (c in 0 until C) for (p in 0 until n) tok[p * C + c] = f.lat[c * n + p]
                System.arraycopy(lin("x_embedder", tok, n), 0, out, at * D, n * D); at += n } }
        val sig = sigmas(N, steps); val rng = java.util.Random(seed)
        val x = FloatArray(C * N) { rng.nextGaussian().toFloat() }; var x0 = x
        for (i in 0 until steps) {
            if (stop()) break
            val t0 = System.currentTimeMillis()
            // the time: 256 dims, then the modulations (their order shift, scale, gate; the engine's scale, shift, gate)
            val tt = sig[i] * 1000.0; val e = FloatArray(256)
            for (k in 0 until 128) { val ph = tt * Math.exp(-Math.log(10000.0) * k / 128); e[k] = Math.cos(ph).toFloat(); e[128 + k] = Math.sin(ph).toFloat() }
            val temb = lin("time_guidance_embed.timestep_embedder.linear_2", silu(lin("time_guidance_embed.timestep_embedder.linear_1", e, 1)), 1); val st = silu(temb)
            val mi = lin("double_stream_modulation_img.linear", st, 1); val mt = lin("double_stream_modulation_txt.linear", st, 1); val ms = lin("single_stream_modulation.linear", st, 1)
            val tv = tvB.asFloatBuffer()
            for (k in intArrayOf(1, 0, 2, 4, 3, 5)) tv.put(mt, k * D, D)
            for (k in intArrayOf(1, 0, 2, 4, 3, 5)) tv.put(mi, k * D, D)
            for (k in intArrayOf(1, 0, 2)) tv.put(ms, k * D, D)
            tv.rewind()
            val tok = FloatArray(N * C); for (c in 0 until C) for (p in 0 until N) tok[p * C + c] = x[c * N + p]
            val vid = lin("x_embedder", tok, N); val hd = lin("norm_out.linear", st, 1); var tEng = 0L
            // one pass of the blocks for a text (its embedding te, Se real rows): the velocity [N x C]; half of the bar each when guided
            fun pass(te: FloatArray, Se: Int, half: Int): FloatArray {
                val seq = seqB.asFloatBuffer(); seq.put(te, 0, Se * D); for (k in 0 until (T - Se) * D) seq.put(0f); seq.put(vid); seq.put(refEmb); while (seq.hasRemaining()) seq.put(0f); seq.rewind()
                val e0 = System.currentTimeMillis()
                Engine.withProgress({ eng!!.progress() }, nb, { b -> onStep(i, if (guided) (half * nb + b) / 2 else b) }) { eng!!.step(seq, tv, cosB.asFloatBuffer(), sinB.asFloatBuffer(), Se, T, T + N + NR, L) }
                tEng += System.currentTimeMillis() - e0
                val out = FloatArray(N * D); seq.position(T * D); seq.get(out); seq.rewind()
                // the last layer: a plain LayerNorm with the time's scale and shift
                java.util.stream.IntStream.range(0, N).parallel().forEach { r -> var m = 0.0; var q = 0.0; for (c in 0 until D) { val v = out[r * D + c].toDouble(); m += v; q += v * v }; m /= D; q = q / D - m * m
                    val k = (1.0 / Math.sqrt(q + 1e-6)).toFloat(); val mf = m.toFloat(); for (c in 0 until D) out[r * D + c] = (out[r * D + c] - mf) * k * (1f + hd[c]) + hd[D + c] }
                return lin("proj_out", out, N)
            }
            val v = pass(txt, S, 0)
            if (guided) { val vn = pass(ntxt!!, negS, 1); for (j in v.indices) v[j] = vn[j] + cfg * (v[j] - vn[j]) }
            onStep(i, nb)
            val dt = (sig[i + 1] - sig[i]).toFloat(); val s = sig[i].toFloat(); val den = FloatArray(C * N)
            for (c in 0 until C) for (p in 0 until N) { val vv = v[p * C + c]; den[c * N + p] = x[c * N + p] - s * vv; x[c * N + p] += dt * vv }
            x0 = if (i + 1 < steps) den else x
            if (i + 1 < steps) onPreview(den)
            log("  step ${i + 1}: ${(System.currentTimeMillis() - t0) / 1000.0}s, blocks ${tEng / 1000.0}s" + if (guided) " (cfg $cfg, 2 passes)" else "")
        }
        return x0
    }

    override fun close() { eng?.close(); eng = null; bufs.forEach { Engine.freeDirect(it) }; bufs.clear(); aux.close() }
}
