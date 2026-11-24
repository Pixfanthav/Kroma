package dev.kroma.open

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.io.File
import java.nio.LongBuffer
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * SDXL on the engine: sdxl_bundle/ holds unet/ (unet.tab / .prm / .w, emb.f32 and the time and label Linears), vae/
 * (xvae.w / .prm, the fp16-fix VAE), te/ (clip_l.onnx, clip_g.onnx + clip_g.data, on the CPU) and tok/
 * (clip_vocab.json, clip_merges.txt). The text and its y as ComfyUI's encode makes them (checked against ComfyUI's: ctx rel 9e-7), the UNet's
 * Euler loop and the VAE on the DSP (xl_gen / xl_vae). Width 1024 only: the engine's last level needs it a multiple of 32.
 */
/** [custom]: a fine-tune's engine folder (Custom, kconv's: the UNet's files and te/ with its own text encoder weights), else the built-in */
class SdxlGen(private val root: File, private val log: (String) -> Unit, custom: File? = null) {
    private val b = File(root, "sdxl_bundle")
    internal val unet = custom ?: File(b, "unet")
    private val te = custom?.let { File(it, "te") }?.takeIf { File(it, "clip_g.onnx").isFile } ?: File(b, "te")
    private val env = OrtEnvironment.getEnvironment()

    companion object {
        const val W = 1024; const val CTX = 96   // one chunk's rows (77 tokens, 32 up); long prompts have more: encode
        fun present(root: File) = File(root, "sdxl_bundle/unet/unet.w").isFile && File(root, "sdxl_bundle/vae/xvae.w").isFile &&
            File(root, "sdxl_bundle/te/clip_g.onnx").isFile && File(root, "sdxl_bundle/tok/clip_vocab.json").isFile
        /** the height for a ratio's (w, h), the width held at 1024 (wide ratios come out square) */
        fun height(w: Int, h: Int) = ((W.toDouble() * maxOf(h, w) / w / 32).roundToInt() * 32).coerceIn(W, 1536)
        /** ComfyUI's sdxl size embedding: height, width, crop top, crop left, target height, target width, 256 (cos, sin) each */
        fun sizeEmbed(h: Int, w: Int): FloatArray {
            val out = FloatArray(1536); val v = intArrayOf(h, w, 0, 0, h, w)
            for (k in 0 until 6) for (i in 0 until 128) { val a = v[k] * exp(-ln(10000.0) * i / 128); out[k * 256 + i] = cos(a).toFloat(); out[k * 256 + 128 + i] = sin(a).toFloat() }
            return out
        }
    }

    private fun opts() = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4); setMemoryPatternOptimization(false); setCPUArenaAllocator(false)
        addConfigEntry("session.disable_prepacking", "1") }

    /** the cond and uncond texts [2 x C x 2048] (L's 768 then G's 1280 a token) and their y [2 x 2816] (G's pooled, the size embedding).
     *  A long prompt goes in chunks of 75 tokens as ComfyUI does (each with its own start and end, at most four): k chunks are rows 77 j ..,
     *  C = 77 k rounded up to 32 (96 for one), the rest zero (masked by the engine). When the two texts have a different number of chunks the
     *  shorter one's repeat (the same keys again leave the attention as it was, as ComfyUI's repeat to a common length). Pooled: the first chunk's. */
    fun encode(prompt: String, negative: String, h: Int, w: Int = W, stop: () -> Boolean = { false }): Pair<FloatArray, FloatArray> {
        val tok = ClipTok(File(b, "tok")); val y = FloatArray(2 * 2816); val se = sizeEmbed(h, w)
        val texts = listOf(prompt, negative); val cl = texts.map { tok.chunks77(it, ClipTok.END) }; val cg = texts.map { tok.chunks77(it, 0) }
        val k = cl.maxOf { it.size }; val C = (77 * k + 31) / 32 * 32; val ctx = FloatArray(2 * C * 2048)
        val t0 = System.currentTimeMillis()
        env.createSession(File(te, "clip_l.onnx").absolutePath, opts()).use { l ->
            for (c in 0..1) for (q in 0 until k) {
                if (stop()) throw InterruptedException("stopped"); val ids = cl[c][q % cl[c].size]
                OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, 77)).use { t -> l.run(mapOf("ids" to t)).use { r ->
                    val hb = (r[0] as OnnxTensor).floatBuffer
                    for (i in 0 until 77) for (j in 0 until 768) ctx[(c * C + q * 77 + i) * 2048 + j] = hb.get(i * 768 + j) } }
            }
        }
        env.createSession(File(te, "clip_g.onnx").absolutePath, opts()).use { g ->
            for (c in 0..1) for (q in 0 until k) {
                if (stop()) throw InterruptedException("stopped"); val ids = cg[c][q % cg[c].size]
                OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, 77)).use { t -> g.run(mapOf("ids" to t)).use { r ->
                    val hb = (r[0] as OnnxTensor).floatBuffer
                    for (i in 0 until 77) for (j in 0 until 1280) ctx[(c * C + q * 77 + i) * 2048 + 768 + j] = hb.get(i * 1280 + j)
                    if (q == 0) { val pb = (r[1] as OnnxTensor).floatBuffer; for (j in 0 until 1280) y[c * 2816 + j] = pb.get(j); System.arraycopy(se, 0, y, c * 2816 + 1280, 1536) } } }
            }
        }
        log("  text clip L+G ${(System.currentTimeMillis() - t0) / 1000.0}s" + if (k > 1) " · $k chunks of 75 tokens" else "")
        return ctx to y
    }

    private var xlErr = 0
    /** the latent [4 x h/8 x 1024/8] for a seed (a standard normal noise), steps of Euler at cfg; step(i, n) reports and returns true to stop */
    fun generate(ctx: FloatArray, y: FloatArray, h: Int, seed: Long, steps: Int, cfg: Float, w: Int = W, unit: ((Int, Int, Int) -> Unit)? = null,
                 step: (Int, Int) -> Boolean): FloatArray? {
        val lh = h / 8; val lw = w / 8; val rnd = java.util.Random(seed); val z = FloatArray(4 * lh * lw) { rnd.nextGaussian().toFloat() }
        // the pieces of each step polled while the UNet runs (Engine.xlProgress), for a bar that moves inside a step as Krea's and Anima's do
        val poll = unit?.let { u -> Thread { var last = -1
            try { while (true) { val (s, d, n) = Engine.xlProgress(); val k = s * 1024 + d; if (n > 0 && k != last) { last = k; u(s, d, n) }; Thread.sleep(100) } } catch (_: InterruptedException) {}
        }.apply { isDaemon = true; start() } }
        // a refusal before the first step is tried again as the engines' opens are (Engine.dspRetry): the same latent, the noise kept
        val z0 = z.copyOf(); var stepped = false
        val err = try { Engine.dspRetry("SDXL UNet") { System.arraycopy(z0, 0, z, 0, z.size)
                val e = Engine.xlGen(unet.absolutePath, ctx, y, z, lh, lw, steps, cfg) { i, n -> stepped = true; step(i, n) }
                if (e == 0 || e == -99 || stepped) { xlErr = e; 1L } else { xlErr = e; Logs.add("  SDXL UNet: ${Engine.why(e)}"); 0L } }.let { xlErr }
            } finally { poll?.interrupt(); poll?.join(500) }
        if (err == -99) return null
        require(err == 0) { "the SDXL UNet stopped (${Engine.why(err)})" }
        return z
    }

    /** the picture from the latent: z / 0.13025 through the VAE on the DSP, -1..1 to 0..255 */
    fun decode(z: FloatArray, h: Int, w: Int = W): Bitmap {
        val lh = h / 8; val lw = w / 8; val zs = FloatArray(z.size) { z[it] / 0.13025f }; val rgb = FloatArray(3 * h * w)
        val err = Engine.xlVae(File(b, "vae").absolutePath, zs, lh, lw, rgb); require(err == 0) { "the SDXL VAE stopped (${Engine.why(err)})" }
        val px = IntArray(h * w); val n = h * w
        fun c8(v: Float) = ((v.coerceIn(-1f, 1f) + 1f) * 127.5f).roundToInt()
        for (i in 0 until n) px[i] = (0xFF shl 24) or (c8(rgb[i]) shl 16) or (c8(rgb[n + i]) shl 8) or c8(rgb[2 * n + i])
        return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
    }
}
