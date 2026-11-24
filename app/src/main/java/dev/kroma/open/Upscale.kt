package dev.kroma.open

import android.graphics.Bitmap
import java.io.File
import kotlin.math.roundToInt

/** 2x upscaling: Real-ESRGAN (realesr-general-x4v3) on the DSP over large tiles, its 4x averaged down to 2x. */
object Upscale {
    private const val T = 1024     // the largest tile a side, for the memory of one DSP call
    private const val M = 16       // tile margin thrown away (edge effects of the convs)

    fun esrgan2x(src: Bitmap, dir: File, stop: () -> Boolean, prog: (Int, Int) -> Unit): Bitmap {
        val W = src.width; val H = src.height
        val px = IntArray(W * H); src.getPixels(px, 0, W, 0, 0, W, H)
        val W2 = W * 2; val H2 = H * 2
        val out = IntArray(W2 * H2)
        // stride T-2M; the last tile is pinned to the edge, so the kept centres tile the image with no gap
        fun starts(n: Int, t: Int): List<Int> {
            val s = ArrayList<Int>(); var p = 0
            while (true) { if (p + t >= n) { s.add(n - t); break }; s.add(p); p += t - 2 * M }
            return s
        }
        val Tw = minOf(T, W); val Th = minOf(T, H)
        val xs = starts(W, Tw); val ys = starts(H, Th)
        val inp = FloatArray(3 * Th * Tw); val Ow = 4 * Tw; val Oh = 4 * Th
        var done = 0; val total = xs.size * ys.size
        for (ty in ys) for (tx in xs) {
            if (stop()) throw InterruptedException()
            for (c in 0 until 3) { val sh = 16 - 8 * c
                for (y in 0 until Th) for (x in 0 until Tw) inp[(c * Th + y) * Tw + x] = ((px[(ty + y) * W + tx + x] shr sh) and 0xFF) / 255f }
            val o = Engine.esr(dir, inp, Th, Tw) ?: throw IllegalStateException("upscaler on the DSP failed or its files are missing")
            // kept region in source pixels (margins dropped except at the image border)
            val vx0 = if (tx == 0) 0 else tx + M; val vx1 = if (tx + Tw >= W) W else tx + Tw - M
            val vy0 = if (ty == 0) 0 else ty + M; val vy1 = if (ty + Th >= H) H else ty + Th - M
            for (oy in 2 * vy0 until 2 * vy1) for (ox in 2 * vx0 until 2 * vx1) {
                val ly = 2 * oy - 4 * ty; val lx = 2 * ox - 4 * tx     // 4x tile coords of the 2x2 block
                var argb = 0xFF shl 24
                for (c in 0 until 3) {
                    val base = c * Oh * Ow
                    val v = (o[base + ly * Ow + lx] + o[base + ly * Ow + lx + 1] +
                             o[base + (ly + 1) * Ow + lx] + o[base + (ly + 1) * Ow + lx + 1]) * 0.25f
                    argb = argb or ((v * 255f).roundToInt().coerceIn(0, 255) shl (16 - 8 * c))
                }
                out[oy * W2 + ox] = argb
            }
            prog(++done, total)
        }
        return Bitmap.createBitmap(out, W2, H2, Bitmap.Config.ARGB_8888)
    }
}
