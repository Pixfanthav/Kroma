package dev.kroma.open

import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Anima's adaLN modulations on the CPU, for bundles whose parts take them as inputs (b<i>_mod, final_mod).
 *
 * Every block scales, shifts and gates its three branches by vectors that depend on the timestep alone: two small Linears on
 * the timestep embedding. Half of the Anima LoRAs in the wild train those Linears too, and the NPU parts have no slots for
 * them. Computed here they cost the NPU nothing and a LoRA on them is exact, at any rank. The weights (adaln.f16, about
 * 390MB, with adaln.json naming each tensor) are mapped and read one modulation at a time; all the steps of a picture are
 * computed in one go, so each weight is converted from half floats once.
 *
 *   s     = [cos, sin](t * exp(-ln(10000) * k / (D/2)))            k < D/2
 *   alora = T2 silu(T1 s)                                          [3D]
 *   temb  = s * rsqrt(mean(s^2) + 1e-6) * w_norm                   [D]
 *   block i, modulation j:  W2 (W1 silu(temb)) + alora             [3D] = shift | scale | gate
 *   final:                  F2 (F1 silu(temb)) + alora[:2D]        [2D]
 * With a LoRA (A [r,in], B [out,r], alpha/rank already in B) at strength g a Linear is W x + g B (A x).
 */
class Adaln(private val dir: File, private val lora: File? = null, private val strength: Float = 1f) {
    private val idx = JSONObject(File(dir, "adaln.json").readText())
    private val ts = idx.getJSONObject("tensors")
    val d = idx.getInt("D")
    private val map: ShortBuffer = RandomAccessFile(File(dir, "adaln.f16"), "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
    val blocks = generateSequence(0) { it + 1 }.takeWhile { ts.has("b$it.0.1.weight") }.count()

    private fun has(n: String) = ts.has(n)
    private fun tensor(n: String): FloatArray {
        val o = ts.getJSONObject(n); val sh = o.getJSONArray("shape"); var c = 1; for (i in 0 until sh.length()) c *= sh.getInt(i)
        val b = map.duplicate(); b.position((o.getLong("offset") / 2).toInt())
        return FloatArray(c) { HALF[b.get().toInt() and 0xFFFF] }
    }
    private fun rows(n: String) = ts.getJSONObject(n).getJSONArray("shape").getInt(0)

    /** y = W x (+ bias) (+ g B (A x)); W is [out, in] row-major. */
    private fun linear(w: FloatArray, out: Int, x: FloatArray, bias: FloatArray?, ab: Pair<FloatArray, FloatArray>?): FloatArray {
        val n = x.size; val y = FloatArray(out)
        for (o in 0 until out) { var a = 0f; val r = o * n; for (i in 0 until n) a += w[r + i] * x[i]; y[o] = a + (bias?.get(o) ?: 0f) }
        if (ab != null) {
            val (la, lb) = ab; val rank = la.size / n; val h = FloatArray(rank)
            for (k in 0 until rank) { var a = 0f; val r = k * n; for (i in 0 until n) a += la[r + i] * x[i]; h[k] = a * strength }
            for (o in 0 until out) { var a = 0f; val r = o * rank; for (k in 0 until rank) a += lb[r + k] * h[k]; y[o] += a }
        }
        return y
    }
    private fun silu(x: FloatArray) = FloatArray(x.size) { val v = x[it]; (v / (1.0 + exp(-v.toDouble()))).toFloat() }

    /** The LoRA's pair for one modulation Linear, as [LoraImport.convertAnima] writes them: adaln_<name>.A.f16 / .B.f16. */
    private fun pair(name: String, inp: Int, out: Int): Pair<FloatArray, FloatArray>? {
        val a = File(lora ?: return null, "adaln_$name.A.f16"); val b = File(lora, "adaln_$name.B.f16")
        if (strength == 0f || !a.exists() || !b.exists() || a.length() % (2L * inp) != 0L) return null
        val rank = (a.length() / (2L * inp)).toInt(); if (b.length() != 2L * out * rank) return null
        fun read(f: File) = RandomAccessFile(f, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, f.length()) }.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().let { s -> FloatArray(s.remaining()) { HALF[s.get().toInt() and 0xFFFF] } }
        return read(a) to read(b)
    }

    /**
     * The inputs of every step: for each timestep, "b<i>_mod" [3 * 3D] (self-attention, cross-attention, MLP) per block and
     * "final_mod" [2D]. A few seconds for ten steps on a phone, on [threads] cores.
     */
    fun compute(timesteps: FloatArray, threads: Int = 4): List<Map<String, FloatArray>> {
        val half = d / 2; val n = timesteps.size
        val t1 = tensor("t.linear_1.weight"); val t2 = tensor("t.linear_2.weight")
        val b1 = if (has("t.linear_1.bias")) tensor("t.linear_1.bias") else null; val b2 = if (has("t.linear_2.bias")) tensor("t.linear_2.bias") else null
        val wn = tensor("t.norm.weight")
        val alora = ArrayList<FloatArray>(); val xs = ArrayList<FloatArray>()
        for (t in timesteps) {
            val s = FloatArray(d); for (k in 0 until half) { val a = t * exp(-ln(10000.0) * k / half); s[k] = cos(a).toFloat(); s[half + k] = sin(a).toFloat() }
            alora.add(linear(t2, rows("t.linear_2.weight"), silu(linear(t1, rows("t.linear_1.weight"), s, b1, null)), b2, null))
            var m = 0.0; for (v in s) m += v * v; val r = (1.0 / sqrt(m / d + 1e-6)).toFloat()
            xs.add(silu(FloatArray(d) { s[it] * r * wn[it] }))
        }
        val out = List(n) { java.util.concurrent.ConcurrentHashMap<String, FloatArray>() }
        // one modulation = its two weights converted once, then every step through them
        fun modulation(prefix: String, loraName: String?, width: Int): List<FloatArray> {
            val w1 = tensor("$prefix.1.weight"); val w2 = tensor("$prefix.2.weight"); val mid = rows("$prefix.1.weight")
            val p1 = loraName?.let { pair("$it.1", d, mid) }; val p2 = loraName?.let { pair("$it.2", mid, width) }
            return List(n) { s -> linear(w2, width, linear(w1, mid, xs[s], null, p1), null, p2).also { y -> val al = alora[s]; for (i in 0 until width) y[i] += al[i] } }
        }
        val pool = Executors.newFixedThreadPool(threads.coerceAtLeast(1))
        try {
            for (i in 0 until blocks) pool.execute {
                val three = (0..2).map { j -> modulation("b$i.$j", "b${i}_${KINDS[j]}", 3 * d) }
                for (s in 0 until n) out[s]["b${i}_mod"] = FloatArray(9 * d).also { v -> for (j in 0..2) System.arraycopy(three[j][s], 0, v, j * 3 * d, 3 * d) }
            }
            pool.execute { modulation("final", null, 2 * d).forEachIndexed { s, v -> out[s]["final_mod"] = v } }
            pool.shutdown(); check(pool.awaitTermination(10, TimeUnit.MINUTES)) { "the modulations took too long" }
        } finally { pool.shutdownNow() }
        check(out.all { it.size == blocks + 1 }) { "a modulation failed to compute" }
        return out
    }

    companion object {
        /** The three modulations of a block, as LoRA files name them (adaln_modulation_<kind>). */
        val KINDS = listOf("self_attn", "cross_attn", "mlp")
        /** Every half float as a float: converting 190M weights through a table takes a fraction of the bit arithmetic. */
        private val HALF = FloatArray(65536) { x ->
            val sign = (x and 0x8000) shl 16; val e = (x ushr 10) and 0x1F; val m = x and 0x3FF
            when {
                e == 0 -> if (m == 0) java.lang.Float.intBitsToFloat(sign) else (if (sign != 0) -1f else 1f) * m * Math.scalb(1f, -24)
                e == 31 -> java.lang.Float.intBitsToFloat(sign or 0x7F800000 or (m shl 13))
                else -> java.lang.Float.intBitsToFloat(sign or ((e - 15 + 127) shl 23) or (m shl 13))
            }
        }
        fun present(dir: File) = File(dir, "adaln.f16").exists() && File(dir, "adaln.json").exists()
    }
}
