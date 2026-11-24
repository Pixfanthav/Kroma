package dev.kroma.open

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow

/**
 * Noise schedules for a shifted flow model (Anima: ComfyUI's ModelSamplingDiscreteFlow, multiplier 1000), as ComfyUI computes
 * them -- the same sigmas for the same name, step count and shift (SchedTest checks them against ComfyUI's own output). The
 * first sigma is moved just under 1: at 1 ER-SDE's log-SNR is infinite.
 */
object Sched {
    val NAMES = listOf("simple", "karras", "linear", "exponential", "normal", "sgm_uniform", "beta")

    fun sigmas(steps: Int, shift: Double = 3.0, scheduler: String = "simple"): DoubleArray {
        fun shifted(t: Double) = shift * t / (1.0 + (shift - 1.0) * t)
        val n = 1000; val buf = DoubleArray(n) { shifted((it + 1).toDouble() / n) }          // the model's sigmas, t = 1..1000
        val lo = buf[0]; val hi = buf[n - 1]
        val s = DoubleArray(steps + 1)
        when (scheduler) {
            "karras" -> {
                val rho = 7.0; val a = lo.pow(1 / rho); val b = hi.pow(1 / rho)
                for (x in 0 until steps) s[x] = (b + (if (steps > 1) x.toDouble() / (steps - 1) else 0.0) * (a - b)).pow(rho)
            }
            "linear" -> for (x in 0 until steps) s[x] = shifted(1.0 - x.toDouble() / steps)
            "exponential" -> for (x in 0 until steps) s[x] = exp(ln(hi) + (if (steps > 1) x.toDouble() / (steps - 1) else 0.0) * (ln(lo) - ln(hi)))
            // timesteps evenly between the largest and smallest sigma's (sigma x 1000), each turned back into a sigma
            "normal", "sgm_uniform" -> {
                val start = hi * n; val end = lo * n; val div = if (scheduler == "sgm_uniform") steps else maxOf(steps - 1, 1)
                for (x in 0 until steps) s[x] = shifted((start + (end - start) * x / div) / n)
            }
            // timesteps at the quantiles of a Beta(0.6, 0.6), rounded; a repeated one is dropped (so fewer sigmas, as ComfyUI)
            "beta" -> {
                val out = ArrayList<Double>(); var last = -1L
                for (x in 0 until steps) {
                    val t = Math.rint(betaPpf(1.0 - x.toDouble() / steps, 0.6, 0.6) * (n - 1)).toLong()
                    if (t != last) out.add(buf[t.toInt()]); last = t
                }
                return DoubleArray(out.size + 1).also { for (i in out.indices) it[i] = out[i]; if (it[0] >= 1.0) it[0] = shifted(1.0 - 1e-4) }
            }
            else -> { val ss = n.toDouble() / steps; for (x in 0 until steps) s[x] = buf[n - 1 - (x * ss).toInt()] }
        }
        if (s[0] >= 1.0) s[0] = shifted(1.0 - 1e-4)
        return s
    }

    /** The inverse of the regularized incomplete beta function (scipy.stats.beta.ppf), by bisection. */
    fun betaPpf(q: Double, a: Double, b: Double): Double {
        if (q <= 0.0) return 0.0
        if (q >= 1.0) return 1.0
        var l = 0.0; var h = 1.0
        repeat(100) { val m = (l + h) / 2; if (betaInc(m, a, b) < q) l = m else h = m }
        return (l + h) / 2
    }
    /** I_x(a, b), Numerical Recipes' continued fraction. */
    fun betaInc(x: Double, a: Double, b: Double): Double {
        if (x <= 0.0) return 0.0
        if (x >= 1.0) return 1.0
        val bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * ln(x) + b * ln(1 - x))
        return if (x < (a + 1) / (a + b + 2)) bt * betacf(x, a, b) / a else 1 - bt * betacf(1 - x, b, a) / b
    }
    private fun betacf(x: Double, a: Double, b: Double): Double {
        val tiny = 1e-300; var c = 1.0; var d = 1 - (a + b) * x / (a + 1); if (abs(d) < tiny) d = tiny; d = 1 / d; var h = d
        for (m in 1..300) {
            val m2 = 2 * m
            var aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
            d = 1 + aa * d; if (abs(d) < tiny) d = tiny; c = 1 + aa / c; if (abs(c) < tiny) c = tiny; d = 1 / d; h *= d * c
            aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
            d = 1 + aa * d; if (abs(d) < tiny) d = tiny; c = 1 + aa / c; if (abs(c) < tiny) c = tiny; d = 1 / d
            val del = d * c; h *= del; if (abs(del - 1) < 1e-15) break
        }
        return h
    }
    /** ln Gamma (Lanczos) */
    private fun lgamma(z: Double): Double {
        val g = 7.0; val p = doubleArrayOf(0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
            12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
        if (z < 0.5) return ln(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z)
        val zz = z - 1; var x = p[0]; for (i in 1 until 9) x += p[i] / (zz + i)
        val t = zz + g + 0.5
        return 0.5 * ln(2 * Math.PI) + (zz + 0.5) * ln(t) - t + ln(x)
    }
}

/**
 * One step of a flow-matching sampler, as ComfyUI does it, from the model's denoised estimate: "er_sde" (sample_er_sde, up to third
 * order), "euler", "euler_ancestral" (sample_euler_ancestral_RF, eta 1) and "dpmpp_2m". [sig] the sigmas (the last one 0), [i0] the
 * first step that runs (img2img starts part way: no history before it), [noise] fresh unit noise for the stochastic samplers.
 * Shared by Anima and Krea 2, which turn their outputs into the denoised estimate first.
 */
class FlowSampler(val name: String, private val sig: DoubleArray, private val i0: Int, private val noise: () -> FloatArray) {
    private var oldDen: FloatArray? = null; private var oldDenD: FloatArray? = null
    private fun ns(x: Double) = x * (Math.exp(Math.pow(x, 0.3)) + 10.0)
    private fun halfLogSnr(s: Double) = -(Math.log(s) - Math.log1p(-s))
    // ER-SDE's ln(sigma / (1 - sigma)) is infinite at sigma 1, where Krea 2 Turbo's own schedule starts (Sched.sigmas keeps below it)
    private val erl = DoubleArray(sig.size) { Math.exp(-halfLogSnr(minOf(sig[it], 1.0 - 1e-6))) }
    private val numI = 200
    /** x at sig[i] -> x at sig[i + 1], given den (the denoised estimate at sig[i]) */
    fun step(x: FloatArray, den: FloatArray, i: Int) {
        val n = x.size
        when (name) {
            "euler_ancestral", "dpmpp_2m" -> {
                if (sig[i + 1] == 0.0) System.arraycopy(den, 0, x, 0, n)
                else if (name == "euler_ancestral") {
                    val sn = sig[i + 1]; val sd = sn * (sn / sig[i]); val aIp1 = 1 - sn; val aDown = 1 - sd
                    val renoise = Math.sqrt(maxOf(0.0, sn * sn - sd * sd * aIp1 * aIp1 / (aDown * aDown))); val ratio = sd / sig[i]
                    val nz = noise()
                    for (j in 0 until n) x[j] = ((aIp1 / aDown) * (ratio * x[j] + (1 - ratio) * den[j]) + nz[j] * renoise).toFloat()
                } else {
                    val h = -Math.log(sig[i + 1]) + Math.log(sig[i]); val ratio = (sig[i + 1] / sig[i]).toFloat(); val c = (-Math.expm1(-h)).toFloat()
                    val od = oldDen
                    if (od == null) for (j in 0 until n) x[j] = ratio * x[j] + c * den[j]
                    else { val r = (Math.log(sig[i - 1]) - Math.log(sig[i])) / h; val w1 = (1 + 1 / (2 * r)).toFloat(); val w0 = (1 / (2 * r)).toFloat()
                        for (j in 0 until n) x[j] = ratio * x[j] + c * (w1 * den[j] - w0 * od[j]) }
                }
            }
            "euler" -> { val r = (sig[i + 1] / sig[i]).toFloat(); for (j in 0 until n) x[j] = den[j] + (x[j] - den[j]) * r }
            else -> {   // er_sde
                if (sig[i + 1] == 0.0) System.arraycopy(den, 0, x, 0, n)
                else {
                    val els = erl[i]; val elt = erl[i + 1]
                    val alphaT = sig[i + 1] / elt; val r = ns(elt) / ns(els); val rAlpha = (sig[i + 1] / elt) / (sig[i] / els)
                    for (j in 0 until n) x[j] = (rAlpha * r * x[j] + alphaT * (1 - r) * den[j]).toFloat()
                    val stage = minOf(3, i - i0 + 1)
                    if (stage >= 2) {
                        val dt = elt - els; val lss = -dt / numI
                        var s = 0.0; var su = 0.0
                        for (k in 0 until numI) { val lp = elt + k * lss; val sp = ns(lp); s += 1.0 / sp; if (stage >= 3) su += (lp - els) / sp }
                        s *= lss; su *= lss
                        val dD = els - erl[i - 1]
                        val dd = FloatArray(n) { ((den[it] - oldDen!![it]) / dD).toFloat() }
                        val c2 = alphaT * (dt + s * ns(elt))
                        for (j in 0 until n) x[j] = (x[j] + c2 * dd[j]).toFloat()
                        if (stage >= 3) {
                            val dU = (els - erl[i - 2]) / 2.0
                            val du = FloatArray(n) { ((dd[it] - oldDenD!![it]) / dU).toFloat() }
                            val c3 = alphaT * ((dt * dt) / 2.0 + su * ns(elt))
                            for (j in 0 until n) x[j] = (x[j] + c3 * du[j]).toFloat()
                        }
                        oldDenD = dd
                    }
                    val ca = elt * elt - els * els * r * r
                    val coef = Math.sqrt(if (ca.isNaN() || ca < 0) 0.0 else ca)
                    val nz = noise()
                    for (j in 0 until n) x[j] = (x[j] + alphaT * nz[j] * coef).toFloat()
                }
            }
        }
        oldDen = den
    }
}
