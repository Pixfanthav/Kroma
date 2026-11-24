package dev.kroma.open

import org.junit.Assert.assertEquals
import org.junit.Test

/** Sched against ComfyUI's calculate_sigmas (ModelSamplingDiscreteFlow, shift 3, 10 steps), a 10-step reference file. */
class SchedTest {
    private val ref = mapOf(
        "simple" to doubleArrayOf(1.0, 0.9642856717, 0.9230769873, 0.8749999404, 0.8181818128, 0.75, 0.6666666865, 0.5625, 0.4285714626, 0.25, 0.0),
        "karras" to doubleArrayOf(1.0, 0.6356975436, 0.3916346729, 0.2327368259, 0.1326435506, 0.071972765, 0.0368316881, 0.0175571479, 0.0076657473, 0.0029940128, 0.0),
        "exponential" to doubleArrayOf(1.0, 0.524304688, 0.2748953998, 0.1441289335, 0.0755674765, 0.0396203771, 0.02077315, 0.010891459, 0.005710443, 0.0029940119, 0.0),
        "sgm_uniform" to doubleArrayOf(1.0, 0.9644002914, 0.9233424664, 0.8754674196, 0.8189232945, 0.7511211038, 0.6683266759, 0.5649496317, 0.4322250783, 0.2555886507, 0.0),
        "normal" to doubleArrayOf(1.0, 0.960129261, 0.9133489132, 0.8576923609, 0.7903683782, 0.7072784901, 0.6021505594, 0.4648760259, 0.2780487835, 0.0089285718, 0.0),
        "beta" to doubleArrayOf(1.0, 0.9863013029, 0.9553377032, 0.9075828791, 0.8409490585, 0.75074929, 0.63093853, 0.4795918167, 0.2980768979, 0.1136783734, 0.0),
    )
    @Test fun sameAsComfyUI() {
        for ((name, want) in ref) {
            val got = Sched.sigmas(10, 3.0, name)
            assertEquals("$name: length", want.size, got.size)
            // the first sigma is moved under 1 on purpose (ER-SDE); the rest must match
            for (i in 1 until want.size) assertEquals("$name[$i]", want[i], got[i], 1e-6)
        }
    }
    @Test fun betaPpfKnownValues() {
        assertEquals(0.5, Sched.betaPpf(0.5, 0.6, 0.6), 1e-9)
        assertEquals(1.0 - Sched.betaPpf(0.2, 0.6, 0.6), Sched.betaPpf(0.8, 0.6, 0.6), 1e-9)
    }
}
