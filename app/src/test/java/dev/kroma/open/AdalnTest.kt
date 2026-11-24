package dev.kroma.open

import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * The modulations the phone computes against the model's own (PyTorch on a PC): ADALN_TEST="<dir>[|<lora.safetensors>]",
 * the dir holding adaln.f16, adaln.json and adaln_ref.f32 (and adaln_ref_lora.f32 for the LoRA) as the export script writes
 * them. Skipped without it: the weights are 390MB.
 */
class AdalnTest {
    private fun floats(f: File) = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().let { b -> FloatArray(b.remaining()).also { b.get(it) } }
    private fun flat(a: Adaln, r: List<Map<String, FloatArray>>) = r.flatMap { s -> (0 until a.blocks).flatMap { s["b${it}_mod"]!!.asList() } + s["final_mod"]!!.asList() }
    private fun rel(x: List<Float>, y: FloatArray): Double {
        assertTrue("sizes ${x.size} and ${y.size}", x.size == y.size)
        var d = 0.0; var n = 0.0; for (i in y.indices) { d += (x[i] - y[i]).toDouble().let { it * it }; n += y[i].toDouble() * y[i] }
        return sqrt(d / n)
    }

    @Test fun matchesTheModel() {
        val spec = System.getenv("ADALN_TEST"); assumeTrue(!spec.isNullOrBlank())
        val dir = File(spec!!.substringBefore('|')); val ts = floatArrayOf(0.6f, 0.95f)
        val a = Adaln(dir); val t0 = System.currentTimeMillis()
        val e = rel(flat(a, a.compute(ts)), floats(File(dir, "adaln_ref.f32")))
        println("ADALN ${a.blocks} blocks, 2 steps in ${System.currentTimeMillis() - t0}ms: off the model by rel $e")
        assertTrue("off by $e", e < 2e-3)              // the weights are half floats here, float32 in the reference
        if ('|' in spec) {
            val out = File(System.getProperty("java.io.tmpdir"), "adaln_lora_test").apply { deleteRecursively() }
            val r = LoraImport.convertAnima(File(spec.substringAfter('|')), out)
            val al = Adaln(dir, out, 1f)
            val got = flat(al, al.compute(ts)); val e1 = rel(got, floats(File(dir, "adaln_ref_lora.f32"))); val moved = rel(got, floats(File(dir, "adaln_ref.f32")))
            println("ADALN with the LoRA (${r.adaln} modulation Linears): off the merged model by rel $e1; the LoRA moves the modulations by rel $moved")
            assertTrue("no modulation Linears converted", r.adaln > 0); assertTrue("off by $e1", e1 < 2e-3); assertTrue(moved > e1 * 3)
            out.deleteRecursively()
        }
    }
}
