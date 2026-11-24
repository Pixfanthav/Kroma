package dev.kroma.open

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * The phone's LoRA conversion against the reference one (a Python script on a PC): the files must be the same byte
 * for byte. The pairs come from the environment -- LORA_TEST="<file.safetensors>|<reference dir>;..." -- and the
 * test is skipped without it, since neither the LoRAs nor the reference output belong in this repository.
 */
class LoraImportTest {
    @Test fun sameBytesAsTheReference() {
        val spec = System.getenv("LORA_TEST"); assumeTrue(!spec.isNullOrBlank())
        for (pair in spec!!.split(";").filter { it.isNotBlank() }) {
            val (src, ref) = pair.split("|").map { File(it.trim()) }
            val out = File(System.getProperty("java.io.tmpdir"), "lora_test_" + src.nameWithoutExtension.hashCode()).apply { deleteRecursively() }
            val r = LoraImport.convert(src, out)
            val want = ref.listFiles { f -> f.name.endsWith(".f16") }!!.map { it.name }.sorted()
            val got = out.listFiles { f -> f.name.endsWith(".f16") }!!.map { it.name }.sorted()
            assertEquals("${src.name}: which Linears", want, got)
            for (n in want) assertArrayEquals("${src.name}: $n", File(ref, n).readBytes(), File(out, n).readBytes())
            println("LORA ${src.name}: ${r.pairs} Linears identical to the reference, ${r.skipped} skipped")
            out.deleteRecursively()
        }
    }

    /**
     * The Anima conversion run on a PC, its output kept for a script to check against the weights it should stand for
     * (a reduced LoKr is not a byte-for-byte matter): ANIMA_LORA_TEST="<file.safetensors>|<output dir>".
     */
    @Test fun animaConversionForChecking() {
        val spec = System.getenv("ANIMA_LORA_TEST"); assumeTrue(!spec.isNullOrBlank())
        val (src, out) = spec!!.split("|").map { File(it.trim()) }
        out.deleteRecursively()
        val t = System.currentTimeMillis(); val r = LoraImport.convertAnima(src, out)
        println("ANIMA ${src.name}: ${r.pairs} Linears, ${r.skipped} skipped, ${(System.currentTimeMillis() - t) / 1000.0}s")
    }

    /**
     * Which model a file is for, and that it converts for that model: DETECT_LORA_TEST="<file>=<anima|krea>;...".
     * (An Anima file once went in as a Krea LoRA of 56 "layers": names alone are not enough.)
     */
    @Test fun detectsTheModel() {
        val spec = System.getenv("DETECT_LORA_TEST"); assumeTrue(!spec.isNullOrBlank())
        for (pair in spec!!.split(";").filter { it.isNotBlank() }) {
            val src = File(pair.substringBeforeLast("=").trim()); val want = pair.substringAfterLast("=").trim()
            assertEquals(src.name, want, LoraImport.detect(src))
            val out = File(System.getProperty("java.io.tmpdir"), "lora_detect_" + src.nameWithoutExtension.hashCode()).apply { deleteRecursively() }
            val r = if (want == "anima") LoraImport.convertAnima(src, out) else LoraImport.convert(src, out)
            println("DETECT ${src.name}: $want, ${r.pairs} Linears, ${r.skipped} skipped")
            // the other model's converter must refuse it
            val other = File(out.path + "_x")
            val wrong = runCatching { if (want == "anima") LoraImport.convert(src, other) else LoraImport.convertAnima(src, other) }
            assertEquals("${src.name} must not convert for the other model", true, wrong.isFailure)
            out.deleteRecursively(); other.deleteRecursively()
        }
    }

    /**
     * Every .safetensors in a folder through detection and conversion, as a table: LORA_SURVEY="<folder>[;<folder>]".
     * Nothing is asserted beyond "no file crashes the reader"; it is for seeing what real files do.
     */
    @Test fun surveysAFolder() {
        val spec = System.getenv("LORA_SURVEY"); assumeTrue(!spec.isNullOrBlank())
        val files = spec!!.split(";").filter { it.isNotBlank() }.flatMap { File(it.trim()).listFiles { f -> f.name.endsWith(".safetensors") && f.length() in (1L shl 20)..(1200L shl 20) }?.toList() ?: emptyList() }.sortedBy { it.name.lowercase() }
        val out = File(System.getProperty("java.io.tmpdir"), "lora_survey_out")
        for (src in files) {
            val model = runCatching { LoraImport.detect(src) }.getOrElse { "ERROR ${it.javaClass.simpleName}: ${it.message?.take(60)}" }
            val line = when (model) {
                "anima", "krea" -> runCatching { out.deleteRecursively(); if (model == "anima") LoraImport.convertAnima(src, out) else LoraImport.convert(src, out) }
                    .fold({ "${it.pairs} layers${if (it.adaln > 0) " + ${it.adaln} modulation" else ""}, ${it.skipped} left out" }, { "FAILED ${it.message?.take(80)}" })
                else -> "-"
            }
            println("SURVEY\t${src.length() shr 20}MB\t${model ?: "unknown"}\t$line\t${src.name}")
        }
        out.deleteRecursively()
    }

    @Test fun halfRounding() {
        assertEquals(0x3C00.toShort(), LoraImport.half(1f)); assertEquals(0xC000.toShort(), LoraImport.half(-2f))
        assertEquals(0x7C00.toShort(), LoraImport.half(70000f)); assertEquals(0x0001.toShort(), LoraImport.half(5.9604645e-8f))
        assertEquals(0x3C00.toShort(), LoraImport.half(1.00048828125f))      // halfway between 1 and the next half: to even
    }
}
