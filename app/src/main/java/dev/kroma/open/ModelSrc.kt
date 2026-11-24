package dev.kroma.open

import org.json.JSONObject
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * What a model file's weights are stored as, found before it is fetched or converted, so that a page offering several
 * (bf16, fp8, Q8_0, MXFP4 ...) gives the converter the best one it reads rather than the first listed, and the format it is
 * converted to is asked knowing the source. kconv reads bf16, fp16, fp32, fp8 (e4m3fn / e5m2, ComfyUI's scaled ones too) and
 * GGUF F32 / F16 / BF16 / Q8_0 / Q4_0.
 */
object ModelSrc {
    /** one file a model is offered as: [fmt] bf16, f16, f32, fp8, q8_0, q4_0, or what it is when kconv does not read it */
    class Src(val path: String, val mb: Int, val fmt: String, val url: String = "")

    private val READ = listOf("bf16", "f16", "f32", "fp8", "q8_0", "q4_0")
    fun readable(fmt: String) = fmt in READ
    /** a file that is part of a model or an add-on to it, not a whole model: LoRAs, LyCORIS, control and adapter weights */
    private val ADDON = Regex("lora|lycoris|loha|lokr|dora|adapter|control|ip[-_]?adapter|(?<![a-z])(depth|canny|pose)(?![a-z])", RegexOption.IGNORE_CASE)
    /** the files worth offering as the model: whole models the converter reads, none much smaller than the largest (a LoRA or
     *  control file beside a 10 GB checkpoint), the best first (bf16 / fp16, fp8, Q8_0, Q4_0, fp32; the smaller of the same kind first) */
    fun offered(list: List<Src>): List<Src> {
        val ok = list.filter { readable(it.fmt) && !ADDON.containsMatchIn(it.path.substringAfterLast('/')) }
        val big = ok.maxOfOrNull { it.mb } ?: return emptyList()
        val rank = listOf("bf16", "f16", "fp8", "q8_0", "q4_0", "f32")
        return ok.filter { it.mb * 100L >= big * 15L }.sortedWith(compareBy({ rank.indexOf(it.fmt).let { i -> if (i <= 1) 0 else i } }, { it.mb }))
    }
    /** the file to take: the most precise one kconv reads (bf16 and fp16 alike; fp32 the same weights at twice the size) */
    fun best(list: List<Src>): Src? = list.filter { readable(it.fmt) }.minWithOrNull(compareBy({ READ.indexOf(it.fmt).let { i -> if (i <= 1) 0 else i } }, { it.mb }))

    fun label(fmt: String) = when (fmt) {
        "bf16" -> "BF16"; "f16" -> "FP16"; "f32" -> "FP32"; "fp8" -> "FP8"; "q8_0" -> "GGUF Q8_0"; "q4_0" -> "GGUF Q4_0"
        "shards" -> tr(R.string.split_in_parts)
        "" -> "?"; else -> fmt.uppercase()
    }
    /** why a file is not offered, for the list */
    fun why(fmt: String) = if (readable(fmt)) "" else tr(R.string.not_readable_yet)

    /** the convert formats worth offering for a source: nothing finer than the source itself (a Q8_0 file as fp16 is the same
     *  weights at twice the size) */
    fun outs(fmt: String) = when (fmt) { "q4_0" -> listOf("q4"); "q8_0" -> listOf("q8", "q4"); else -> listOf("f16", "q8", "q4") }

    /** the weights' type from a safetensors header: the dtype most of the bytes are in; packed 8- and 4-bit ones (U8 / I8: int8
     *  ConvRot, nvfp4, int4 ...) told apart by the file's name */
    fun fromHeader(hdr: String, name: String = ""): String {
        val o = JSONObject(hdr); val bytes = HashMap<String, Long>()
        for (k in o.keys()) {
            if (k == "__metadata__") continue
            val t = o.optJSONObject(k) ?: continue; val off = t.optJSONArray("data_offsets") ?: continue
            bytes.merge(t.optString("dtype"), off.getLong(1) - off.getLong(0), Long::plus)
        }
        val top = bytes.maxByOrNull { it.value }?.key ?: return ""
        val byName = nameFmt(name)
        return when (top) {
            "BF16" -> "bf16"; "F16" -> "f16"; "F32" -> "f32"
            // an mxfp8 file's E8M0 block scales are U8 beside the fp8 weights: not the per-tensor / per-row scales kconv applies
            "F8_E4M3", "F8_E5M2" -> if (byName == "mxfp8" || bytes.containsKey("U8") || bytes.containsKey("F8_E8M0")) "mxfp8" else "fp8"
            "U8", "I8" -> byName?.takeIf { !readable(it) } ?: "int8 / packed"
            else -> top.lowercase()
        }
    }

    /** what a file's name says (a GGUF's quant type; a safetensors' words like fp8, mxfp4, nf4), null when it says nothing */
    fun nameFmt(name: String): String? {
        val n = name.lowercase()
        if (n.endsWith(".gguf")) return (Regex("(?<![a-z0-9])(i?q[1-8]_[0-9a-z]+(?:_[a-z])?)(?![a-z0-9])").find(n) ?: Regex("(?<![a-z0-9])(bf16|f16|f32)(?![a-z0-9])").find(n))?.value
        return when {
            "mxfp4" in n -> "mxfp4"; "nvfp4" in n -> "nvfp4"; "mxfp8" in n -> "mxfp8"; Regex("(?<![a-z0-9])nf4").containsMatchIn(n) -> "nf4"
            Regex("int4|w4a4|svdquant").containsMatchIn(n) -> "int4"
            Regex("int8|(?<![a-z0-9])i8(?![a-z0-9])|w8a8|convrot").containsMatchIn(n) -> "int8 / packed"
            Regex("fp8|f8_e|e4m3|e5m2").containsMatchIn(n) -> "fp8"
            "bf16" in n -> "bf16"; Regex("fp16|(?<![a-z0-9])f16(?![a-z0-9])|half").containsMatchIn(n) -> "f16"; Regex("fp32|(?<![a-z0-9])f32(?![a-z0-9])").containsMatchIn(n) -> "f32"
            else -> null
        }
    }

    /** a file on a model's page that is not the model: the text encoder, the VAE (pages often carry them beside it) */
    fun isPart(path: String) = Regex("vae|qwen|text_?encoder|(?<![a-z])clip|(?<![a-z])u?t5|tokenizer|(?<![a-z])te[_-]|[_-]te(?![a-z])|llm|gemma|mistral", RegexOption.IGNORE_CASE)
        .containsMatchIn(path.substringAfterLast('/')) || Regex("(^|/)(vae|text_encoder\\w*|tokenizer\\w*)/").containsMatchIn(path)

    /** up to [n] bytes (fewer only at the end of the stream); readNBytes is API 33 and the app starts at 31 */
    private fun readUp(s: InputStream, n: Int): ByteArray { val b = ByteArray(n); var got = 0; while (got < n) { val r = s.read(b, got, n - got); if (r < 0) break; got += r }; return if (got == n) b else b.copyOf(got) }

    /** a stream's safetensors header (8-byte length, then the JSON) as its format; GGUF by name */
    fun fromStream(name: String, s: InputStream): String {
        if (name.endsWith(".gguf", true)) return nameFmt(name) ?: "gguf"
        val lb = readUp(s, 8); if (lb.size != 8) return ""
        val n = java.nio.ByteBuffer.wrap(lb).order(java.nio.ByteOrder.LITTLE_ENDIAN).long
        if (n !in 2..(100L shl 20)) return ""
        val hb = readUp(s, n.toInt()); if (hb.size != n.toInt()) return ""
        return fromHeader(String(hb, Charsets.UTF_8), name)
    }

    /** a file online, by its header alone (two ranged requests, a few hundred KB) */
    fun fromUrl(name: String, url: String): String {
        if (name.endsWith(".gguf", true)) return nameFmt(name) ?: "gguf"
        fun range(a: Long, b: Long): ByteArray {
            val c = URL(url).openConnection() as HttpURLConnection
            c.connectTimeout = 15000; c.readTimeout = 30000; c.setRequestProperty("Range", "bytes=$a-$b")
            try { if (c.responseCode !in 200..299) throw java.io.IOException("answered ${c.responseCode}"); return c.inputStream.use { readUp(it, (b - a + 1).toInt()) } }
            finally { c.disconnect() }
        }
        val n = java.nio.ByteBuffer.wrap(range(0, 7)).order(java.nio.ByteOrder.LITTLE_ENDIAN).long
        if (n !in 2..(100L shl 20)) return ""
        return fromHeader(String(range(8, 8 + n - 1), Charsets.UTF_8), name)
    }
}
