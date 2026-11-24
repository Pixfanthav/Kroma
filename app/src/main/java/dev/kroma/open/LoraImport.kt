package dev.kroma.open

import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** The LoRAs chosen for a picture, for the engine's exact bake: a file of "<folder>\t<strength>" lines, in the order they go in. */
object LoraSet {
    fun write(f: File, libs: List<Pair<File, Float>>): File {
        val text = libs.joinToString("") { "${it.first.absolutePath}\t${it.second}\n" }
        if (runCatching { f.readText() }.getOrNull() != text) { f.parentFile?.mkdirs(); f.writeText(text) }
        return f
    }
    fun dirs(f: File): List<File> = runCatching { f.readLines().mapNotNull { l -> l.substringBefore('\t').takeIf { it.isNotEmpty() }?.let { File(it) } } }.getOrDefault(emptyList())
    /** what the set is: its lines and when each folder last changed */
    fun tag(f: File): String = runCatching { f.readText() }.getOrDefault("") + dirs(f).joinToString("|") { "${File(it, "info.json").lastModified()}" }
}

/**
 * A Krea 2 LoRA as people share it (one .safetensors) -> the per-Linear fp16 files the generator reads:
 *   <key>.A.f16 [rank,in] and <key>.B.f16 [out,rank], key like blocks.3.attn.wq or txtfusion.layerwise_blocks.0.mlp.up,
 * with alpha/rank folded into B. Done on the phone, so nothing has to be prepared on a PC.
 *
 * Accepts ComfyUI-native keys (diffusion_model.blocks.N.attn.wq.lora_A.weight), diffusers/peft ones
 * (transformer.transformer_blocks.N.attn.to_q.lora_A.weight, with or without ".default") and kohya ones
 * (lora_unet_blocks_3_attn_wq.lora_down.weight). Everything goes in exactly, for the engine to bake (kengine's bake_set):
 * a pair at its own rank (a full-rank one too), a LoKr as its two Kronecker factors (<key>.lokr.f32), a DoRA magnitude
 * (<key>.dora.f32), and the text path's RMSNorm scales and projector as small pairs. LoHa is refused.
 *
 * A safetensors file is an 8-byte little-endian header length, a JSON header (name -> dtype, shape, data_offsets)
 * and the raw tensors; each tensor is read on its own, so a large file never has to fit in memory.
 */
object LoraImport {
    /** [adaln]: Anima's modulation Linears, kept for bundles that compute the modulations on the CPU ([Adaln]); others leave them out. */
    class Result(val pairs: Int, val skipped: Int, val adaln: Int = 0)

    private val ATTN = listOf("to_q" to "wq", "to_k" to "wk", "to_v" to "wv", "to_gate" to "gate", "to_out.0" to "wo")
    private val FF = listOf("ff.up" to "mlp.up", "ff.gate" to "mlp.gate", "ff.down" to "mlp.down")
    // text-path Linears are (in, out). Some merged or extracted LoRAs store these transposed; if any non-square one
    // is, none of the file's text extras is trusted (a square one cannot be told apart).
    private val TEXT_IO = mapOf("txtmlp.1" to (2560 to 6144), "txtmlp.3" to (6144 to 6144), "txtfusion.projector" to (12 to 1))
    private val KOHYA_TEXT = Regex("^lora_unet_(txtmlp)_(\\d)$")
    private val KOHYA_PROJ = Regex("^lora_unet_txtfusion_(projector)$")
    private val KOHYA = Regex("^lora_unet_(?:(txtfusion)_(layerwise_blocks|refiner_blocks)_|)(?:blocks_)?(\\d+)_(attn|mlp)_(\\w+)$")
    private val PREFIX = Regex("^(diffusion_model|transformer|model\\.diffusion_model)\\.")
    /** the Linears outside the blocks and the text path, by kohya, diffusers and the checkpoint's own names */
    private val OUTER = mapOf("first" to "first", "img_in" to "first", "last_linear" to "last.linear", "last.linear" to "last.linear",
        "final_layer.linear" to "last.linear", "tmlp_0" to "tmlp.0", "tmlp.0" to "tmlp.0", "time_embed.linear_1" to "tmlp.0",
        "tmlp_2" to "tmlp.2", "tmlp.2" to "tmlp.2", "time_embed.linear_2" to "tmlp.2", "tproj_1" to "tproj.1", "tproj.1" to "tproj.1", "time_mod_proj" to "tproj.1")
    /** their (in, out): the first and last layers (first.onnx / last.onnx) and the timestep path (tpath.bf16) */
    val OUTER_IO = mapOf("first" to (64 to 6144), "last.linear" to (6144 to 64), "tmlp.0" to (256 to 6144), "tmlp.2" to (6144 to 6144), "tproj.1" to (6144 to 36864))
    private val DOWN = Regex("(.*)\\.(lora_A|lora_down)(\\.default)?\\.weight$")

    /** diffusers / peft / kohya names -> the checkpoint's own (blocks.N.attn.wq, txtfusion.*.mlp.up, ...). */
    fun native(key: String): String {
        KOHYA_TEXT.find(key)?.let { return "txtmlp." + it.groupValues[2] }
        OUTER[key.removePrefix("lora_unet_")]?.let { return it }
        PREFIX.replaceFirst(key, "").let { k -> OUTER[k]?.let { return it } }
        if (KOHYA_PROJ.containsMatchIn(key)) return "txtfusion.projector"
        KOHYA.find(key)?.let { m ->
            val (tf, grp, n, sub, ln) = m.destructured
            return (if (tf.isNotEmpty()) "txtfusion.$grp.$n" else "blocks.$n") + ".$sub.$ln"
        }
        var k = PREFIX.replaceFirst(key, "")
        k = k.replace("transformer_blocks.", "blocks.").replace("text_fusion.", "txtfusion.")
        k = k.replace("txt_in.linear_1", "txtmlp.1").replace("txt_in.linear_2", "txtmlp.3")
        for ((a, b) in ATTN) if (k.endsWith(".attn.$a")) return k.dropLast(a.length) + b
        for ((a, b) in FF) if (k.endsWith(".$a")) return k.dropLast(a.length) + b
        return k
    }

    /** IEEE float -> half, round to nearest even (what numpy's astype(float16) does). */
    fun half(f: Float): Short {
        val x = java.lang.Float.floatToRawIntBits(f); val sign = (x ushr 16) and 0x8000
        val exp = (x ushr 23) and 0xFF; var man = x and 0x7FFFFF
        if (exp == 255) return (sign or 0x7C00 or (if (man != 0) 0x200 else 0)).toShort()
        val e = exp - 127 + 15
        if (e >= 31) return (sign or 0x7C00).toShort()
        if (e <= 0) {
            if (e < -10) return sign.toShort()
            man = man or 0x800000; val shift = 14 - e
            var h = man ushr shift; val rem = man and ((1 shl shift) - 1); val halfway = 1 shl (shift - 1)
            if (rem > halfway || (rem == halfway && (h and 1) == 1)) h++
            return (sign or h).toShort()
        }
        var h = (e shl 10) or (man ushr 13); val rem = man and 0x1FFF
        if (rem > 0x1000 || (rem == 0x1000 && (h and 1) == 1)) h++          // a carry runs into the exponent, which is right
        return (sign or h).toShort()
    }

    private fun unhalf(h: Short): Float {
        val x = h.toInt() and 0xFFFF; val sign = (x and 0x8000) shl 16; val e = (x ushr 10) and 0x1F; val m = x and 0x3FF
        return when {
            e == 0 -> if (m == 0) java.lang.Float.intBitsToFloat(sign) else (if (sign != 0) -1f else 1f) * m * Math.scalb(1f, -24)
            e == 31 -> java.lang.Float.intBitsToFloat(sign or 0x7F800000 or (m shl 13))
            else -> java.lang.Float.intBitsToFloat(sign or ((e - 15 + 127) shl 23) or (m shl 13))
        }
    }

    /** fp8 bytes to floats, exactly: e4m3fn (bias 7, no inf, 0x7F / 0xFF NaN) or e5m2 (bias 15, IEEE-like). Some LoRAs are
     *  uploaded in fp8 to halve the file; ComfyUI widens them on load, and so does this */
    private fun fp8Lut(e4m3: Boolean) = FloatArray(256) { b ->
        val sg = b shr 7; val e = if (e4m3) (b shr 3) and 15 else (b shr 2) and 31; val m = if (e4m3) b and 7 else b and 3
        val mb = if (e4m3) 3 else 2; val bias = if (e4m3) 7 else 15
        var v = if (e != 0) Math.scalb(1f + m.toFloat() / (1 shl mb), e - bias) else Math.scalb(m.toFloat() / (1 shl mb), 1 - bias)
        if (if (e4m3) (b and 127) == 127 else e == 31) v = if (e4m3 || m != 0) Float.NaN else Float.POSITIVE_INFINITY
        if (sg == 1) -v else v
    }
    private class T(val dtype: String, val shape: IntArray, val a: Long, val b: Long)

    /** The tensors of a safetensors file by name, read one at a time. */
    private class Reader(val raf: RandomAccessFile) {
        val ts = HashMap<String, T>(); private val base0: Long
        var meta: JSONObject? = null                                   // the file's own __metadata__, for [infoOf]
        init {
            val lenB = ByteArray(8); raf.readFully(lenB)
            val n = ByteBuffer.wrap(lenB).order(ByteOrder.LITTLE_ENDIAN).long
            require(n in 2..(100L shl 20)) { "not a safetensors file" }
            val hb = ByteArray(n.toInt()); raf.readFully(hb)
            val hdr = JSONObject(String(hb, Charsets.UTF_8)); base0 = 8 + n
            meta = hdr.optJSONObject("__metadata__")
            for (k in hdr.keys()) {
                if (k == "__metadata__") continue
                val o = hdr.getJSONObject(k); val sh = o.getJSONArray("shape"); val off = o.getJSONArray("data_offsets")
                ts[k] = T(o.getString("dtype"), IntArray(sh.length()) { sh.getInt(it) }, off.getLong(0), off.getLong(1))
            }
        }
        fun read(t: T): FloatArray {
            val bytes = ByteArray((t.b - t.a).toInt()); raf.seek(base0 + t.a); raf.readFully(bytes)
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            return when (t.dtype) {
                "F32" -> FloatArray(bytes.size / 4) { bb.getFloat(it * 4) }
                "F16" -> FloatArray(bytes.size / 2) { unhalf(bb.getShort(it * 2)) }
                "BF16" -> FloatArray(bytes.size / 2) { java.lang.Float.intBitsToFloat((bb.getShort(it * 2).toInt() and 0xFFFF) shl 16) }
                "F64" -> FloatArray(bytes.size / 8) { bb.getDouble(it * 8).toFloat() }
                "F8_E4M3", "F8_E5M2" -> { val lut = fp8Lut(t.dtype == "F8_E4M3"); FloatArray(bytes.size) { lut[bytes[it].toInt() and 0xFF] } }
                else -> error("tensor type ${t.dtype} is not supported")
            }
        }
        /** a tensor straight into an fp16 file, a million values at a time: a full-rank pair (2560 x 6912 floats) is too big to
         *  hold as a FloatArray beside its bytes */
        fun toHalf(t: T, f: File, scale: Float = 1f) {
            val es = when (t.dtype) { "F32" -> 4; "F16", "BF16" -> 2; "F64" -> 8; "F8_E4M3", "F8_E5M2" -> 1; else -> error("tensor type ${t.dtype} is not supported") }
            val lut = if (es == 1) fp8Lut(t.dtype == "F8_E4M3") else null
            val n = (t.b - t.a) / es; val chunk = 1 shl 20
            raf.seek(base0 + t.a)
            f.outputStream().buffered(1 shl 20).use { os ->
                val inb = ByteArray(chunk * es); val ob = ByteBuffer.allocate(chunk * 2).order(ByteOrder.LITTLE_ENDIAN); var done = 0L
                while (done < n) {
                    val c = minOf(chunk.toLong(), n - done).toInt(); raf.readFully(inb, 0, c * es)
                    val bb = ByteBuffer.wrap(inb, 0, c * es).order(ByteOrder.LITTLE_ENDIAN); ob.clear()
                    for (i in 0 until c) {
                        val v = when (es) { 4 -> bb.getFloat(i * 4); 8 -> bb.getDouble(i * 8).toFloat(); 1 -> lut!![inb[i].toInt() and 0xFF]
                            else -> if (t.dtype == "F16") unhalf(bb.getShort(i * 2)) else java.lang.Float.intBitsToFloat((bb.getShort(i * 2).toInt() and 0xFFFF) shl 16) }
                        ob.putShort(half(v * scale))
                    }
                    os.write(ob.array(), 0, c * 2); done += c
                }
            }
        }
        /** what diffusers (peft) writes instead of .alpha tensors: lora_alpha, r, use_rslora, alpha_pattern, under "<component>." names */
        private val peft: Map<String, Any?>? by lazy {
            val o = meta?.optString("lora_adapter_metadata")?.takeIf { it.isNotEmpty() }?.let { runCatching { JSONObject(it) }.getOrNull() } ?: return@lazy null
            val pre = o.keys().asSequence().firstOrNull { it.endsWith("lora_alpha") }?.removeSuffix("lora_alpha") ?: return@lazy null
            mapOf("alpha" to o.optDouble(pre + "lora_alpha", Double.NaN), "rs" to o.optBoolean(pre + "use_rslora", false), "pat" to o.optJSONObject(pre + "alpha_pattern"))
        }
        /** alpha / rank for a pair: its own .alpha tensor (kohya, ComfyUI), else the peft metadata's (alpha / sqrt(rank) with
         *  use_rslora: a pix LoRA at r 64, alpha 128 once went in at 1/16 of its strength), else 1 */
        fun scale(base: String, alpha: String?, r: Int): Float {
            alpha?.let { return read(ts[it]!!)[0] / r }
            val p = peft ?: return 1f
            var a = p["alpha"] as Double
            (p["pat"] as JSONObject?)?.let { pat -> pat.keys().asSequence().firstOrNull { Regex("(.*\\.)?($it)$").matches(base) }?.let { a = pat.getDouble(it) } }
            if (a.isNaN()) return 1f
            return (if (p["rs"] == true) a / Math.sqrt(r.toDouble()) else a / r).toFloat()
        }
        /** base name -> (down, up, alpha) of every plain LoRA pair in the file. */
        fun pairs(): LinkedHashMap<String, Triple<String, String, String?>> {
            val pairs = LinkedHashMap<String, Triple<String, String, String?>>()
            for (k in ts.keys) {
                val m = DOWN.matchEntire(k) ?: continue
                val base = m.groupValues[1]; val dflt = m.groupValues[3]
                val up = listOf("$base.lora_B$dflt.weight", "$base.lora_up.weight").firstOrNull { it in ts } ?: continue
                pairs[base] = Triple(k, up, "$base.alpha".takeIf { it in ts })
            }
            return pairs
        }
    }
    private fun write(f: File, a: FloatArray, scale: Float = 1f) {
        val bb = ByteBuffer.allocate(a.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        if (scale == 1f) for (v in a) bb.putShort(half(v)) else for (v in a) bb.putShort(half(v * scale))
        f.writeBytes(bb.array())
    }

    private fun writeF32(f: File, head: IntArray, vararg parts: FloatArray) {
        val bb = ByteBuffer.allocate(head.size * 4 + parts.sumOf { it.size } * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (v in head) bb.putInt(v); for (p in parts) for (v in p) bb.putFloat(v)
        f.writeBytes(bb.array())
    }
    /** a DoRA magnitude beside the change of [name]: on the output axis when it has a value per output row (ComfyUI's test) */
    private fun putDora(rd: Reader, base: String, name: String, out: File, outD: Int): Boolean {
        val t = rd.ts["$base.dora_scale"] ?: return false
        val v = rd.read(t); writeF32(File(out, "$name.dora.f32"), intArrayOf(if (t.shape[0] == outD) 0 else 1, v.size), v); return true
    }
    /** a LoKr ([base].lokr_w1 or _w1_a _w1_b, times lokr_w2 or _w2_a _w2_b) as its two factors, alpha / dim in w1, for the
     *  change kron(w1, w2) of an [o] x [i] Linear; false when it is not one or does not fit */
    private fun putLokr(rd: Reader, base: String, name: String, out: File, o: Int, i: Int): Boolean {
        val ts = rd.ts; fun t(n: String) = ts["$base.$n"]
        var dim: Int? = null
        fun mat(full: String): Triple<FloatArray, Int, Int>? {
            t(full)?.let { if (it.shape.size == 2) return Triple(rd.read(it), it.shape[0], it.shape[1]) }
            val a = t(full + "_a") ?: return null; val b = t(full + "_b") ?: return null
            if (a.shape.size != 2 || b.shape.size != 2 || a.shape[1] != b.shape[0]) return null
            dim = b.shape[0]
            val av = rd.read(a); val bv = rd.read(b); val m = a.shape[0]; val r = a.shape[1]; val n = b.shape[1]; val c = FloatArray(m * n)
            for (y in 0 until m) for (k in 0 until r) { val f = av[y * r + k]; if (f != 0f) for (x in 0 until n) c[y * n + x] += f * bv[k * n + x] }
            return Triple(c, m, n)
        }
        val w1 = mat("lokr_w1") ?: return false; val w2 = mat("lokr_w2") ?: return false
        if (w1.second * w2.second != o || w1.third * w2.third != i) return false
        val s = t("alpha")?.let { al -> dim?.let { rd.read(al)[0] / it } } ?: 1f
        writeF32(File(out, "$name.lokr.f32"), intArrayOf(w1.second, w1.third, w2.second, w2.third), FloatArray(w1.first.size) { w1.first[it] * s }, w2.first)
        return true
    }

    fun convert(src: File, out: File): Result = RandomAccessFile(src, "r").use { raf ->
        val rd = Reader(raf); val ts = rd.ts; fun read(t: T) = rd.read(t)
        val pairs = rd.pairs()
        val badText = pairs.any { (b, p) -> TEXT_IO[native(b)]?.let { it != (ts[p.first]!!.shape[1] to ts[p.second]!!.shape[0]) } ?: false }
        out.mkdirs()
        var done = 0; var skipped = 0
        for ((base, p) in pairs) {
            val key = native(base); val textExtra = key in TEXT_IO
            if (textExtra && badText) { skipped++; continue }
            // the blocks' attention and MLP Linears, and the text path's Linears and RMSNorm scales
            val parts = key.split(".")
            val text = textIO(key) ?: OUTER_IO[key]
            if (text == null && (!key.startsWith("blocks.") || parts.getOrNull(parts.size - 2) !in setOf("attn", "mlp"))) { skipped++; continue }
            val ta = ts[p.first]!!; val tb = ts[p.second]!!
            if (ta.shape.size != 2 || tb.shape.size != 2 || ta.shape[0] != tb.shape[1]) { skipped++; continue }
            if (text != null && (ta.shape[1] != text.first || tb.shape[0] != text.second)) { skipped++; continue }
            // a block Linear must be one of Krea's own, with its sizes: a LoRA for another model has "blocks.N.mlp.*" names too
            // (an Anima file once went in as 56 "layers")
            if (key.startsWith("blocks.")) {
                val io = KREA_IO[parts.takeLast(2).joinToString(".")]
                if (io == null || ta.shape[1] != io.first || tb.shape[0] != io.second) { skipped++; continue }
            }
            val rank = ta.shape[0]
            val scale = rd.scale(base, p.third, rank)
            rd.toHalf(ta, File(out, "$key.A.f16")); rd.toHalf(tb, File(out, "$key.B.f16"), scale)
            putDora(rd, base, key, out, tb.shape[0])
            done++
        }
        for (b0 in ts.keys.filter { it.endsWith(".lokr_w1") || it.endsWith(".lokr_w1_a") }.map { it.substringBeforeLast('.') }) {
            val key = native(b0); val io = textIO(key) ?: OUTER_IO[key] ?: KREA_IO[key.split(".").takeLast(2).joinToString(".")].takeIf { key.startsWith("blocks.") }
            if (io == null || !putLokr(rd, b0, key, out, io.second, io.first)) { skipped++; continue }
            putDora(rd, b0, key, out, io.second); done++
        }
        if (done == 0) { out.deleteRecursively(); error(if (ts.keys.any { "hada_" in it }) UNSUPPORTED else "no LoRA weights for Krea 2 in this file") }
        File(out, "info.json").writeText(infoOf(src).put("source", src.name).put("pairs", done).put("skipped", skipped).put("rank", rankOf(out, false)).toString())
        Result(done, skipped)
    }

    /** the text path's (in, out) by name: txtfusion's blocks, the projector and txtmlp; an RMSNorm scale as a (1, n) Linear */
    fun textIO(key: String): Pair<Int, Int>? = when {
        key.endsWith("qnorm.scale") || key.endsWith("knorm.scale") -> 1 to 128
        key.endsWith("prenorm.scale") || key.endsWith("postnorm.scale") || key == "txtmlp.0.scale" -> 1 to 2560
        !key.startsWith("txtfusion.") -> TEXT_IO[key]
        key.endsWith("mlp.down") -> 6912 to 2560
        key.endsWith("mlp.gate") || key.endsWith("mlp.up") -> 2560 to 6912
        key.endsWith(".attn.wq") || key.endsWith(".attn.wk") || key.endsWith(".attn.wv") || key.endsWith(".attn.gate") || key.endsWith(".attn.wo") -> 2560 to 2560
        else -> TEXT_IO[key]
    }
    /** a folder (or a set of them) changes the text path in a way txt2.onnx's LoRA inputs cannot carry, or only at a large cost:
     *  a LoKr, a DoRA magnitude, an RMSNorm scale or a rank above 256 -- so the text path is baked instead */
    fun textNeedsBake(dirs: List<File>): Boolean = dirs.any { d ->
        (d.listFiles { f -> f.name.startsWith("txtfusion.") || f.name.startsWith("txtmlp.") } ?: emptyArray()).any { f ->
            f.name.endsWith(".lokr.f32") || f.name.endsWith(".dora.f32") || f.name.endsWith(".scale.A.f16") ||
                (f.name.endsWith(".A.f16") && textIO(f.name.removeSuffix(".A.f16"))?.let { f.length() / 2 / it.first > 256 } == true) } }
    /** a change to the timestep path (computed on the CPU from tpath.bf16 instead of the step tables) */
    fun hasTime(dirs: List<File>): Boolean = dirs.any { d -> d.list { _, n -> n.startsWith("tmlp.") || n.startsWith("tproj.") }?.isNotEmpty() == true }
    /** a change to first.onnx / last.onnx */
    fun hasOuter(dirs: List<File>, prefix: String): Boolean = dirs.any { d -> d.list { _, n -> n.startsWith(prefix) }?.isNotEmpty() == true }
    /** any change to Krea's 28 blocks (else there is nothing to bake into them) */
    fun hasBlocks(dirs: List<File>): Boolean = dirs.any { d -> d.list { _, n -> n.startsWith("blocks.") || n.startsWith("layers.") || n.startsWith("kb.") }?.isNotEmpty() == true }   // layers.: MOSS-Image

    /** The Linears of a Krea 2 block that take an adapter, with (in, out). */
    val KREA_IO = mapOf("attn.wq" to (6144 to 6144), "attn.wk" to (6144 to 1536), "attn.wv" to (6144 to 1536), "attn.gate" to (6144 to 6144), "attn.wo" to (6144 to 6144),
        "mlp.gate" to (6144 to 16384), "mlp.up" to (6144 to 16384), "mlp.down" to (16384 to 6144))

    /**
     * What a trainer wrote into the file, for the app's info sheet: a title, the rank and alpha, the trigger words (the most
     * common tags of ss_tag_frequency), who trained it and when. Everything is optional: plenty of files carry nothing.
     */
    fun infoOf(src: File): JSONObject = RandomAccessFile(src, "r").use { raf ->
        val o = JSONObject(); val m = Reader(raf).meta ?: return@use o
        fun str(vararg k: String) = k.firstNotNullOfOrNull { kk -> m.optString(kk).takeIf { v -> v.isNotEmpty() } }
        str("ss_output_name", "modelspec.title", "name")?.let { o.put("title", it) }
        str("ss_network_dim")?.toIntOrNull()?.let { o.put("rank", it) }
        str("ss_network_alpha")?.toFloatOrNull()?.let { o.put("alpha", it) }
        str("ss_base_model_version", "modelspec.architecture")?.let { o.put("base", it) }
        str("modelspec.date", "ss_training_finished_at")?.let { o.put("date", it.take(10)) }
        runCatching { JSONObject(m.optString("software")).optString("name") }.getOrNull()?.takeIf { it.isNotEmpty() }?.let { o.put("by", it) }
        runCatching { o.put("steps", JSONObject(m.optString("training_info")).optInt("step")) }
        // ss_tag_frequency: {folder: {tag: count}} -- the most used tags are what the LoRA answers to
        runCatching {
            val tf = JSONObject(m.optString("ss_tag_frequency")); val all = HashMap<String, Int>()
            for (f in tf.keys()) { val g = tf.getJSONObject(f); for (t in g.keys()) all[t] = (all[t] ?: 0) + g.getInt(t) }
            val top = all.entries.sortedByDescending { it.value }.map { it.key.take(40) }.filter { it.isNotBlank() }.take(8)
            if (top.isNotEmpty()) o.put("triggers", org.json.JSONArray(top))
        }
        o
    }

    /** the largest rank among a converted folder's pairs (A [rank x in]) */
    fun rankOf(dir: File, anima: Boolean): Int =
        (dir.listFiles { f -> f.name.endsWith(".A.f16") } ?: emptyArray()).maxOfOrNull { f ->
            val n = f.name.removeSuffix(".A.f16"); val inD = inDim(n, anima) ?: return@maxOfOrNull 0
            (f.length() / 2 / inD).toInt() } ?: 0

    /**
     * Which model a LoRA file is for, from its tensor names and sizes: "anima", "krea", or null when it is neither. The app
     * converts a picked file for the model it is made for, whichever model the main screen happens to show.
     */
    fun detect(src: File): String? = RandomAccessFile(src, "r").use { raf ->
        val ts = Reader(raf).ts
        fun animaKey(b: String) = ANIMA_KEY.find(b)?.groupValues?.get(2)?.replace('_', '.')?.let { m -> ANIMA.keys.any { it.replace('_', '.') == m } } ?: false
        val bases = ts.keys.map { it.replace(Regex("\\.(lora_(A|B|down|up)(\\.default)?\\.weight|lokr_\\w+|hada_\\w+|dora_scale|alpha)$"), "") }.toSet()
        // SDXL (kohya's SGM keys: lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q, middle_block, output_blocks)
        if (bases.any { SDXL_KEY.containsMatchIn(it) }) return@use "sdxl"
        var anima = 0; var krea = 0; var moss = 0; var klein = 0
        for (b in bases) {
            if (KLEIN_KEY.matches(b) && KLEIN_ONLY.containsMatchIn(b)) { klein++; continue }
            if (MOSS_KEY.matches(b)) { moss++; continue }
            if (animaKey(b)) { anima++; continue }
            // a file may touch only the text path (txtfusion.*), which has the same Linear names as a block
            val k = native(b); if ((k.startsWith("blocks.") || k.startsWith("txtfusion.")) && k.split(".").takeLast(2).joinToString(".") in KREA_IO) krea++
        }
        // FLUX.2 Klein 4B (width 3072) or 9B (4096): a 9B LoRA used to go to lora/klein, where the 9B never looks. No 4B
        // Linear has a side of 4096 (3072, 9216, 12288, 7680), every 9B block Linear has one
        val kb = bases.filter { KLEIN_KEY.matches(it) && KLEIN_ONLY.containsMatchIn(it) }
        val k9 = klein > 0 && ts.entries.any { (n, t) -> t.shape.contains(4096) && kb.any { n.startsWith(it) } }
        when { k9 && klein >= anima && klein >= krea -> "klein9"; klein > 0 && klein >= anima && klein >= krea -> "klein"; moss > 0 && moss >= anima && moss >= krea -> "moss"; anima > 0 && anima >= krea -> "anima"; krea > 0 -> "krea"; else -> null }
    }

    // ---------------------------------------------------------------- SDXL
    // kohya's SGM keys; the transformer blocks' Linears are what the engine takes, proj_in/out, convolutions and the text encoders are not
    private val SDXL_KEY = Regex("^(lora_unet_|unet\\.|base_model\\.model\\.)?(input_blocks|middle_block|output_blocks|down_blocks|mid_block|up_blocks)[._].*transformer_blocks")
    private val SDXL_LIN = Regex("^lora_unet_(input_blocks_\\d+_1|middle_block_1|output_blocks_\\d+_1)_transformer_blocks_(\\d+)_(attn[12]_to_(?:q|k|v|out_0)|ff_net_0_proj|ff_net_2)$")
    /** any of kohya's SGM names, kohya's diffusers names (down_blocks_1_attentions_0) or PEFT's (unet.down_blocks.1.attentions.0...) as kohya's SGM one */
    private fun sdxlCanon(base0: String): String {
        var b = base0.removePrefix("base_model.model.").removePrefix("unet.").removePrefix("lora_unet_").replace('.', '_')
        b = Regex("^down_blocks_(\\d)_attentions_(\\d)_").replace(b) { m -> "input_blocks_${3 * m.groupValues[1].toInt() + m.groupValues[2].toInt() + 1}_1_" }
        b = Regex("^up_blocks_(\\d)_attentions_(\\d)_").replace(b) { m -> "output_blocks_${3 * m.groupValues[1].toInt() + m.groupValues[2].toInt()}_1_" }
        b = b.replace(Regex("^mid_block_attentions_0_"), "middle_block_1_")
        return "lora_unet_$b"
    }
    private val SDXL_DOWN = listOf(".lora_down.weight", ".lora_A.weight", ".lora_A.default.weight")
    private fun sdxlUp(down: String) = down.replace(".lora_down.", ".lora_up.").replace(".lora_A.", ".lora_B.")
    private fun sdxlModule(base: String): String? = SDXL_LIN.find(sdxlCanon(base))?.let { m ->
        val b = m.groupValues[1].replace(Regex("^(input_blocks|output_blocks)_(\\d+)_1$"), "$1.$2.1").replace("middle_block_1", "middle_block.1")
        "$b.transformer_blocks.${m.groupValues[2]}.${m.groupValues[3]}" }
    /** an SDXL LoRA kept as it is (sdxl.safetensors in [out]); [buildSdxl] makes the engine's blobs for the picture's set and strengths */
    fun convertSdxl(src: File, out: File): Result {
        val (used, left) = RandomAccessFile(src, "r").use { raf -> val ts = Reader(raf).ts
            val downs = ts.keys.filter { k -> SDXL_DOWN.any { k.endsWith(it) } }
            // SD 1.5's LoRAs have the same diffusers names: the cross attention's k takes SDXL's 2048-wide text, SD 1.5's 768
            val k2 = downs.firstOrNull { sdxlModule(SDXL_DOWN.fold(it) { a, x -> a.removeSuffix(x) })?.endsWith("attn2_to_k") == true }
            if (k2 != null && ts.getValue(k2).shape.last() != 2048) error(UNSUPPORTED)
            val bases = downs.map { d -> SDXL_DOWN.fold(d) { a, x -> a.removeSuffix(x) } }
            bases.count { sdxlModule(it) != null } to bases.count { sdxlModule(it) == null } }
        if (used == 0) error(UNSUPPORTED)
        out.mkdirs(); src.copyTo(File(out, "sdxl.safetensors"), overwrite = true)
        return Result(used, left)
    }
    /** the converter's fp16 weight layout of m [N x K] (row-major): per (32-row, 32-column) block, k in pairs, the 32 rows inside */
    private fun layHalf(m: FloatArray, N: Int, K: Int): ByteArray {
        val bb = ByteBuffer.allocate(N * K * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (nb in 0 until N / 32) for (kb in 0 until K / 32) for (k16 in 0 until 16) for (n in 0 until 32) for (k2 in 0..1)
            bb.putShort(half(m[(nb * 32 + n) * K + kb * 32 + k16 * 2 + k2]))
        return bb.array()
    }
    private fun loraHead(lr: Int) = ByteArray(128).also { it[0] = (lr and 255).toByte(); it[1] = (lr shr 8).toByte() }
    /**
     * The engine's runtime LoRA for SDXL: [out]/lora.w, a blob per transformer block (unet.tab's op 5 layers), and lora.tab
     * "layer offset bytes". A blob is the block's eleven Linears (attn1 q k v o, attn2 q k v o, the GEGLU's up half, its gate half, down),
     * each { lr } then A^T, { lr } then B^T in the weight layout, lr 0 for a Linear no LoRA touches; several LoRAs side by side (ranks added,
     * up to 512, padded to 64); strength * alpha / rank and q's log2 e / sqrt(64) in B. Returns the Linears that have one.
     */
    fun buildSdxl(unetTab: File, sets: List<Pair<File, Float>>, out: File): Int {
        val order = listOf("input_blocks.4.1" to 2, "input_blocks.5.1" to 2, "input_blocks.7.1" to 10, "input_blocks.8.1" to 10, "middle_block.1" to 10,
            "output_blocks.0.1" to 10, "output_blocks.1.1" to 10, "output_blocks.2.1" to 10, "output_blocks.3.1" to 2, "output_blocks.4.1" to 2, "output_blocks.5.1" to 2)
        val blocks = order.flatMap { (k, n) -> (0 until n).map { "$k.transformer_blocks.$it" } }
        val rows = unetTab.readLines().filter { it.isNotBlank() }.map { it.trim().split(Regex("\\s+")) }
        val op5 = rows.indices.filter { rows[it][0] == "5" }; require(op5.size == blocks.size) { "unet.tab: ${op5.size} transformer blocks" }
        class E(val rd: Reader, val down: T, val up: T, val scale: Float)
        val raf = sets.map { RandomAccessFile(it.first, "r") }
        try {
            val mods = HashMap<String, MutableList<E>>()
            for ((i, set) in sets.withIndex()) { val rd = Reader(raf[i]); val ts = rd.ts
                for (k in ts.keys) { val suf = SDXL_DOWN.firstOrNull { k.endsWith(it) } ?: continue; val base = k.removeSuffix(suf); val mod = sdxlModule(base) ?: continue
                    val d = ts.getValue(k); val u = ts[sdxlUp(k)] ?: continue; if (d.shape.size != 2) continue
                    // alpha / rank: the .alpha tensor (kohya), else PEFT's adapter metadata (rsLoRA too), else 1
                    val r = d.shape[0]; val sc = rd.scale(base, "$base.alpha".takeIf { it in ts }, r)
                    mods.getOrPut(mod) { ArrayList() }.add(E(rd, d, u, set.second * sc)) } }
            val lin = listOf("attn1_to_q", "attn1_to_k", "attn1_to_v", "attn1_to_out_0", "attn2_to_q", "attn2_to_k", "attn2_to_v", "attn2_to_out_0", "ff_up", "ff_gate", "ff_net_2")
            val qs = (1.0 / Math.log(2.0) / 8.0).toFloat()   // log2 e / sqrt(64)
            out.mkdirs(); var used = 0; val tab = StringBuilder()
            java.io.BufferedOutputStream(java.io.FileOutputStream(File(out, "lora.w")), 1 shl 20).use { fw ->
                var off = 0L
                for ((bi, blk) in blocks.withIndex()) {
                    val c = rows[op5[bi]][1].toInt(); val f = 4 * c
                    val kn = listOf(c to c, c to c, c to c, c to c, c to c, 2048 to c, 2048 to c, c to c, c to f, c to f, f to c); var len = 0L
                    for ((li, name) in lin.withIndex()) {
                        val (K, N) = kn[li]; val L = mods["$blk.${if (name == "ff_up" || name == "ff_gate") "ff_net_0_proj" else name}"].orEmpty()
                        if (L.isEmpty()) { fw.write(loraHead(0)); len += 128; continue }
                        used++
                        val rs = L.map { it.down.shape[0] }; val R = rs.sum(); require(R <= 512) { "LoRA rank $R past 512" }; val lr = (R + 63) / 64 * 64
                        val dp = FloatArray(lr * K); val up = FloatArray(N * lr); var r0 = 0
                        for ((ei, e) in L.withIndex()) {
                            val r = rs[ei]; val d = e.rd.read(e.down); System.arraycopy(d, 0, dp, r0 * K, r * K)
                            val u = e.rd.read(e.up); val n0 = if (name == "ff_gate") f else 0; val sc = e.scale * if (name.endsWith("to_q")) qs else 1f
                            for (n in 0 until N) for (j in 0 until r) up[n * lr + r0 + j] = u[(n0 + n) * r + j] * sc
                            r0 += r
                        }
                        val a = layHalf(dp, lr, K); val b = layHalf(up, N, lr)
                        fw.write(loraHead(lr)); fw.write(a); fw.write(loraHead(lr)); fw.write(b); len += 256L + a.size + b.size
                    }
                    tab.append("${op5[bi]} $off $len\n"); off += len
                }
            }
            File(out, "lora.tab").writeText(tab.toString())
            return used
        } finally { raf.forEach { runCatching { it.close() } } }
    }

    // ---------------------------------------------------------------- MOSS-Image
    // ---------------------------------------------------------------- FLUX.2 Klein
    /** names only FLUX.2 Klein has: its fused / added projections and its own block names (a diffusers Krea LoRA shares "transformer_blocks.N.attn.to_q") */
    private val KLEIN_ONLY = Regex("to_qkv_mlp_proj|linear1|linear2|ff[.]linear_in|ff[.]linear_out|ff_context|add_[qkv]_proj|to_add_out|img_attn|txt_attn|img_mlp|txt_mlp|^single_|^double_blocks|single_transformer_blocks|(^|[.])single_blocks")
    private val KLEIN_KEY = Regex("^(?:diffusion_model[.]|transformer[.]|model[.]diffusion_model[.])?((?:transformer_blocks|single_transformer_blocks|double_blocks|single_blocks)[.]([0-9]+)[.]([a-z_0-9.]+))$")
    /** plain pairs under the names the engine reads: kb.d<i>.<0 text, 1 image>.<q k v o g u d>, kb.s<i>.<...>; the fused Linears cut by rows or columns */
    fun convertKlein(src: File, out: File): Result = RandomAccessFile(src, "r").use { raf ->
        val rd = Reader(raf); val ts = rd.ts; out.mkdirs(); var done = 0; var skipped = 0
        // the width from the file's own fused Linears: 4B 3072, 9B 4096
        val D = rd.pairs().entries.firstOrNull { it.key.endsWith("to_qkv_mlp_proj") || it.key.endsWith("linear1") || it.key.endsWith("attn.to_q") || it.key.endsWith("attn.qkv") }?.let { ts[it.value.first]!!.shape[1] } ?: 3072; val F = 3 * D
        fun put(n: String, a: FloatArray, b: FloatArray, sc: Float) { write(File(out, "$n.A.f16"), a); write(File(out, "$n.B.f16"), b, sc); done++ }
        // the Linears outside the blocks are computed on the CPU ([KleinGen]), a LoRA on them at any rank
        val outer = mapOf("x_embedder" to "x_embedder", "img_in" to "x_embedder", "context_embedder" to "context_embedder", "txt_in" to "context_embedder",
            "time_guidance_embed.timestep_embedder.linear_1" to "time_guidance_embed.timestep_embedder.linear_1", "time_in.in_layer" to "time_guidance_embed.timestep_embedder.linear_1",
            "time_guidance_embed.timestep_embedder.linear_2" to "time_guidance_embed.timestep_embedder.linear_2", "time_in.out_layer" to "time_guidance_embed.timestep_embedder.linear_2",
            "double_stream_modulation_img.linear" to "double_stream_modulation_img.linear", "double_stream_modulation_img.lin" to "double_stream_modulation_img.linear",
            "double_stream_modulation_txt.linear" to "double_stream_modulation_txt.linear", "double_stream_modulation_txt.lin" to "double_stream_modulation_txt.linear",
            "single_stream_modulation.linear" to "single_stream_modulation.linear", "single_stream_modulation.lin" to "single_stream_modulation.linear",
            "norm_out.linear" to "norm_out.linear", "proj_out" to "proj_out", "final_layer.linear" to "proj_out")
        for ((b0, p) in rd.pairs()) {
            outer[b0.removePrefix("transformer.").removePrefix("diffusion_model.").removePrefix("model.diffusion_model.")]?.let { n ->
                val ta = ts[p.first]!!; val tb = ts[p.second]!!
                put("outer.$n", rd.read(ta), rd.read(tb), rd.scale(b0, p.third, ta.shape[0])); return@let true } ?.let { continue }
            val m = KLEIN_KEY.matchEntire(b0) ?: run { skipped++; null } ?: continue
            val single = m.groupValues[1].startsWith("single"); val i = m.groupValues[2]; val l = m.groupValues[3]
            val ta = ts[p.first]!!; val tb = ts[p.second]!!
            if (ta.shape.size != 2 || tb.shape.size != 2 || ta.shape[0] != tb.shape[1]) { skipped++; continue }
            val r = ta.shape[0]; val K = ta.shape[1]; val N = tb.shape[0]; val sc = rd.scale(b0, p.third, r); val a = rd.read(ta); val b = rd.read(tb)
            fun rows(n0: Int, n: Int) = b.copyOfRange(n0 * r, (n0 + n) * r)
            fun colsA(c0: Int, n: Int) = FloatArray(r * n) { a[(it / n) * K + c0 + it % n] }
            val pre = if (single) "kb.s$i" else "kb.d$i"
            when {
                single && (l == "attn.to_qkv_mlp_proj" || l == "linear1") && N == 3 * D + 2 * F -> { put("$pre.q", a, rows(0, D), sc); put("$pre.k", a, rows(D, D), sc); put("$pre.v", a, rows(2 * D, D), sc); put("$pre.g", a, rows(3 * D, F), sc); put("$pre.u", a, rows(3 * D + F, F), sc) }
                single && (l == "attn.to_out" || l == "linear2") && K == D + F -> { put("$pre.o", colsA(0, D), b, sc); put("$pre.d", colsA(D, F), b, sc) }
                !single -> {
                    val s = if (l.startsWith("txt") || l.startsWith("ff_context") || "add" in l) 0 else 1; val ps = "$pre.$s"
                    when (l) {
                        "img_attn.qkv", "txt_attn.qkv" -> { put("$ps.q", a, rows(0, D), sc); put("$ps.k", a, rows(D, D), sc); put("$ps.v", a, rows(2 * D, D), sc) }
                        "attn.to_q", "attn.add_q_proj" -> put("$ps.q", a, b, sc); "attn.to_k", "attn.add_k_proj" -> put("$ps.k", a, b, sc); "attn.to_v", "attn.add_v_proj" -> put("$ps.v", a, b, sc)
                        "attn.to_out.0", "attn.to_add_out", "img_attn.proj", "txt_attn.proj" -> put("$ps.o", a, b, sc)
                        "ff.linear_in", "ff_context.linear_in", "img_mlp.0", "txt_mlp.0" -> { put("$ps.g", a, rows(0, F), sc); put("$ps.u", a, rows(F, F), sc) }
                        "ff.linear_out", "ff_context.linear_out", "img_mlp.2", "txt_mlp.2" -> put("$ps.d", a, b, sc)
                        else -> skipped++
                    }
                }
                else -> skipped++
            }
        }
        if (done == 0) { out.deleteRecursively(); error(UNSUPPORTED) }
        File(out, "info.json").writeText(infoOf(src).put("source", src.name).put("pairs", done).put("skipped", skipped).toString())
        Result(done, skipped)
    }

    private val MOSS_KEY = Regex("^(?:diffusion_model[.]|transformer[.])?layers[.]([0-9]+)[.]proj[.](q|k|v|qkv|o|ffn_gate|ffn_up|ffn_down|ffn_up_gate)$")
    /** plain pairs under the names the engine reads: layers.<i>.proj.<name>, ffn_up_gate's B split in two as the release's chunk(2) takes it */
    fun convertMoss(src: File, out: File): Result = RandomAccessFile(src, "r").use { raf ->
        val rd = Reader(raf); val ts = rd.ts; out.mkdirs(); var done = 0; var skipped = 0
        for ((b0, p) in rd.pairs()) {
            val m = MOSS_KEY.matchEntire(b0) ?: run { skipped++; null } ?: continue
            val ta = ts[p.first]!!; val tb = ts[p.second]!!
            if (ta.shape.size != 2 || tb.shape.size != 2 || ta.shape[0] != tb.shape[1]) { skipped++; continue }
            val name = "layers.${m.groupValues[1]}.proj."; val mod = m.groupValues[2]; val scale = rd.scale(b0, p.third, ta.shape[0])
            // fused ones split as kconv splits the weights: ffn_up_gate = up | gate (halves), qkv = q | k | v (thirds; HaDeS-MOSS-Turbo's)
            if (mod == "ffn_up_gate" || mod == "qkv") {
                val a = rd.read(ta); val b = rd.read(tb)
                val names = if (mod == "qkv") listOf("q", "k", "v") else listOf("ffn_up", "ffn_gate"); val part0 = b.size / names.size
                for ((nm, part) in names.mapIndexed { j, nm -> nm to b.copyOfRange(j * part0, (j + 1) * part0) }) { write(File(out, "$name$nm.A.f16"), a); write(File(out, "$name$nm.B.f16"), part, scale); done++ }
            } else { rd.toHalf(ta, File(out, "$name$mod.A.f16")); rd.toHalf(tb, File(out, "$name$mod.B.f16"), scale); done++ }
        }
        if (done == 0) { out.deleteRecursively(); error(UNSUPPORTED) }
        File(out, "info.json").writeText(infoOf(src).put("source", src.name).put("pairs", done).put("skipped", skipped).toString())
        Result(done, skipped)
    }

    // ---------------------------------------------------------------- Anima
    /** The ten Linears of an Anima block that take an adapter, by the generator's slot key, and their (in, out). */
    val ANIMA = linkedMapOf("self_attn.q_proj" to "sa_q", "self_attn.k_proj" to "sa_k", "self_attn.v_proj" to "sa_v", "self_attn.output_proj" to "sa_o",
        "cross_attn.q_proj" to "ca_q", "cross_attn.k_proj" to "ca_k", "cross_attn.v_proj" to "ca_v", "cross_attn.output_proj" to "ca_o",
        "mlp.layer1" to "mlp1", "mlp.layer2" to "mlp2")
    val ANIMA_IO = mapOf("sa_q" to (2048 to 2048), "sa_k" to (2048 to 2048), "sa_v" to (2048 to 2048), "sa_o" to (2048 to 2048), "ca_q" to (2048 to 2048),
        "ca_k" to (1024 to 2048), "ca_v" to (1024 to 2048), "ca_o" to (2048 to 2048), "mlp1" to (2048 to 8192), "mlp2" to (8192 to 2048))
    const val ANIMA_RANK = 64
    /** a Krea 2 block matrix's in-dimension, by the name [convert] writes */
    val KREA_IN = mapOf("attn.wq" to 6144, "attn.wk" to 6144, "attn.wv" to 6144, "attn.gate" to 6144, "attn.wo" to 6144,
        "mlp.gate" to 6144, "mlp.up" to 6144, "mlp.down" to 16384)
    /** the in-dimension of a converted matrix by its file name without .A.f16: Anima "b<i>_<key>", Krea "blocks.<i>.<mat>" */
    private fun inDim(name: String, anima: Boolean) = when {
        name.startsWith("adaln_") -> if (name.endsWith(".1")) 2048 else if (name.endsWith(".2")) ADALN_MID else null   // [Adaln]
        anima -> ANIMA_IO[name.substringAfter('_')]?.first
        name.startsWith("txtfusion.") || name.startsWith("txtmlp.") -> textIO(name)?.first
        else -> KREA_IN[name.substringAfter("blocks.").substringAfter('.')]
    }
    private fun readF16(f: File): FloatArray {
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        return FloatArray(b.limit()) { unhalf(b.get(it)) }
    }

    /**
     * Several converted LoRAs as one folder, by putting their ranks side by side: A is the A's stacked ([r1 + r2, in]) and B the
     * B's with each one's strength folded in ([out, r1 + r2]), so a run with the one folder makes exactly the sum of the changes.
     * Anima's 64 slots cap the total (the ones that do not fit are left out, latest first); a Krea bake takes any rank.
     * Cached by [out]/tag.txt. Returns the number of matrices written, 0 when nothing matched.
     */
    fun mergeLoras(libs: List<Pair<File, Float>>, out: File, anima: Boolean, maxRank: Int): Int {
        val tag = libs.joinToString("|") { "${it.first.name}:${it.second}:${File(it.first, "info.json").lastModified()}" } + "|$maxRank"
        if (runCatching { File(out, "tag.txt").readText() }.getOrNull() == tag) return -1
        out.deleteRecursively(); out.mkdirs()
        val names = libs.flatMap { (d, _) -> (d.listFiles { f -> f.name.endsWith(".A.f16") } ?: emptyArray()).map { it.name.removeSuffix(".A.f16") } }.distinct()
        var written = 0
        for (n in names) {
            val inD = inDim(n, anima) ?: continue
            val cap = if (n.startsWith("adaln_")) Int.MAX_VALUE else maxRank   // the modulation Linears are not the generator's 64 slots
            val parts = ArrayList<Triple<FloatArray, FloatArray, Float>>(); var outD = 0; var total = 0
            for ((d, s) in libs) {
                val fa = File(d, "$n.A.f16"); val fb = File(d, "$n.B.f16")
                if (!fa.isFile || !fb.isFile) continue
                val r = (fa.length() / 2 / inD).toInt()
                if (r < 1 || fa.length() != r.toLong() * inD * 2 || fb.length() % (r * 2L) != 0L) continue
                val o = (fb.length() / 2 / r).toInt()
                if (outD == 0) outD = o else if (o != outD) continue
                if (total + r > cap) continue                          // no room left: this one is left out of this matrix
                parts.add(Triple(readF16(fa), readF16(fb), s)); total += r
            }
            if (parts.isEmpty() || outD == 0 || total == 0) continue
            val a = FloatArray(total * inD); val b = FloatArray(outD * total)
            var off = 0
            for ((pa, pb, s) in parts) {
                val r = pa.size / inD
                System.arraycopy(pa, 0, a, off * inD, r * inD)
                for (y in 0 until outD) for (k in 0 until r) b[y * total + off + k] = pb[y * r + k] * s
                off += r
            }
            write(File(out, "$n.A.f16"), a); write(File(out, "$n.B.f16"), b)
            written++
        }
        // the modulation Linears are merged like the rest above, so their strengths are already folded into B
        if (out.list { _, nm -> nm.startsWith("adaln_") }?.isNotEmpty() == true) File(out, "adaln_strength.txt").writeText("1.0\n")
        File(out, "info.json").writeText("{\"merged\": ${libs.size}}")
        File(out, "tag.txt").writeText(tag)
        return written
    }
    const val UNSUPPORTED = "unsupported LoRA format"
    private val ANIMA_ADALN = Regex("^(?:lora_unet_|diffusion_model\\.|model\\.diffusion_model\\.)blocks[._](\\d+)[._]adaln_modulation_(self_attn|cross_attn|mlp)[._]([12])$")
    private const val ADALN_MID = 256
    private val ANIMA_KEY = Regex("^(?:lora_unet_|diffusion_model\\.|model\\.diffusion_model\\.)blocks[._](\\d+)[._](.+)$")

    /**
     * An Anima LoRA (ComfyUI or kohya key names) -> b<i>_<key>.A.f16 [rank,in] and .B.f16 [out,rank], alpha/rank folded into B.
     *
     * Everything goes in exactly, for the engine's bake (kengine's bake_set), which has the base weights: a pair at any rank,
     * a LoKr as its two factors, a DoRA magnitude beside the change.
     */
    fun convertAnima(src: File, out: File): Result = RandomAccessFile(src, "r").use { raf ->
        val rd = Reader(raf); val ts = rd.ts
        out.mkdirs()
        var done = 0; var skipped = 0; var extra = 0
        fun slot(base: String): Triple<String, Int, Int>? {
            val m = ANIMA_KEY.find(base) ?: return null; val mod = m.groupValues[2].replace('_', '.')
            val key = ANIMA.entries.firstOrNull { it.key.replace('_', '.') == mod }?.value ?: return null
            return Triple("b${m.groupValues[1]}_$key", ANIMA_IO[key]!!.first, ANIMA_IO[key]!!.second)
        }
        fun put(name: String, ab: Pair<FloatArray, FloatArray>, scale: Float) { write(File(out, "$name.A.f16"), ab.first); write(File(out, "$name.B.f16"), ab.second, scale); done++ }
        // plain pairs
        for ((b0, p) in rd.pairs()) {
            // an adaLN modulation Linear: no NPU slot, computed on the CPU ([Adaln]) by bundles that take the modulations as
            // inputs, so the pair is kept as it is, at any rank
            val am = ANIMA_ADALN.find(b0)
            if (am != null) {
                val (blk, kind, nth) = am.destructured
                val ta = ts[p.first]!!; val tb = ts[p.second]!!
                val ok = ta.shape.size == 2 && tb.shape.size == 2 && ta.shape[0] == tb.shape[1] &&
                    (if (nth == "1") ta.shape[1] == 2048 && tb.shape[0] == ADALN_MID else ta.shape[1] == ADALN_MID && tb.shape[0] == 3 * 2048)
                if (!ok) { skipped++; continue }
                val n = "adaln_b${blk}_$kind.$nth"
                write(File(out, "$n.A.f16"), rd.read(ta)); write(File(out, "$n.B.f16"), rd.read(tb), rd.scale(b0, p.third, ta.shape[0]))
                extra++; continue
            }
            val (name, i, o) = slot(b0) ?: run { skipped++; null } ?: continue
            val ta = ts[p.first]!!; val tb = ts[p.second]!!
            if (ta.shape.size != 2 || tb.shape.size != 2 || ta.shape[0] != tb.shape[1] || ta.shape[1] != i || tb.shape[0] != o) { skipped++; continue }
            val r = ta.shape[0]; val scale = rd.scale(b0, p.third, r)
            rd.toHalf(ta, File(out, "$name.A.f16")); rd.toHalf(tb, File(out, "$name.B.f16"), scale); done++
            putDora(rd, b0, name, out, o)
        }
        // LoKr: w1 or w1_a w1_b, times w2 or w2_a w2_b, kept as the two factors
        for (b0 in ts.keys.filter { it.endsWith(".lokr_w1") || it.endsWith(".lokr_w1_a") }.map { it.substringBeforeLast('.') }) {
            val (name, i, o) = slot(b0) ?: run { skipped++; null } ?: continue
            if (!putLokr(rd, b0, name, out, o, i)) { skipped++; continue }
            putDora(rd, b0, name, out, o); done++
        }
        if (done == 0) { out.deleteRecursively(); error(if (skipped > 0 || ts.keys.any { "hada_" in it }) UNSUPPORTED else "no LoRA weights for Anima in this file") }
        File(out, "info.json").writeText(infoOf(src).put("source", src.name).put("pairs", done).put("skipped", skipped).put("rank", rankOf(out, true)).put("adaln", extra).toString())
        Result(done, skipped, extra)
    }

    /**
     * What the generator needs of a converted Anima LoRA beside the engine's set: where its modulation Linears are and at what strength
     * (<active>/adaln.txt, read by [Adaln]), noted once for a LoRA and a strength (tag.txt says which).
     */
    fun activateAnima(lib: File, active: File, strength: Float): File {
        val tag = "${lib.name}|${File(lib, "info.json").lastModified()}|$strength"
        if (runCatching { File(active, "tag.txt").readText() }.getOrNull() == tag) return active
        active.deleteRecursively(); active.mkdirs()
        // the generator reads them from the LoRA's own folder, at this strength ([Adaln])
        if (lib.list { _, n -> n.startsWith("adaln_") }?.isNotEmpty() == true) {
            // a merged folder ([mergeLoras]) already has the strengths in B, and keeps the modulation one beside its files
            val ms = runCatching { File(lib, "adaln_strength.txt").readText().trim().toFloat() }.getOrNull() ?: strength
            File(active, "adaln.txt").writeText("${lib.absolutePath}\n$ms")
        }
        File(active, "tag.txt").writeText(tag)
        return active
    }
}
