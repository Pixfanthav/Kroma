package dev.kroma.open

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Tag snapping for the models that read Danbooru tags (Anima, SDXL): a translated prompt's short comma items (4 words or fewer) are put
 * in one space with the tag list by EmbeddingGemma 2 (Google, Apache 2.0; the onnx-community int8 export on the CPU, never fp16: it
 * overflows), and an item close enough to a real tag (cosine >= 0.955) becomes that tag; the rest stays as written, so tags and natural
 * language mix as Anima takes them. No tag is made up: only the list's (the 20000 most used, Danbooru via deepghs/danbooru_wikis_full).
 *
 * Files in <models>/tags: model_quantized.onnx (+ model_quantized.onnx_data), tok.bin (the tokenizer cut to ASCII pieces, exact for ASCII
 * input: an item with other letters is left alone), tags_e256.f16 [20000 x 256] (the tags' vectors, first 256 dims normalized),
 * tags_e256.txt (their names). The vectors were made with the same model and prefix ("task: sentence similarity | query: ").
 */
object TagSnap {
    private const val PRE = "task: sentence similarity | query: "
    private const val THR = 0.955f
    private const val DIM = 256

    fun present(root: File) = File(root, "tags/model_quantized.onnx").isFile && File(root, "tags/tok.bin").isFile && File(root, "tags/tags_e256.f16").isFile

    // ---- the tokenizer: BPE over the whole text (spaces as U+2581), the merges by rank, unknown letters as UTF-8 byte tokens, <bos> ... <eos>
    private class Tok(f: File) {
        val charId = HashMap<Int, Int>(); val byteId = IntArray(256)
        private val cap = 1 shl 20; private val keys = LongArray(cap) { -1L }; private val rank = IntArray(cap); private val res = IntArray(cap)
        init {
            val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            repeat(b.int) { charId[b.int] = b.int }
            val n = b.int
            for (r in 0 until n) { val a = b.int; val c = b.int; val ab = b.int; val k = (a.toLong() shl 32) or (c.toLong() and 0xffffffffL); var i = slot(k); while (keys[i] != -1L && keys[i] != k) i = (i + 1) and (cap - 1)
                                   if (keys[i] == -1L) { keys[i] = k; rank[i] = r; res[i] = ab } }
            for (i in 0 until 256) byteId[i] = b.int
        }
        private fun slot(k: Long) = ((k * -7046029254386353131L) ushr 44).toInt() and (cap - 1)
        private fun find(a: Int, c: Int): Int { val k = (a.toLong() shl 32) or (c.toLong() and 0xffffffffL); var i = slot(k); while (keys[i] != -1L) { if (keys[i] == k) return i; i = (i + 1) and (cap - 1) }; return -1 }
        fun encode(s: String): LongArray {
            val sym = ArrayList<Int>()
            for (ch in s.replace(' ', '▁')) { val id = charId[ch.code]; if (id != null) sym.add(id) else for (x in ch.toString().toByteArray()) sym.add(byteId[x.toInt() and 255]) }
            while (sym.size > 1) {
                var best = -1; var bi = -1
                for (i in 0 until sym.size - 1) { val j = find(sym[i], sym[i + 1]); if (j >= 0 && (best < 0 || rank[j] < rank[best])) { best = j; bi = i } }
                if (best < 0) break
                sym[bi] = res[best]; sym.removeAt(bi + 1)
            }
            return LongArray(sym.size + 2).also { it[0] = 2; for (i in sym.indices) it[i + 1] = sym[i].toLong(); it[sym.size + 1] = 1 }
        }
    }

    private class Index(root: File) {
        val tok = Tok(File(root, "tags/tok.bin"))
        val names = File(root, "tags/tags_e256.txt").readLines().filter { it.isNotEmpty() }
        val nameSet = names.toHashSet()
        /** Danbooru's own aliases (tags/aliases.tsv, active ones whose tag is in the list: maid outfit -> maid, twin tails -> twintails) */
        val alias: Map<String, String> = runCatching { File(root, "tags/aliases.tsv").readLines().mapNotNull { l -> l.split('\t').takeIf { it.size == 2 }?.let { it[0] to it[1] } }.toMap() }.getOrDefault(emptyMap())
        val vec: FloatArray = ByteBuffer.wrap(File(root, "tags/tags_e256.f16").readBytes()).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().let { s ->
            FloatArray(s.remaining()) { android.util.Half.toFloat(s.get(it)) } }
        val env: OrtEnvironment = OrtEnvironment.getEnvironment()
        val sess: OrtSession = env.createSession(File(root, "tags/model_quantized.onnx").absolutePath, OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) })
    }
    @Volatile private var index: Pair<String, Index>? = null
    private fun index(root: File): Index = index?.takeIf { it.first == root.absolutePath }?.second ?: synchronized(this) {
        index?.takeIf { it.first == root.absolutePath }?.second ?: Index(root).also { index = root.absolutePath to it }
    }

    /** the items' vectors [n x 256], normalized */
    private fun embed(ix: Index, items: List<String>): Array<FloatArray> {
        val ids = items.map { ix.tok.encode(PRE + it) }; val L = ids.maxOf { it.size }
        val a = LongArray(items.size * L); val m = LongArray(items.size * L)
        for ((k, v) in ids.withIndex()) for (i in v.indices) { a[k * L + i] = v[i]; m[k * L + i] = 1 }
        val env = ix.env; val shape = longArrayOf(items.size.toLong(), L.toLong()); val none = longArrayOf(0, 512)
        OnnxTensor.createTensor(env, LongBuffer.wrap(a), shape).use { ti -> OnnxTensor.createTensor(env, LongBuffer.wrap(m), shape).use { tm ->
        OnnxTensor.createTensor(env, FloatBuffer.allocate(0), none).use { e1 -> OnnxTensor.createTensor(env, FloatBuffer.allocate(0), none).use { e2 ->
        OnnxTensor.createTensor(env, FloatBuffer.allocate(0), none).use { e3 ->
            ix.sess.run(mapOf("input_ids" to ti, "attention_mask" to tm, "image_features" to e1, "video_features" to e2, "audio_features" to e3), setOf("sentence_embedding")).use { r ->
                val b = (r[0] as OnnxTensor).floatBuffer; val w = b.remaining() / items.size
                return Array(items.size) { k -> FloatArray(DIM) { b.get(k * w + it) }.also { v -> val n = Math.sqrt(v.sumOf { (it * it).toDouble() }).toFloat(); for (i in v.indices) v[i] /= n } }
            } } } } } }
    }

    /** text with its short ASCII comma items snapped to the tag list where one is close enough (unchanged on any failure) */
    fun snap(root: File, text: String, log: (String) -> Unit): String {
        if (!present(root)) return text
        val t0 = System.currentTimeMillis()
        return runCatching {
            val ix = index(root)
            val parts = text.split(",").map { it.trim() }.filter { it.isNotEmpty() }
            // first a tag as it is (any case, _ or spaces) or one of its aliases; the model only for the rest
            val to = HashMap<String, String>()
            fun key(p: String) = p.lowercase().replace('_', ' ').split(' ').filter { it.isNotEmpty() }.joinToString(" ")
            for (p in parts) { val k = key(p); val t = if (k in ix.nameSet) k else ix.alias[k]; if (t != null && t != p) to[p] = t }
            val short = parts.filter { p -> p !in to && key(p) !in ix.nameSet && p.split(' ').count { it.isNotEmpty() } <= 4 && p.all { it.code < 128 } }.distinct()
            if (short.isEmpty()) { log("tags ${System.currentTimeMillis() - t0} ms: " + (if (to.isEmpty()) "no change" else to.entries.joinToString { "${it.key} -> ${it.value}" })); return parts.joinToString(", ") { to[it] ?: it } }
            val e = embed(ix, short); val n = ix.names.size.coerceAtMost(ix.vec.size / DIM)
            for ((k, item) in short.withIndex()) {
                var best = -1f; var bi = -1; val q = e[k]
                for (t in 0 until n) { var d = 0f; val o = t * DIM; for (i in 0 until DIM) d += q[i] * ix.vec[o + i]; if (d > best) { best = d; bi = t } }
                if (bi >= 0 && best >= THR && !ix.names[bi].equals(item, ignoreCase = false)) to[item] = ix.names[bi]
            }
            val out = parts.joinToString(", ") { to[it] ?: it }
            log("tags (EmbeddingGemma) ${System.currentTimeMillis() - t0} ms: " + (if (to.isEmpty()) "no change" else to.entries.joinToString { "${it.key} -> ${it.value}" }))
            out
        }.getOrElse { log("tags failed: ${it.message}"); text }
    }
}
