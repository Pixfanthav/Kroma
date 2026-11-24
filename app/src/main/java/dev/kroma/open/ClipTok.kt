package dev.kroma.open

import java.io.File
import java.text.Normalizer
import java.util.regex.Pattern

/**
 * SDXL's CLIP tokenizer (L and G share it), as HF tokenizers runs it for ComfyUI: NFC -> whitespace runs to one space -> lowercase ->
 * CLIP's split (contractions, letter runs, single digits, other runs) -> ByteLevel -> BPE with "</w>" on each piece's last symbol.
 * Data: clip_vocab.json, clip_merges.txt (from ComfyUI's sd1_tokenizer). ComfyUI's "(word:1.2)" weights are not read: brackets are text.
 */
class ClipTok(dir: File) {
    private val vocab = HashMap<String, Int>(50_000)
    private val ranks = HashMap<String, Int>(50_000)
    private val cache = HashMap<String, IntArray>()

    init {
        val j = org.json.JSONObject(File(dir, "clip_vocab.json").readText())
        for (k in j.keys()) vocab[k] = j.getInt(k)
        File(dir, "clip_merges.txt").bufferedReader().useLines { ls -> ls.filter { !it.startsWith("#version") && it.isNotBlank() }.forEachIndexed { i, m -> ranks[m] = i } }
    }

    /** the text's tokens, no start or end */
    fun encode(text: String): IntArray {
        val out = ArrayList<Int>()
        val norm = Normalizer.normalize(text, Normalizer.Form.NFC).replace(WS, " ").lowercase()
        val m = SPLIT.matcher(norm)
        while (m.find()) {
            val sb = StringBuilder()
            for (b in m.group().toByteArray(Charsets.UTF_8)) sb.append(BYTE2CHAR[b.toInt() and 0xFF])
            val w = sb.toString()
            for (id in cache.getOrPut(w) { bpe(w) }) out.add(id)
        }
        return out.toIntArray()
    }

    /** a long prompt as ComfyUI takes it: 75 tokens a chunk, each its own [ids77] (start, the tokens, end, pad), at most [most] chunks */
    fun chunks77(text: String, pad: Int, most: Int = 4): List<LongArray> {
        val t = encode(text); val k = ((t.size + 74) / 75).coerceIn(1, most)
        return List(k) { c -> val s = c * 75; val n = (t.size - s).coerceIn(0, 75)
            LongArray(77) { i -> when { i == 0 -> START.toLong(); i <= n -> t[s + i - 1].toLong(); i == n + 1 -> END.toLong(); else -> pad.toLong() } } }
    }

    /** start, the first 75 tokens, end, then [pad] up to 77: L pads with the end token (49407), G with 0 */
    fun ids77(text: String, pad: Int): LongArray {
        val t = encode(text); val n = minOf(t.size, 75)
        return LongArray(77) { i -> when { i == 0 -> START.toLong(); i <= n -> t[i - 1].toLong(); i == n + 1 -> END.toLong(); else -> pad.toLong() } }
    }

    // the merge loop as Qwen's (QwenTok), on symbols whose last one carries "</w>"
    private fun bpe(word: String): IntArray {
        val cps = word.codePoints().toArray().map { String(Character.toChars(it)) }
        var parts = if (cps.isEmpty()) cps else cps.dropLast(1) + (cps.last() + "</w>")
        while (parts.size > 1) {
            var best = -1; var bestRank = Int.MAX_VALUE
            for (i in 0 until parts.size - 1) {
                val r = ranks["${parts[i]} ${parts[i + 1]}"] ?: continue
                if (r < bestRank) { bestRank = r; best = i }
            }
            if (best < 0) break
            val a = parts[best]; val b = parts[best + 1]
            val next = ArrayList<String>(parts.size)
            var i = 0
            while (i < parts.size) {
                if (i < parts.size - 1 && parts[i] == a && parts[i + 1] == b) { next.add(a + b); i += 2 }
                else { next.add(parts[i]); i++ }
            }
            parts = next
        }
        return IntArray(parts.size) { vocab[parts[it]] ?: END }   // the unknown token is the end token, as the tokenizer's model has it
    }

    companion object {
        const val START = 49406; const val END = 49407
        // \s as Unicode's White_Space, as the tokenizers' regexes have it, spelled out: Android's ICU has no (?U) and the JVM reads
        // \p{White_Space} otherwise (both failed), a plain class is the same on both
        private const val SP = "\\t\\n\\u000B\\f\\r \\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000"
        private val WS = Regex("[$SP]+")
        private val SPLIT: Pattern = Pattern.compile("<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^$SP\\p{L}\\p{N}]+")
        // GPT2 bytes_to_unicode (as QwenTok's)
        private val BYTE2CHAR = CharArray(256).also { t ->
            val direct = (33..126) + (161..172) + (174..255)
            var n = 0
            for (b in 0 until 256) t[b] = if (b in direct) b.toChar() else (256 + n++).toChar()
        }
    }
}
