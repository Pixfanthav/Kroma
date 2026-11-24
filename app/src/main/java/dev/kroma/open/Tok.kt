package dev.kroma.open

import java.io.File
import java.text.BreakIterator
import java.text.Normalizer
import java.util.regex.Pattern

// Pure-Kotlin tokenizer for the Qwen3-VL text encoder (no native tokenizer library: the available one
// ships 4KB-aligned .so files, which fail the Android 16KB page-size check). Follows HF tokenizers
// step for step; data files: tok/qwen_vocab.txt, tok/qwen_merges.txt.

/** Qwen2 byte-level BPE (NFC -> regex split -> ByteLevel -> merges), no special tokens. */
class QwenTok(dir: File) {
    private val vocab = HashMap<String, Int>(200_000)
    private val ranks = HashMap<String, Int>(200_000)
    private val cache = HashMap<String, IntArray>()

    init {
        File(dir, "qwen_vocab.txt").bufferedReader().useLines { ls -> ls.forEachIndexed { i, t -> vocab[t] = i } }
        File(dir, "qwen_merges.txt").bufferedReader().useLines { ls -> ls.forEachIndexed { i, m -> ranks[m] = i } }
    }

    fun encode(text: String): LongArray {
        val out = ArrayList<Long>()
        val m = SPLIT.matcher(Normalizer.normalize(text, Normalizer.Form.NFC))
        while (m.find()) {
            val piece = m.group()
            val sb = StringBuilder()
            for (b in piece.toByteArray(Charsets.UTF_8)) sb.append(BYTE2CHAR[b.toInt() and 0xFF])
            for (id in cache.getOrPut(sb.toString()) { bpe(sb.toString()) }) out.add(id.toLong())
        }
        return out.toLongArray()
    }

    // GPT2 merge loop: repeatedly merge every occurrence of the lowest-ranked adjacent pair.
    private fun bpe(word: String): IntArray {
        var parts = word.codePoints().toArray().map { String(Character.toChars(it)) }
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
        return IntArray(parts.size) { vocab[parts[it]] ?: throw IllegalStateException("qwen: no id for '${parts[it]}'") }
    }

    companion object {
        // Qwen2 pre-tokenizer regex; \s is spelled as White_Space so ICU and Rust agree on it.
        private val SPLIT: Pattern = Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\p{White_Space}\\p{L}\\p{N}]+[\\r\\n]*" +
            "|\\p{White_Space}*[\\r\\n]+|\\p{White_Space}+(?![^\\p{White_Space}])|\\p{White_Space}+")

        // GPT2 bytes_to_unicode
        private val BYTE2CHAR = CharArray(256).also { t ->
            val direct = (33..126) + (161..172) + (174..255)
            var n = 0
            for (b in 0 until 256) t[b] = if (b in direct) b.toChar() else (256 + n++).toChar()
        }
    }
}

/** T5 SentencePiece unigram as HF runs it: Precompiled(nmt_nfkc) -> Strip right ->
 *  ' {2,}'->'▁' -> Metaspace(first) -> Viterbi per word (fused unk) -> + </s>. */
class T5Tok(dir: File) {
    private val pieces = HashMap<String, Int>(40_000)
    private val scores: DoubleArray
    private var maxLen = 1
    private val unkScore: Double
    private val trie: IntArray
    private val pool: ByteArray

    init {
        val sc = ArrayList<Double>()
        File(dir, "t5_vocab.tsv").bufferedReader().useLines { ls ->
            ls.forEachIndexed { i, l ->
                val tab = l.lastIndexOf('\t'); val p = l.substring(0, tab)
                pieces.putIfAbsent(p, i); sc.add(l.substring(tab + 1).toDouble())
                maxLen = maxOf(maxLen, p.codePointCount(0, p.length))
            }
        }
        scores = sc.toDoubleArray()
        unkScore = scores.min() - 10.0
        val cm = File(dir, "t5_charsmap.bin").readBytes()
        val bb = java.nio.ByteBuffer.wrap(cm).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val trieBytes = bb.getInt(0)
        trie = IntArray(trieBytes / 4) { bb.getInt(4 + it * 4) }
        pool = cm.copyOfRange(4 + trieBytes, cm.size)
    }

    fun encode(text: String): LongArray {
        var s = precompiled(text).trimEnd { it.isWhitespace() }
        s = MULTI_SPACE.matcher(s).replaceAll(META)
        s = s.replace(" ", META)
        if (s.isEmpty()) return longArrayOf(EOS)  // HF prepends nothing to an empty string
        if (!s.startsWith(META)) s = META + s
        val out = ArrayList<Long>()
        // Metaspace split, MergedWithNext: each '▁' starts a new word
        var start = 0
        for (i in 1..s.length) {
            if (i == s.length || s[i] == META[0]) {
                if (i > start) viterbi(s.substring(start, i), out)
                start = i
            }
        }
        out.add(EOS)
        return out.toLongArray()
    }

    private fun viterbi(w: String, out: MutableList<Long>) {
        val cps = w.codePoints().toArray()
        val n = cps.size
        val best = DoubleArray(n + 1) { Double.NEGATIVE_INFINITY }; best[0] = 0.0
        val from = IntArray(n + 1) { -1 }; val id = IntArray(n + 1) { -1 }
        for (i in 0 until n) {
            if (best[i] == Double.NEGATIVE_INFINITY) continue
            var single = false
            val sb = StringBuilder()
            for (j in i until minOf(n, i + maxLen)) {
                sb.appendCodePoint(cps[j])
                val pid = pieces[sb.toString()] ?: continue
                if (j == i) single = true
                val c = best[i] + scores[pid]
                if (from[j + 1] < 0 || c > best[j + 1]) { best[j + 1] = c; from[j + 1] = i; id[j + 1] = pid }
            }
            if (!single) {
                val c = best[i] + unkScore
                if (from[i + 1] < 0 || c > best[i + 1]) { best[i + 1] = c; from[i + 1] = i; id[i + 1] = UNK }
            }
        }
        val ids = ArrayList<Int>()
        var k = n
        while (k > 0) { ids.add(id[k]); k = from[k] }
        ids.reverse()
        for ((i, t) in ids.withIndex()) if (!(t == UNK && i > 0 && ids[i - 1] == UNK)) out.add(t.toLong())  // fuse_unk
    }

    // spm_precompiled: per grapheme (<6 bytes) try the whole cluster, else each char.
    private fun precompiled(text: String): String {
        val sb = StringBuilder()
        val bi = BreakIterator.getCharacterInstance(); bi.setText(text)
        var a = bi.first(); var b = bi.next()
        while (b != BreakIterator.DONE) {
            val g = text.substring(a, b)
            val gb = g.toByteArray(Charsets.UTF_8)
            val whole = if (gb.size < 6) transform(gb) else null
            if (whole != null) sb.append(whole)
            else {
                var i = 0
                while (i < g.length) {
                    val c = g.codePointAt(i); val part = String(Character.toChars(c))
                    sb.append(transform(part.toByteArray(Charsets.UTF_8)) ?: part)
                    i += Character.charCount(c)
                }
            }
            a = b; b = bi.next()
        }
        return sb.toString()
    }

    // darts-clone common-prefix search; the shortest match wins (as spm_precompiled does).
    private fun transform(key: ByteArray): String? {
        var pos = offset(trie[0])
        for (kb in key) {
            val c = kb.toInt() and 0xFF
            pos = pos xor c
            val u = trie[pos]
            if ((u and (0x80000000.toInt() or 0xFF)) != c) return null
            pos = pos xor offset(u)
            if ((u ushr 8) and 1 == 1) {
                val v = trie[pos] and 0x7FFFFFFF
                var e = v; while (e < pool.size && pool[e] != 0.toByte()) e++
                return String(pool, v, e - v, Charsets.UTF_8)
            }
        }
        return null
    }
    private fun offset(u: Int) = (u ushr 10) shl ((u and (1 shl 9)) ushr 6)

    companion object {
        private const val META = "▁"
        private const val UNK = 2
        private const val EOS = 1L
        private val MULTI_SPACE: Pattern = Pattern.compile(" {2,}")
    }
}
