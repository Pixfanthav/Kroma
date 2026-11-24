package dev.kroma.open

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Prompt translation into English for the image models: PLaMo 3 Translate 0.8B (Preferred Networks, Built with PLaMo) through
 * llama.cpp's llama-completion, built with the NDK as one static program (CPU) and shipped as libllamacomp.so.
 *
 *  input : a prompt with Korean, Japanese, Chinese or Russian in it (anything else is left as written)
 *  prompt: PLaMo's translation format with a context line (an image prompt: tags stay short tags, descriptions natural English) and
 *          a glossary of the Danbooru tag list's other names found in the prompt (plamo/gloss.json: name -> [tag, posts, shared])
 *  output: greedy, up to 256 tokens, cut at PLaMo's next tag
 *
 * Files: <models>/plamo/plamo-3-translate-0.8b-Q4_K_M.gguf, <models>/plamo/gloss.json. About 4 s a prompt on a Snapdragon 8 Gen 2 (CPU).
 */
object Translator {
    private const val GGUF = "plamo-3-translate-0.8b-Q4_K_M.gguf"
    private const val CTX = "A prompt for an AI image generator. Comma-separated short items are Danbooru-style tags: translate them as short English tags. " +
        "Longer phrases are natural-language scene descriptions: translate them as natural English. Keep the commas and the order."

    fun present(root: File) = File(root, "plamo/$GGUF").isFile

    /** the prompt's language for PLaMo (ko, ja, zh, ru) by its letters, or null when it has none of those */
    fun lang(s: String): String? {
        var ko = 0; var kana = 0; var han = 0; var ru = 0
        for (c in s) when (c) {
            in '가'..'힯', in 'ᄀ'..'ᇿ', in '㄰'..'㆏' -> ko++
            in '぀'..'ヿ' -> kana++
            in '㐀'..'鿿' -> han++
            in 'Ѐ'..'ӿ' -> ru++
        }
        return when {
            ko > 0 && ko >= kana -> "ko"
            kana > 0 -> "ja"
            han > 0 -> "zh"
            ru > 0 -> "ru"
            else -> null
        }
    }

    /** gloss.json's names used for the glossary (one tag only, at least 1000 posts), longest first */
    @Volatile private var gloss: Pair<List<String>, Map<String, String>>? = null
    private fun gloss(root: File): Pair<List<String>, Map<String, String>> = gloss ?: synchronized(this) {
        gloss ?: run {
            val m = HashMap<String, String>()
            runCatching {
                val o = JSONObject(File(root, "plamo/gloss.json").readText())
                for (k in o.keys()) { val v = o.getJSONArray(k); if (v.optInt(2) == 0 && v.optInt(1) >= 1000) m[k] = v.getString(0) }
            }
            (m.keys.sortedByDescending { it.length } to m as Map<String, String>).also { gloss = it }
        }
    }

    /** the longest glossary names found in src, not overlapping, at most cap; a name of two letters only as a whole comma item */
    private fun glossary(root: File, src: String, cap: Int = 24): List<Pair<String, String>> {
        val (names, map) = gloss(root); val used = BooleanArray(src.length); val items = src.split(",").map { it.trim() }.toSet(); val out = ArrayList<Pair<String, String>>()
        for (n in names) {
            val i = src.indexOf(n); if (i < 0 || (i until i + n.length).any { used[it] }) continue
            if (n.length < 3 && n !in items) continue
            for (k in i until i + n.length) used[k] = true
            out.add(n to map.getValue(n)); if (out.size >= cap) break
        }
        return out
    }

    private fun format(root: File, src: String, lang: String): String {
        val sb = StringBuilder("<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>context<|plamo:msg|>").append(CTX)
        val g = glossary(root, src)
        if (g.isNotEmpty()) { sb.append("<|plamo:tag|>glossary"); for ((s, t) in g) sb.append("<|plamo:index|>").append(s).append("<|plamo:item|>").append(t) }
        return sb.append("<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>").append(lang).append("<|plamo:msg|>").append(src)
            .append("<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>").toString()
    }

    /** src in English, or null (no language to translate from, no model, or the program failed: the caller keeps src) */
    fun translate(ctx: Context, root: File, src: String, log: (String) -> Unit): String? {
        val lang = lang(src) ?: return null
        if (!present(root)) { log("translation: no PLaMo model (plamo/$GGUF)"); return null }
        val exe = File(ctx.applicationInfo.nativeLibraryDir, "libllamacomp.so"); if (!exe.isFile) { log("translation: libllamacomp.so missing"); return null }
        val t0 = System.currentTimeMillis()
        val inp = File(ctx.cacheDir, "plamo_in.txt").apply { writeText(format(root, src, lang)) }
        val cmd = listOf(exe.absolutePath, "-m", File(root, "plamo/$GGUF").absolutePath, "-f", inp.absolutePath, "-n", "256", "--temp", "0",
            "-t", "4", "-c", "2048", "-no-cnv", "--no-display-prompt", "-r", "<|plamo:tag|>")
        return runCatching {
            val p = ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.to(File("/dev/null"))).start()
            val out = p.inputStream.bufferedReader().readText()
            if (!p.waitFor(90, TimeUnit.SECONDS)) { p.destroyForcibly(); null }
            else out.substringBefore("[end of text]").substringBefore("<|plamo").trim().takeIf { it.isNotEmpty() }
        }.getOrNull().also { log("translation $lang -> en (PLaMo) ${System.currentTimeMillis() - t0} ms: ${it ?: "failed"}") }
    }
}
