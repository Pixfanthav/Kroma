package dev.kroma.open

import org.junit.Assert.assertArrayEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/** ClipTok against ComfyUI's SDXL tokenizer (clip_ids.json: its L and G ids for six prompts, made by the PC's ComfyUI). The vocabulary is read
 *  from ComfyUI's sd1_tokenizer (KROMA_CLIP_TOKENIZER, else comfy/sd1_tokenizer), the test skipped where it is not. */
class ClipTokTest {
    @Test fun sameAsComfyUI() {
        val src = File(System.getenv("KROMA_CLIP_TOKENIZER") ?: "comfy/sd1_tokenizer"); assumeTrue(src.isDirectory)
        val dir = java.nio.file.Files.createTempDirectory("cliptok").toFile(); File(src, "vocab.json").copyTo(File(dir, "clip_vocab.json")); File(src, "merges.txt").copyTo(File(dir, "clip_merges.txt"))
        val tok = ClipTok(dir)
        val cases = org.json.JSONArray(javaClass.classLoader!!.getResource("clip_ids.json")!!.readText())
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i); val p = c.getString("p")
            fun ids(k: String) = c.getJSONArray(k).let { a -> LongArray(a.length()) { a.getLong(it) } }
            assertArrayEquals("L: $p", ids("l"), tok.ids77(p, ClipTok.END))
            assertArrayEquals("G: $p", ids("g"), tok.ids77(p, 0))
        }
        dir.deleteRecursively()
    }
}
