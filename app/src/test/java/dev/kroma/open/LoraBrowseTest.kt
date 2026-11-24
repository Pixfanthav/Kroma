package dev.kroma.open

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** The Civitai listing as the app reads it, against a saved answer cut down to the fields that are used. */
class LoraBrowseTest {
    private val json = javaClass.getResource("/civitai_anima.json")!!.readText()

    @Test fun keepsOnlyWhatCanBeUsed() {
        val (hits, next) = LoraBrowse.parse(json, "anima")
        assertEquals("20", next)
        assertTrue(hits.isNotEmpty())
        // dropped: the entry marked adult, the one made for another model, the one too large to be a LoRA
        assertTrue(hits.none { it.name in setOf("adult entry", "other model", "whole model") })
        hits.forEach {
            assertTrue(it.file, it.file.endsWith(".safetensors", true))
            assertTrue(it.url, it.url.startsWith("https://civitai.com/api/download/"))
            assertTrue(it.sizeMb <= LoraBrowse.MAX_MB); assertTrue(it.page.startsWith("https://civitai.com/models/"))
        }
        assertEquals(emptyList<LoraBrowse.Hit>(), LoraBrowse.parse(json, "krea").first)
    }

    @Test fun listAddress() {
        assertEquals("https://civitai.com/api/v1/models?types=LORA&baseModels=Krea%202&sort=Most%20Downloaded&limit=20&nsfw=false",
                     LoraBrowse.listUrl("krea", 0, " ", null))
        assertEquals("https://civitai.com/api/v1/models?types=LORA&baseModels=Anima&sort=Newest&limit=20&nsfw=false&query=line%20art&cursor=40",
                     LoraBrowse.listUrl("anima", 2, " line art ", "40"))
    }
}
