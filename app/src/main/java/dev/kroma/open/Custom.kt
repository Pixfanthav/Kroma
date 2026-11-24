package dev.kroma.open

import java.io.File

/**
 * Fine-tunes the user brings (extended mode): an Anima or Krea 2 checkpoint in .safetensors, converted on the phone into the
 * engine's files by kconv (engine/kconv, shipped as libkconv.so and run as its own process, so a bad file cannot take the app
 * down). Each one lives in <models>/custom/<name>/engine, beside the bundle's own engine/; the text encoder, the VAE and the
 * rest still come from the model's bundle. Which one draws is kept per model in [Advanced]; none means the bundle's own.
 */
object Custom {
    class Item(val name: String, val kind: String, val dir: File) { val engine get() = File(dir, "engine") }

    fun root(models: File) = File(models, "custom")
    const val UNFINISHED = ".converting"
    /** conversions the app started and did not finish (a folder the user put here is never one of them) */
    fun unfinished(models: File) = root(models).listFiles()?.filter { it.isDirectory && File(it, UNFINISHED).exists() } ?: emptyList()

    /** the converted ones (an unfinished conversion has no kind.txt and is not listed) */
    fun list(models: File, kind: String? = null): List<Item> =
        (root(models).listFiles()?.toList() ?: emptyList()).mapNotNull { d ->
            val k = runCatching { File(d, "kind.txt").readText().trim() }.getOrNull() ?: return@mapNotNull null
            Item(d.name, k, d).takeIf { kind == null || it.kind == kind }
        }.sortedBy { it.name.lowercase() }

    /** a fine-tune's own sampling settings, defaults.txt in its folder (key=value lines: sampler, scheduler, shift, cfg, steps) */
    fun defaults(models: File, name: String): Advanced.Defaults? {
        val f = File(root(models), "$name/defaults.txt"); if (!f.isFile) return null
        val m = runCatching { f.readLines().mapNotNull { l -> l.split('=', limit = 2).takeIf { it.size == 2 }?.let { it[0].trim() to it[1].trim() } }.toMap() }.getOrNull() ?: return null
        val b = if (name.contains("turbo", true)) Advanced.TURBO else Advanced.BASE
        return Advanced.Defaults(m["sampler"]?.takeIf { it in Advanced.SAMPLERS_MOSS } ?: b.sampler, m["scheduler"]?.takeIf { it in Advanced.SCHEDULERS } ?: b.scheduler,
            m["shift"]?.toFloatOrNull() ?: b.shift, m["cfg"]?.toFloatOrNull() ?: b.cfg, m["steps"]?.toIntOrNull() ?: b.steps)
    }
    fun writeDefaults(dir: File, d: Advanced.Defaults) = runCatching {
        File(dir, "defaults.txt").writeText("sampler=${d.sampler}\nscheduler=${d.scheduler}\nshift=${d.shift}\ncfg=${d.cfg}\nsteps=${d.steps}\n") }

    /** the engine folder that draws for [kind] ("anima" / "krea"): a chosen fine-tune in extended mode -- or in any mode when the
     *  built-in model is not on the phone (kept without it) -- else null (the bundle's) */
    fun engineDir(models: File, kind: String): File? {
        if (!Advanced.on && File(models, "${kind}_bundle/engine/weights_b0.f16").exists()) return null
        val n = Advanced.custom(kind).ifEmpty { return null }
        return list(models, kind).firstOrNull { it.name == n }?.engine?.takeIf { it.isDirectory }
    }

    /** the converted size of a checkpoint file of [bytes] in the chosen format (a bf16 / fp16 file; an fp32 or GGUF one differs) */
    fun outBytes(bytes: Long) = when (Advanced.convertFormat) { "q8" -> bytes * 17 / 32; "q4" -> bytes * 9 / 32; else -> bytes }
    /** model files kconv reads */
    fun isModelFile(name: String) = name.endsWith(".safetensors", true) || name.endsWith(".gguf", true)

    /** a folder name from a file name: letters, digits, . _ - only, and not one in use */
    fun freeName(models: File, file: String): String {
        val base = file.substringBeforeLast('.').replace(Regex("[^\\p{L}\\p{N}._-]"), "_").trim('_').take(40).ifEmpty { "model" }
        var n = base; var i = 2
        while (File(root(models), n).exists()) n = "${base}_${i++}"
        return n
    }

    /**
     * Converts [src] (a path the converter process can open: /proc/<pid>/fd/<n> for a picked document) into custom/[name].
     * [progress] gets (blocks done, blocks); the result is the kind ("anima" / "krea"), or an exception with the converter's last words.
     * A failed or stopped conversion leaves nothing behind.
     */
    fun convert(libDir: String, models: File, src: String, name: String, stop: () -> Boolean, progress: (Int, Int) -> Unit): String {
        val fmt = Advanced.convertFormat
        val dir = File(root(models), name); val out = File(dir, "engine").apply { mkdirs() }
        // marks the folder as the app's own work in progress: only such folders are ever cleaned up as unfinished
        File(dir, UNFINISHED).writeText("")
        // SDXL: its text encoders' weights written over the built-in ones' copies (kconv, the bundle's te/clip_map.txt)
        val p = ProcessBuilder(File(libDir, "libkconv.so").absolutePath, src, out.absolutePath, "4", fmt).redirectErrorStream(true)
            .apply { environment()["KX_TE"] = File(models, "sdxl_bundle/te").absolutePath }.start()
        val tail = ArrayDeque<String>(); var nb = 0; var done = 0; var kind = "anima"
        try {
            val stopper = Thread { while (p.isAlive) { if (stop()) { p.destroy(); break }; Thread.sleep(300) } }.apply { isDaemon = true; start() }
            p.inputStream.bufferedReader().forEachLine { l ->
                tail.addLast(l); if (tail.size > 4) tail.removeFirst()
                Regex("(\\d+) blocks").find(l)?.let { nb = it.groupValues[1].toInt(); if (l.startsWith("Krea")) kind = "krea" else if (l.startsWith("MOSS")) kind = "moss" else if (l.startsWith("FLUX.2 Klein")) kind = "klein" else if (l.startsWith("SDXL")) kind = "sdxl"; progress(0, nb) }
                if (l.startsWith("block ") && l.contains("done")) progress(++done, nb)
            }
            val code = p.waitFor(); stopper.join(1000)
            if (stop()) throw InterruptedException()
            if (code != 0 || nb == 0 || done < nb) throw IllegalStateException(tail.joinToString(" / ").ifEmpty { "converter exit $code" })
            // a FLUX.2 Klein conversion: klein.txt ("klein nd ns kD") says the width, 4096 is the 9B
            if (kind == "klein" && runCatching { File(out, "klein.txt").readText().trim().split(Regex("\\s+")).getOrNull(3)?.toIntOrNull() }.getOrNull() == 4096) kind = "klein9"
            File(dir, "kind.txt").writeText(kind); File(dir, UNFINISHED).delete()
            return kind
        } catch (e: Throwable) { p.destroy(); dir.deleteRecursively(); throw e }
    }
}
