package dev.kroma.open

import java.io.File

/**
 * YuE2 (style tags and lyrics -> a 48 kHz song), experimental: the yue-synth program shipped as libyuesynth.so (with the ggml
 * libraries and the kblock stub) and run as its own process, the way kconv is. The score and the semantic tokens are made on
 * the CPU, the flow-matching NAR's 28 layers on the DSP (through this chip's libkengine_skel), the
 * VAE on the CPU. Files: <models>/yue2/YuE2-3B-Q8_0.gguf, YuE2-Vae-F32.gguf, eng/ and eng_ar/ (made by the maintainers' converter); without eng/ the NAR runs on the CPU too (about an hour for 30 s). The weights are CC BY-NC 4.0.
 */
object Yue {
    fun dir(root: File) = File(root, "yue2")
    fun present(root: File) = File(dir(root), "YuE2-3B-Q8_0.gguf").isFile && File(dir(root), "YuE2-Vae-F32.gguf").isFile

    /** The prompt box as YuE2 reads it: the lines before the first "[section]" are the style, the rest the lyrics. */
    fun split(prompt: String): Pair<String, String> {
        val lines = prompt.lines(); val at = lines.indexOfFirst { it.trim().startsWith("[") }
        return if (at < 0) prompt.trim() to "" else lines.take(at).joinToString(" ").trim() to lines.drop(at).joinToString("\n").trim()
    }

    /**
     * One song into [out] (mp3), cut at [seconds] when that is above 0 (else as long as the score the model writes). [onStage] gets (stage, done, total) as yue-synth prints it: "score", "semantic",
     * "nar", "vae". Throws with the last lines of its log when it fails; returns false when stopped.
     */
    /** [seconds]: the length aimed at (yue-synth's --duration, a target it ends near); [floor]: no end before this many seconds
     *  (a length the user chose; 0 leaves it to the model) */
    fun make(libDir: String, root: File, prompt: String, seconds: Int, seed: Long, out: File, log: (String) -> Unit,
             onStage: (String, Int, Int) -> Unit, stop: () -> Boolean, floor: Int = 0): Boolean {
        val d = dir(root); val (style, lyrics) = split(prompt)
        val req = File(d, "request.json")
        req.writeText(org.json.JSONObject().put("style", style).put("lyrics", lyrics).put("cot", "full").toString())
        out.delete()
        val cmd = listOf(File(libDir, "libyuesynth.so").absolutePath, "--model", File(d, "YuE2-3B-Q8_0.gguf").absolutePath,
            "--vae", File(d, "YuE2-Vae-F32.gguf").absolutePath, "--request", req.absolutePath, "--out", out.absolutePath,
            "--seed", "$seed", "--lm-seed", "$seed") + (if (seconds > 0) listOf("--duration", "$seconds") else emptyList())
        val pb = ProcessBuilder(cmd).directory(d).redirectErrorStream(true)
        // /vendor/lib64 for libcdsprpc.so: a process the app starts does not get the vendor libraries the app's own one does
        val env = pb.environment(); env["LD_LIBRARY_PATH"] = "$libDir:/vendor/lib64"
        if (floor > 0) env["KROMA_YUE_MIN"] = "$floor"
        // flash attention on the CPU too (the program turns it on for a GPU only): the tokens' cost grows 27 ms per 1000 keys, not 68 (8 Gen 2)
        env["YUE_CPU_FA"] = "1"
        val eng = File(d, "eng")
        val arch = Chip.detect()?.arch
        if (File(eng, "io.f32").isFile && arch != null) {
            env["YUE_ENGINE"] = eng.absolutePath; env["KROMA_SKEL"] = "libkengine_skel_$arch.so"
            // the prompts' prefill on the DSP too the AR layers' files
            File(d, "eng_ar").takeIf { File(it, "weights_l0.bin").isFile }?.let { env["YUE_ENGINE_AR"] = it.absolutePath }
            // the VAE decoder on the DSP the VAE's files
            File(d, "eng_vae").takeIf { File(it, "adec.w").isFile }?.let { env["YUE_ENGINE_VAE"] = it.absolutePath }
            env["ADSP_LIBRARY_PATH"] = "$libDir;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp"
        } else log("  yue: no engine files (yue2/eng) or no skel for this chip: the NAR runs on the CPU")
        val p = pb.start(); val tail = ArrayDeque<String>()
        // the budgets yue-synth prints before each part, for the parts' totals
        var semTotal = 0; var narTotal = 0; var vaeTotal = 0
        val rx = Regex("""\[AR\] (Score|Semantic) (\d+)/(\d+)""")
        val watcher = Thread { while (true) { if (stop()) { p.destroy(); break }; try { Thread.sleep(300) } catch (_: InterruptedException) { break } } }.apply { isDaemon = true; start() }
        try {
            // a stop destroys the process (the watcher above), which ends this read with an IOException: that is a stop, not a failure
            try { p.inputStream.bufferedReader().forEachLine { l ->
                tail.addLast(l); if (tail.size > 30) tail.removeFirst()
                if (l.startsWith("[Kroma]") || l.contains("Solved") || l.contains("Done") || l.contains("prefill") || l.startsWith("[Pipeline]")) log("  $l")
                rx.find(l)?.let { m -> val (st, a, b) = m.destructured
                    if (st == "Semantic") semTotal = b.toInt()
                    // the budget printed is the safety cap, 130% of the length aimed at (the model's soft end): the screen counts to the aim
                    onStage(st.lowercase(), a.toInt(), if (st == "Semantic" && seconds > 0) minOf(b.toInt(), seconds * 25) else b.toInt()) }
                Regex("""\[NAR\] Step (\d+)/(\d+)""").find(l)?.let { m -> narTotal = m.groupValues[2].toInt(); onStage("nar", m.groupValues[1].toInt(), narTotal) }
                Regex("""Tiled decode: (\d+) tiles""").find(l)?.let { m -> vaeTotal = m.groupValues[1].toInt(); onStage("vae", 0, vaeTotal) }
                if (l.startsWith("[VAE] Graph:") && vaeTotal > 0) onStage("vae", (tail.count { it.startsWith("[VAE] Graph:") }).coerceAtMost(vaeTotal), vaeTotal)
            } } catch (e: java.io.IOException) {
                // the same exception when yue-synth itself ended: the system killing it for memory (exit 137 = SIGKILL), a crash (134, 139)
                if (stop()) return false
                val rc = runCatching { p.waitFor() }.getOrDefault(-1)
                error("yue-synth ended (exit $rc${if (rc == 137) ", killed: likely low memory" else if (rc == 134 || rc == 139) ", crashed" else ""}): " + tail.takeLast(6).joinToString(" | "))
            }
            val rc = p.waitFor()
            if (stop()) return false
            if (rc != 0 || !out.isFile) error("yue-synth exit $rc: " + tail.takeLast(6).joinToString(" | "))
            return true
        } finally { watcher.interrupt(); p.destroy() }
    }
}
