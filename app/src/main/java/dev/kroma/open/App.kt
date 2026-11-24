package dev.kroma.open

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.StatFs
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * The app's language: "" follows the phone (Korean, Chinese, Japanese or Russian on a phone set to one of them, English
 * everywhere else); "en", "ko", "zh", "ja", "ru" are the user's own choice. Kept in the "kroma" preferences and loaded
 * when the process starts.
 */
object Lang {
    @Volatile var code = ""
    private lateinit var app: Context
    private var cached: Pair<String, android.content.res.Resources>? = null
    fun load(ctx: Context) { app = ctx.applicationContext; code = app.getSharedPreferences("kroma", Context.MODE_PRIVATE).getString("lang", "") ?: "" }
    fun set(ctx: Context, c: String) { code = c; ctx.getSharedPreferences("kroma", Context.MODE_PRIVATE).edit().putString("lang", c).apply() }
    /** The string resources of the chosen language; the phone's own when none is chosen. */
    fun res(): android.content.res.Resources {
        val c = code
        if (c.isEmpty()) return app.resources
        cached?.takeIf { it.first == c }?.let { return it.second }
        val conf = android.content.res.Configuration(app.resources.configuration).apply { setLocale(Locale.forLanguageTag(c)) }
        return app.createConfigurationContext(conf).resources.also { cached = c to it }
    }
    /** The app's language as tr() keys it: the chosen one, or the phone's ("zh-TW" for Traditional Chinese: Taiwan, Hong Kong,
     *  Macau or the Hant script; "pt-BR" for any Portuguese). */
    fun now(): String = code.ifEmpty {
        val l = Locale.getDefault()
        when (l.language) {
            "zh" -> if (l.script == "Hant" || l.country in setOf("TW", "HK", "MO")) "zh-TW" else "zh"
            "pt" -> "pt-BR"
            else -> l.language
        }
    }
}

/**
 * A UI text from the string resources (strings.xml under res/values and res/values-ko, -zh, -ja), in the app's
 * language. [args] fill the text's %1$s, %2$s ... Licence and policy wording is not there: it shows in English.
 */
fun tr(id: Int, vararg args: Any?): String = if (args.isEmpty()) Lang.res().getString(id) else Lang.res().getString(id, *args)

/** A text that arrives with its own translations (the model list's pack names): the one for the app's language. */
fun tr(en: String, ko: String, zh: String? = null, ja: String? = null, ru: String? = null,
       tw: String? = null, es: String? = null, pt: String? = null) = when (Lang.now()) {
    "ko" -> ko
    "zh" -> zh ?: en
    "zh-TW" -> tw ?: zh ?: en
    "ja" -> ja ?: en
    "ru" -> ru ?: en
    "es" -> es ?: en
    "pt-BR" -> pt ?: en
    else -> en
}

/**
 * The phone's NPU generation (Hexagon architecture), which is what the engine and the downloads depend on: the DSP code
 * (libkengine_skel_<arch>.so) and the HTP context binaries are per architecture, so every Snapdragon of one generation takes
 * the same set. [socs]: the SoCs known to have it, the first the one the downloads are named after (tag) and the times are for;
 * [tested]: SoCs the app has run on (a real phone). Others of the generation should work but are
 * unconfirmed (their VTCM and memory can be smaller), and are shown so.
 */
enum class Chip(val tag: String, val arch: String, val socs: Map<String, String>, val tested: Set<String>) {
    // 8 Gen 1 (v69) is left out of this release: its engine runs but the numbers came out wrong on one phone, not fixed;
    // libkengine_skel_v69.so is not in the APK and the chip is not detected (the "not supported" screen)
    // 8s Gen 3 (SM8635), 7+ Gen 3 (SM7675) and 8s Gen 4 (SM8735) are v73 by number but carry a smaller NPU:
    // possibly too little VTCM for the engine's 8 MB or no fp16 in the HMX. Not listed
    // until a phone's diagnosis (Logs.report: VTCM, HMX self-test) says what is missing; [CUT] names them with the reason.
    GEN2("8gen2", "v73", linkedMapOf("SM8550" to "Snapdragon 8 Gen 2"), setOf("SM8550")),
    GEN3("8gen3", "v75", linkedMapOf("SM8650" to "Snapdragon 8 Gen 3"), setOf("SM8650")),
    ELITE("8elite", "v79", linkedMapOf("SM8750" to "Snapdragon 8 Elite"), setOf("SM8750")),
    ELITE5("8elite_gen5", "v81", linkedMapOf("SM8850" to "Snapdragon 8 Elite Gen 5"), setOf("SM8850")),
    // the cut-down v73s (8s Gen 3, 8s Gen 4, 7+ Gen 3): 8 Gen 2's downloads and engine, Anima only and untried: the engine asks
    // for 4 MB of VTCM where the chip has less than 8 and sizes the blocks' tiles to it; whatever is laid out for 8 MB says so and stops.
    // No fp16 in the HMX, if that is so on a phone, fails the self-test and the log says it
    SMALL73("8gen2", "v73", linkedMapOf("SM8635" to "Snapdragon 8s Gen 3", "SM8735" to "Snapdragon 8s Gen 4", "SM7675" to "Snapdragon 7+ Gen 3"), emptySet());

    /** a chip on which only Anima is offered, as a trial ([SMALL73]) */
    val animaOnly get() = this == SMALL73

    /** the NPU's name, the same for every SoC of the generation */
    val npu get() = "Hexagon $arch"
    /** this phone's SoC (or the generation's first) */
    val soc get() = socs.keys.firstOrNull { Build.SOC_MODEL.uppercase().startsWith(it) } ?: socs.keys.first()
    /** this phone's Snapdragon by name, with its NPU */
    val title get() = socs[soc]!!
    /** run on this SoC already (else: same NPU generation, unconfirmed) */
    val isTested get() = soc in tested

    companion object {
        /** null = not a supported chip (the app cannot run here). */
        fun detect(): Chip? = Build.SOC_MODEL.uppercase().let { m -> entries.firstOrNull { c -> c.socs.keys.any { m.startsWith(it) } } }
        /** Snapdragons known not to run yet, by SoC: the name, for saying why instead of "unsupported" alone */
        val CUT = linkedMapOf<String, String>()    // were 8s Gen 3 / 8s Gen 4 / 7+ Gen 3, now SMALL73 (Anima only, a trial)
        /** this phone's chip among [CUT], or null */
        fun cut(): String? = Build.SOC_MODEL.uppercase().let { m -> CUT.entries.firstOrNull { m.startsWith(it.key) }?.value }
    }
}

/** A rolling log kept in memory and in a file, so a bug report can carry what happened. */
object Logs {
    private const val MAX = 2_000_000L
    private var file: File? = null
    private val fmt = SimpleDateFormat("MM-dd HH:mm:ss", Locale.US)

    fun init(ctx: Context) {
        file = File(ctx.cacheDir, "kroma.log")
        // an exception nothing caught (a thread of the app's own, a callback) is written down before the app goes
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e -> add("CRASH in ${t.name}: ${e.stackTraceToString().take(4000)}"); prev?.uncaughtException(t, e) }
    }

    @Synchronized fun add(line: String) {
        val f = file ?: return
        runCatching {
            if (f.length() > MAX) f.writeText(f.readText().takeLast((MAX / 2).toInt()))
            f.appendText("${fmt.format(Date())} $line\n")
        }
    }

    /** the app's own process's last log lines from the DSP's side (fastrpc, the engine's LOG): why an open was refused. No prompt text is in them. */
    fun ownLog() = runCatching {
        val p = Runtime.getRuntime().exec(arrayOf("logcat", "-d", "-t", "400", "--pid=${android.os.Process.myPid()}"))
        val keep = Regex("rpc|dsp|DSP|kengine|kblock|kte|qte|remote|session|VTCM|rpcmem|dmabuf", RegexOption.IGNORE_CASE)
        val lines = p.inputStream.bufferedReader().readLines().filter { keep.containsMatchIn(it) }.takeLast(30); p.waitFor()
        add("  logcat (${lines.size} lines):" + lines.joinToString("") { "\n    " + it.take(240) })
    }.onFailure { add("  logcat: ${it.message}") }

    /** The report a user shares: device facts first, then the log. No prompt text is added beyond what the log holds. */
    fun report(ctx: Context, root: File): File {
        val am = ctx.getSystemService(ActivityManager::class.java)
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val fs = StatFs(root.absolutePath)
        val ver = ctx.packageManager.getPackageInfo(ctx.packageName, 0).versionName
        val head = buildString {
            appendLine("Kroma $ver")
            appendLine("device ${Build.MANUFACTURER} ${Build.MODEL} · Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
            appendLine("soc ${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL} · chip set ${Chip.detect()?.tag ?: "unsupported"}")
            appendLine("ram ${mi.totalMem / (1 shl 20)}MB total, ${mi.availMem / (1 shl 20)}MB free")
            appendLine("storage ${fs.availableBytes / (1 shl 30)}GB free")
            appendLine("models " + (File(root, "manifest.json").takeIf { it.exists() }?.let { "manifest ${it.length()}B ${Date(it.lastModified())}" } ?: "none"))
            appendLine(hardware(ctx))
            appendLine("----")
        }
        return File(ctx.cacheDir, "kroma_report.txt").apply { writeText(head + recent() + exits(am, ctx) + logcat()) }
    }

    /** The phone and the app's state beyond the first lines, for a bug report: the CPU, battery and heat, the settings that change the
     *  engine's path, and what the DSP itself reports (Engine.hwInfo: its VTCM, HVX threads, the HMX clock vote and a one-tile HMX fp16
     *  self-test, run on the spot) -- with that, a chip that will not run says why (the cut-down v73s: VTCM, fp16) */
    private fun hardware(ctx: Context): String = buildString {
        runCatching { appendLine("cpu ${Runtime.getRuntime().availableProcessors()} cores · hardware ${Build.HARDWARE} · board ${Build.BOARD} · abi ${Build.SUPPORTED_ABIS.firstOrNull()}") }
        runCatching {
            val b = ctx.registerReceiver(null, android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED))
            val lvl = b?.getIntExtra(android.os.BatteryManager.EXTRA_LEVEL, -1) ?: -1; val temp = (b?.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0
            val plug = b?.getIntExtra(android.os.BatteryManager.EXTRA_PLUGGED, 0) ?: 0
            val th = if (Build.VERSION.SDK_INT >= 29) ctx.getSystemService(android.os.PowerManager::class.java).currentThermalStatus else -1
            appendLine("battery $lvl% ${temp}C ${if (plug != 0) "charging" else "on battery"} · thermal status $th (0 none .. 6 shutdown)")
        }
        runCatching {
            val p = ctx.getSharedPreferences("kroma", Context.MODE_PRIVATE)
            appendLine("packs ${p.getStringSet("packs", emptySet())?.sorted()} · convert ${Advanced.convertFormat} · lora ${Advanced.loraMode} · extended ${Advanced.on}")
        }
        val c = Chip.detect()
        appendLine("npu " + (if (c != null) runCatching { Engine.hwInfo(c) }.getOrElse { "info failed: ${it.message}" } else Chip.cut()?.let { "$it: not run (cut-down v73)" } ?: "none"))
    }

    /** The per-block diagnosis switched on by itself where a run first breaks: the engines' weight folders get an empty kdebug (the
     *  engine reads it at each step's first block), so the steps after the broken one log every block's max |x| and NaN count; at the
     *  run's end those lines are copied in here and the markers the app made are removed (a user's report then shows the block where the
     *  numbers first broke). Markers a developer put there are left alone. */
    private var diagRoot: File? = null
    private val diagMade = mutableListOf<File>()
    @Volatile private var broke = false
    fun armDiag(root: File) { diagRoot = root; broke = false; diagMade.clear() }
    private fun onBroke() {
        if (broke) return; broke = true
        val root = diagRoot ?: return
        val dirs = (root.listFiles()?.filter { it.name.endsWith("_bundle") }?.map { File(it, "engine") }.orEmpty() +
                    File(root, "custom").listFiles()?.flatMap { c -> listOf(File(c, "engine"), c) }.orEmpty())
            .filter { File(it, "weights_b0.f16").isFile }
        for (d in dirs) { val f = File(d, "kdebug"); if (!f.exists() && runCatching { f.createNewFile() }.getOrDefault(false)) diagMade.add(f) }
        add("  diagnosis on from the next step (${dirs.size} engine folder(s))")
    }
    fun diagEnd() {
        if (broke) runCatching {
            val p = Runtime.getRuntime().exec(arrayOf("logcat", "-d", "-t", "4000", "--pid=${android.os.Process.myPid()}"))
            val lines = p.inputStream.bufferedReader().readLines().filter { it.contains(" dbg ") }; p.waitFor()
            // the first step that the diagnosis saw: one line per block is enough to show where it breaks
            val first = lines.take(80)
            add("  block diagnosis (${lines.size} lines, the first ${first.size}):" + first.joinToString("") { "\n    " + it.substringAfter(" dbg ").take(120) })
        }
        diagMade.forEach { runCatching { it.delete() } }; diagMade.clear(); broke = false; diagRoot = null
    }

    /** a step's latent in a few words for its log line: the largest |value|, and NaN / inf when there are any -- the first step that
     *  shows them is where the numbers broke (with the engine's per-block diagnosis, Advanced "diagnosis log", for which block) */
    fun stats(a: FloatArray): String {
        var mx = 0f; var bad = 0
        for (v in a) { if (v.isNaN() || v.isInfinite()) bad++ else { val f = kotlin.math.abs(v); if (f > mx) mx = f } }
        if (bad > 0) onBroke()
        return " · max|x| %.3g".format(mx) + if (bad > 0) " · NaN/inf $bad <- broke here" else ""
    }

    /** where a run (a generation, an import, a download) starts in the log: [recent] cuts at these */
    const val RUN = "==== run "
    /** a run's first line: what it is, after the marker */
    fun run(what: String) = add(RUN + what)

    /** the log's last 5 runs and what came after them, at most its last 1MB: what a report needs, and no more of the user's history
     *  (it was two days, which some would not want to send) */
    private fun recent(): String {
        val all = file?.takeIf { it.exists() }?.readText() ?: return ""
        var at = all.length; var n = 0
        while (n < 5) { val i = all.lastIndexOf(RUN, at - 1); if (i < 0) { at = 0; break }; at = all.lastIndexOf('\n', i).let { if (it < 0) 0 else it + 1 }; n++ }
        return all.substring(at).takeLast(1_000_000)
    }

    /** how the app's last runs ended (Android 11+): a crash, the system killing it for memory, an ANR (with its trace) -- a native crash in
     *  the engine or ONNX Runtime ends the process before anything can be written to the log, and is only seen here */
    private fun exits(am: ActivityManager, ctx: Context): String = if (Build.VERSION.SDK_INT < 30) "" else runCatching {
        buildString {
            appendLine("---- recent exits")
            for (x in am.getHistoricalProcessExitReasons(ctx.packageName, 0, 5)) {
                val why = mapOf(1 to "exit", 2 to "signal", 3 to "low memory", 4 to "crash", 5 to "native crash", 6 to "ANR", 7 to "init failure",
                    8 to "permission change", 9 to "excessive resource", 10 to "user requested", 11 to "user stopped", 12 to "dependency died", 13 to "other",
                    14 to "frozen", 15 to "package state change", 16 to "package updated")[x.reason] ?: "${x.reason}"
                appendLine("${fmt.format(Date(x.timestamp))} $why status ${x.status} importance ${x.importance} pss ${x.pss / 1024}MB · ${x.description ?: ""}")
                if (x.reason == android.app.ApplicationExitInfo.REASON_ANR)
                    x.traceInputStream?.use { appendLine(it.readBytes().toString(Charsets.UTF_8).take(20000)) }
            }
        }
    }.getOrDefault("")

    /** this process's own logcat (the engine, ONNX Runtime, the DSP's FastRPC and the system's lines about the app), the last lines */
    private fun logcat(): String = runCatching {
        val p = ProcessBuilder("logcat", "-d", "-v", "time", "-t", "3000", "--pid=${android.os.Process.myPid()}").redirectErrorStream(true).start()
        "---- logcat\n" + p.inputStream.bufferedReader().readText().also { p.waitFor() }
    }.getOrDefault("")
}

/** A place for a development build to attach test automation; nothing is attached in a release build. */
object Hooks {
    var onStart: ((MainActivity) -> Unit)? = null
}
