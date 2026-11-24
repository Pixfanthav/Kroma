package dev.kroma.open

import kotlin.math.roundToInt
import android.app.Activity
import android.app.AlertDialog
import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import androidx.core.content.FileProvider
import java.io.File
import kotlin.concurrent.thread
import dev.kroma.open.MainActivity.Job
import dev.kroma.open.MainActivity.Companion.PICK_GRAPH_IMAGE
import dev.kroma.open.MainActivity.Companion.PICK_IMG_START
import dev.kroma.open.MainActivity.Companion.PICK_LORA
import dev.kroma.open.MainActivity.Companion.PICK_MODEL
import dev.kroma.open.MainActivity.Companion.REPO_URL
import dev.kroma.open.MainActivity.Companion.modelName

/* MainActivity: first run: language, the terms, the model server and the model download (moved out of MainActivity.kt unchanged) */

// ---------------------------------------------------------------- first run
/** The very first screen: which language. Each choice is written in its own language; it can be changed later in Settings. */
internal fun MainActivity.language() {
    fun pick(code: String) { Lang.set(this, code); Logs.add("language ${code.ifEmpty { "system" }}"); route() }
    show(column(
        text("Kroma", 26f, true),
        note("Language · 언어 · 语言 · 語言 · 言語 · Язык · Idioma"),
        button("English") { pick("en") },
        button("한국어") { pick("ko") },
        button("简体中文") { pick("zh") },
        button("繁體中文") { pick("zh-TW") },
        button("日本語") { pick("ja") },
        button("Русский") { pick("ru") },
        button("Español") { pick("es") },
        button("Português") { pick("pt-BR") }))
}

internal fun MainActivity.gate() {
    val agree = check(tr(R.string.i_have_read_and_agree))
    val go = primary(tr(R.string.continue_button)) {
        prefs.edit().putBoolean("agreed_terms_2", true).apply(); route()
    }.apply { isEnabled = false }
    agree.setOnCheckedChangeListener { _, c -> go.isEnabled = c }
    show(column(
        text("Kroma", 26f, true),
        note(tr(R.string.krea_2_turbo_unquantized_fp16)),
        card(text(tr(R.string.before_you_start), 17f, true),
        // the terms are shown in English in every language (nobody here can vouch for a legal translation), and say so
        note(tr(R.string.terms_english_only)).apply { if (text.isEmpty()) visibility = View.GONE },
        text("• The models are converted to run on this phone. Each keeps its own license (the full texts are under the button below), " +
                "none is an official product of its maker or endorsed by it, and the app and the models come with no warranty: neither their makers nor Kroma's authors are " +
                "liable for any damage from using them, to the fullest extent the law allows:\n" +
                "   – Krea 2 (Krea.ai): Krea 2 Community License Agreement and Krea's Acceptable Use Policy.\n" +
                "   – Non-commercial use only: Anima (CircleStone Labs, CircleStone Non-Commercial License; built on NVIDIA Cosmos, NVIDIA Open Model License), FLUX.2 [klein] 9B (Black Forest Labs, " +
                "FLUX Non-Commercial License) and YuE2 (M-A-P, CC BY-NC 4.0).\n" +
                "   – SDXL 1.0 (Stability AI): CreativeML Open RAIL++-M. Its use restrictions (paragraph 5 and Attachment A) are part of these terms: " +
                "you agree not to use SDXL, or anything made from it, for any of those uses.\n" +
                "   – Apache License 2.0: MOSS-Image with the HaDeS turbo, FLUX.2 [klein] 4B.\n" +
                "   – Translation (optional): PLaMo 3 Translate (Preferred Networks), PLaMo Community License; you must be 18 or older, or have a " +
                "guardian's consent. Tag matching: EmbeddingGemma 2 (Google), under the Gemma Prohibited Use Policy.\n" +
                "• Not allowed: sexual content involving minors, sexual or deceptive images of real people without their consent, harassment, and anything unlawful.\n" +
                "• Prompts that sexualise minors are always refused. A picture the classifier flags is blurred and not saved, unless an adult turns that off in Settings (the option is not in every build).\n" +
                "• You are responsible for what you generate. Where the law requires it, say that a picture is AI-generated.\n" +
                "• This app is an independent project, not made or endorsed by any of the models' makers.")),
        button(tr(R.string.read_the_full_license_and)) { notices() },
        agree, go))
}

/** The app's own license and the model's are different things: each gets its own heading. */
internal fun MainActivity.notices() {
    fun head(t: String) = text(t, 16f, true)
    fun body(t: String) = note(t, 12f).apply { setTextIsSelectable(true) }
    val v = column(
        head(tr(R.string.this_app)), body(tr(R.string.kroma_is_free_software_under)),
        text(REPO_URL, 13f).apply { setTextColor(ACCENT); paintFlags = paintFlags or android.graphics.Paint.UNDERLINE_TEXT_FLAG
            setOnClickListener { runCatching { startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(REPO_URL))) } } },
        head(tr(R.string.models_and_components)), body(asset("NOTICE.txt")),
        head(tr(R.string.krea_2_model_license)),
        body(tr(R.string.krea_ai_s_license_for)),
        body(asset("KREA_LICENSE.txt")),
        head(tr(R.string.anima_model_license)),
        body(tr(R.string.circlestone_labs_license_for_the)),
        body(asset("ANIMA_LICENSE.txt")),
        head("NVIDIA Open Model License (Anima is built on NVIDIA Cosmos)"),
        body("Licensed by NVIDIA Corporation under the NVIDIA Open Model License."),
        body(asset("NVIDIA_LICENSE.txt")),
        head(tr(R.string.sdxl_1_0_license_creativeml_open)),
        body(asset("SDXL_LICENSE.txt")),
        head(tr(R.string.flux_2_klein_9b_license_flux)),
        body(asset("FLUX_NC_LICENSE.txt")),
        head(tr(R.string.plamo_community_license_translation)),
        body(asset("PLAMO_LICENSE.txt")),
        head("Third-party software (MIT, BSD-3, Apache-2.0, CC BY-NC)"), body(asset("THIRD_PARTY.txt")),
        // ONNX Runtime's notice file is 330 KB: read only when asked
        primary("ONNX Runtime third-party notices") { Thread { val t = asset("ONNXRUNTIME_NOTICES.txt")
            runOnUiThread { AlertDialog.Builder(this).setView(ScrollView(this).apply { setBackgroundColor(BG); addView(body(t)) }).setPositiveButton("OK", null).show() } }.start() })
    AlertDialog.Builder(this).setView(ScrollView(this).apply { setBackgroundColor(BG); addView(v) }).setPositiveButton("OK", null).show()
}

internal fun MainActivity.unsupported() = show(column(*listOfNotNull(
    text("Kroma", 26f, true),
    text(tr(R.string.this_phone_s_chip_is, Build.SOC_MODEL)),
    // the cut-down v73 chips: why, and that the log is what can change it (it carries the NPU's VTCM and its HMX self-test)
    Chip.cut()?.let { n -> note(tr(R.string.s_has_a_hexagon_npu_but).format(n)) },
    button(tr(R.string.send_the_log)) { bugReport() },
    // debug builds: the rest of the app's screens without the models, to check the layout and the wording on any phone
    if (BuildConfig.DEBUG) button(tr(R.string.ui_preview_debug_build_nothing_is)) {
        prefs.edit().putBoolean("ui_preview", true).putBoolean("setup_later", true).apply(); recreate() } else null).toTypedArray()))

// ---------------------------------------------------------------- model download
internal fun MainActivity.setup() {
    val c = chip!!
    val info = note("")
    val wifi = check(tr(R.string.download_on_wi_fi_only), prefs.getBoolean("wifi_only", true)).apply {
        setOnCheckedChangeListener { _, v -> prefs.edit().putBoolean("wifi_only", v).apply() } }
    val mirror = check(tr(R.string.download_through_the_hf_mirror), models.hfMirror).apply { setOnCheckedChangeListener { _, v -> models.hfMirror = v } }
    val pb = progress().apply { max = 1000; visibility = View.GONE }
    val sumTv = note("")
    val picks = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    val boxes = LinkedHashMap<String, CheckBox>()
    lateinit var go: Button
    var removeConfirmed = false
    fun say(s: String) = runOnUiThread { info.text = s }
    fun wanted() = boxes.filter { it.value.isChecked }.keys
    fun modelPicked() = boxes.values.any { it.isChecked }         // an option alone (upscale, translation) can be downloaded too
    // packs on the phone that are no longer ticked: confirming removes them
    fun unticked() = models.offered().filter { boxes[it.id]?.isChecked == false && models.filesOf(it.id, c).let { f -> f.isNotEmpty() && models.missing(f).size < f.size } }
    // what the ticked packs still need, next to what the phone has left; the button says what it will do
    fun totals() {
        val need = if (boxes.isEmpty()) 0L else models.missing(models.needed(c, wanted().toSet())).sumOf { it.size }
        sumTv.text = if (boxes.isEmpty()) tr(R.string.free_storage, gb(models.freeBytes()))
                     else tr(R.string.to_download_free_storage, gb(need), gb(models.freeBytes()))
        go.text = when {
            boxes.isEmpty() -> tr(R.string.get_the_model_list)
            need > 0 -> tr(R.string.download, gb(need))
            unticked().isNotEmpty() -> tr(R.string.apply)
            else -> tr(R.string.done)
        }
    }
    // one row per pack the server offers: tick · name · size · fold arrow; unfolded, the parts it consists of
    fun breakdown(groups: List<Models.Group>, files: List<Models.Item>) = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL; visibility = View.GONE; setPadding(px(34), 0, 0, px(6))
        for (g in groups) { val n = files.filter { it.group == g.id }.sumOf { it.size }; if (n > 0) addView(line(note(g.name), note(gb(n)))) }
    }
    fun foldable(head: View, size: String, detail: View, extra: View? = null): View {
        val arrow = note("▾", 16f).apply { setPadding(px(8), px(8), px(4), px(8)) }
        val top = if (extra == null) line(head, note(size, 14f), arrow) else line(head, note(size, 14f), extra, arrow)
        val toggle = View.OnClickListener { val open = detail.visibility != View.VISIBLE; detail.visibility = if (open) View.VISIBLE else View.GONE; arrow.text = if (open) "▴" else "▾" }
        arrow.setOnClickListener(toggle); if (head !is CheckBox) top.setOnClickListener(toggle)
        return LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; addView(top); addView(detail) }
    }
    fun state(have: Boolean) = if (have) tr(R.string.downloaded) else tr(R.string.not_downloaded)
    fun section(title: String) = note(title, 12f).apply { setPadding(0, px(6), 0, 0) }
    lateinit var relist: () -> Unit
    // "Delete" on a pack that has files on the phone: asks first, keeps whatever another kept pack or the shared set uses
    fun deleteLink(p: Models.Pack) = text(tr(R.string.delete), 13f).apply {
        setTextColor(ACCENT); setPadding(px(10), px(8), px(2), px(8))
        setOnClickListener {
            if (running) return@setOnClickListener
            val keep = models.needed(c, models.chosen - p.id).map { it.path }.toSet()
            val files = models.filesOf(p.id, c).filter { it.path !in keep }
            AlertDialog.Builder(this@setup).setTitle(tr(R.string.delete_from_this_phone, p.name))
                .setMessage(tr(R.string.frees_it_can_be_downloaded, gb(files.sumOf { f -> File(models.root, f.path).length() })))
                .setPositiveButton(tr(R.string.delete)) { _, _ ->
                    files.forEach { f -> File(models.root, f.path).delete(); File(models.root, f.path + ".part").delete() }
                    models.chosen = models.chosen - p.id; boxes[p.id]?.isChecked = false; Logs.add("deleted pack: ${p.id}"); relist() }
                .setNegativeButton(tr(R.string.cancel), null).show()
        }
    }
    /* in a model's breakdown: its built-in image model can be left out (someone who only uses fine-tunes); leaving it out deletes
       it from the phone if it is here, after asking */
    fun baseToggle(p: Models.Pack) = check(tr(R.string.include_the_built_in_model_untick),
                                           models.hasBase(p.id)).apply {
        setOnCheckedChangeListener { b, on ->
            if (running) { b.isChecked = !on; return@setOnCheckedChangeListener }
            if (on) { models.noBase = models.noBase - p.id; relist(); return@setOnCheckedChangeListener }
            val here = models.allFilesOf(p.id, c).filter { it.group == "dit" && File(models.root, it.path).exists() }
            if (here.isEmpty()) { models.noBase = models.noBase + p.id; relist(); return@setOnCheckedChangeListener }
            AlertDialog.Builder(this@setup).setTitle(tr(R.string.delete_from_this_phone, p.name))
                .setMessage(tr(R.string.frees_it_can_be_downloaded, gb(here.sumOf { f -> File(models.root, f.path).length() })))
                .setPositiveButton(tr(R.string.delete)) { _, _ ->
                    here.forEach { f -> File(models.root, f.path).delete(); File(models.root, f.path + ".part").delete() }
                    models.noBase = models.noBase + p.id; Logs.add("built-in model removed: ${p.id}"); relist() }
                .setNegativeButton(tr(R.string.cancel)) { _, _ -> b.isChecked = true }
                .setOnCancelListener { b.isChecked = true }.show()
        }
    }
    /* in a model's breakdown: the weight format of its built-in model (FP16 as made, the default; Q8 / Q4 smaller). Changing it
       downloads the other block files; the old ones go when the download starts (Models.applyFormats) */
    fun fmtName(f: String) = when (f) {
        "q8" -> tr(R.string.q8_about_half_the_size_close)
        "q4" -> tr(R.string.q4_about_a_quarter_lower_quality)
        else -> tr(R.string.fp16_as_made_default) }
    fun formatRow(p: Models.Pack): View = text(tr(R.string.weights) + fmtName(models.fmt(p.id)) + "  ▾", 13f).apply {
        setTextColor(ACCENT); setPadding(px(6), px(8), px(2), px(8))
        setOnClickListener {
            if (running) return@setOnClickListener
            val fs = models.formats(p.id)
            AlertDialog.Builder(this@setup).setSingleChoiceItems(fs.map { fmtName(it) }.toTypedArray(), fs.indexOf(models.fmt(p.id))) { d, i ->
                models.setFmt(p.id, fs[i]); Logs.add("weight format chosen ${p.id}: ${fs[i]}"); d.dismiss(); relist() }.show()
        }
    }
    /* the text encoder's own format: FP16 as made (the default) or Q8, apart from the image model's (Models.teFmt) */
    fun teFormatRow(p: Models.Pack): View = text(tr(R.string.encoder) + fmtName(models.teFmt(p.id)) + "  ▾", 13f).apply {
        setTextColor(ACCENT); setPadding(px(6), px(8), px(2), px(8))
        setOnClickListener {
            if (running) return@setOnClickListener
            val fs = models.teFormats(p.id)
            AlertDialog.Builder(this@setup).setSingleChoiceItems(fs.map { fmtName(it) }.toTypedArray(), fs.indexOf(models.teFmt(p.id))) { d, i ->
                models.setTeFmt(p.id, fs[i]); Logs.add("encoder format chosen ${p.id}: ${fs[i]}"); d.dismiss(); relist() }.show()
        }
    }
    fun listModels() {
        // the ticks survive a redraw (a format or the built-in model changed in a breakdown): they used to go back to the defaults
        val ticked = boxes.mapValues { it.value.isChecked }
        picks.removeAllViews(); boxes.clear()
        val offered = models.offered()
        for (isModel in listOf(true, false)) {
            val ps = offered.filter { it.isModel == isModel }
            if (ps.isEmpty()) continue
            picks.addView(section(if (isModel) tr(R.string.models_pick_at_least_one) else tr(R.string.options)))
            for (p in ps) {
                val files = models.filesOf(p.id, c)
                val need = models.missing(files).sumOf { it.size }
                val whole = files.isNotEmpty() && need == 0L
                val cb = check(p.name, ticked[p.id] ?: (p.id in models.chosen || whole || (models.chosen.isEmpty() && (boxes.isEmpty() || !isModel)))).apply {
                    setOnCheckedChangeListener { _, _ -> go.isEnabled = modelPicked(); totals() } }
                boxes[p.id] = cb
                val onPhone = files.isNotEmpty() && models.missing(files).size < files.size      // all of it or part of it
                val detail = breakdown(p.groups, files)
                if (isModel && models.allFilesOf(p.id, c).any { it.group == "dit" }) detail.addView(baseToggle(p))
                if (isModel && models.formats(p.id).size > 1) detail.addView(formatRow(p))
                if (isModel && models.teFormats(p.id).size > 1) detail.addView(teFormatRow(p))
                picks.addView(foldable(cb, state(need == 0L) + gb(files.sumOf { it.size }), detail, if (onPhone) deleteLink(p) else null))
            }
        }
        if (offered.isNotEmpty()) {
            val bf = models.baseFiles(c)
            picks.addView(section(tr(R.string.always_included)))
            picks.addView(foldable(text(tr(R.string.shared_files)).apply { setPadding(px(6), px(8), 0, px(8)) },
                state(models.missing(bf).isEmpty()) + gb(bf.sumOf { it.size }), breakdown(models.baseGroups(), bf)))
        } else picks.addView(note(if (models.server.isEmpty()) tr(R.string.models_not_published, c.npu)
                                  else tr(R.string.press_the_button_below_to_read)))
        go.isEnabled = boxes.isEmpty() || modelPicked()
        totals()
    }
    relist = { listModels() }
    go = primary("") {
        if (running) { stopRequested = true; return@primary }
        if (models.server.isEmpty()) { say(tr(R.string.models_not_published, c.npu)); return@primary }
        if (boxes.isEmpty()) {          // first press: only the list
            say(tr(R.string.reading_the_model_list))
            thread {
                runCatching { models.fetchManifest() }.onSuccess { runOnUiThread { listModels(); info.text = "" } }
                    .onFailure { e -> Logs.add("manifest failed: ${e.message}"); say(tr(R.string.failed) + (e.message ?: e.javaClass.simpleName)) }
            }
            return@primary
        }
        if (wifi.isChecked && !models.onWifi()) { say(tr(R.string.not_on_wi_fi)); return@primary }
        val ids = wanted().toSet()
        val gone = unticked()
        if (gone.isNotEmpty() && !removeConfirmed) {
            val keep = models.needed(c, ids).map { it.path }.toSet()
            val files = gone.flatMap { models.filesOf(it.id, c) }.filter { it.path !in keep }
            AlertDialog.Builder(this).setTitle(tr(R.string.remove_from_this_phone))
                .setMessage(gone.joinToString("\n") { "• " + it.name } + "\n\n" + tr(R.string.frees_they_can_be_downloaded, gb(files.sumOf { f -> File(models.root, f.path).length() })))
                .setPositiveButton(tr(R.string.remove)) { _, _ ->
                    files.forEach { f -> File(models.root, f.path).delete() }; Logs.add("removed packs: ${gone.joinToString { it.id }}")
                    removeConfirmed = true; go.performClick(); removeConfirmed = false }
                .setNegativeButton(tr(R.string.cancel), null).show()
            return@primary
        }
        pb.visibility = View.VISIBLE
        running = true; stopRequested = false; go.text = tr(R.string.pause); boxes.values.forEach { it.isEnabled = false }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); GenService.start(this)
        thread {
            try {
                runCatching { models.fetchManifest() }       // a newer list if the server has one; the saved one otherwise
                models.applyFormats(ids)
                val items = models.needed(c, ids)
                val need = models.missing(items).sumOf { it.size }
                val free = models.freeBytes()
                if (need > free - (2L shl 30)) error(tr(R.string.not_enough_storage_to_download, gb(need), gb(free)))
                Logs.add("download start: ${ids.joinToString()} · ${models.missing(items).size} files, ${gb(need)}")
                models.offered().filter { it.isModel && it.id in ids && models.missing(models.filesOf(it.id, c)).isNotEmpty() }.forEach { models.countInstall(it.id, c) }
                val t0 = System.currentTimeMillis(); var first = -1L
                models.download(items, { stopRequested || (wifi.isChecked && !models.onWifi()) }) { done, total, name ->
                    if (first < 0) first = done
                    val el = (System.currentTimeMillis() - t0) / 1000.0
                    val rate = if (el > 3) (done - first) / el else 0.0
                    val eta = if (rate > 0) ((total - done) / rate / 60).toInt() else -1
                    val line = "${gb(done)} / ${gb(total)}" + (if (eta >= 0) tr(R.string.about_min_left, eta) else "") + "\n$name"
                    runOnUiThread { pb.progress = (done * 1000 / total.coerceAtLeast(1)).toInt(); info.text = line }
                    GenService.update(this, "${gb(done)} / ${gb(total)}", (done * 100 / total.coerceAtLeast(1)).toInt(), 100)
                }
                models.chosen = ids
                Logs.add("download done"); GenService.stop(this, tr(R.string.models_are_ready))
                runOnUiThread { running = false; if (models.ready(c)) route() else { listModels(); info.text = tr(R.string.done) } }
            } catch (e: InterruptedException) {
                say(tr(R.string.paused_press_again_to_resume)); GenService.stop(this, null)
            } catch (e: Throwable) {
                Logs.add("download failed: ${e.stackTraceToString().take(1500)}")
                say(tr(R.string.failed) + (e.message ?: e.javaClass.simpleName)); GenService.stop(this, null)
            } finally {
                runOnUiThread { running = false; pb.visibility = View.GONE; listModels(); boxes.values.forEach { it.isEnabled = true }; window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            }
        }
    }
    listModels()
    // the list on screen is the saved one; refresh it from the server in the background when one is set
    if (models.server.isNotEmpty()) thread { runCatching { models.fetchManifest() }.onSuccess { runOnUiThread { if (!running) listModels() } } }
    show(column(
        text(tr(R.string.models), 24f, true),
        note(c.title + if (c.isTested) "" else tr(R.string.beta), 14f),
        note(tr(R.string.kroma_downloads_the_unquantized_fp16)),
        note(tr(R.string.tap_the_arrow_for_options)),
        card(picks, sumTv),
        storageCard(),
        note(tr(R.string.uninstall_deletes_models)),
        card(wifi, mirror),
        pb, info, go,
        *(if (models.ready(c)) arrayOf(button(tr(R.string.back)) { if (!running) main() })
          // without models yet: look around first and download later (Settings > Models comes back here)
          else arrayOf(button(tr(R.string.later)) { if (!running) { prefs.edit().putBoolean("setup_later", true).apply(); main() } })),
        button(tr(R.string.send_the_log)) { bugReport() }))
    onBack = { if (!running && models.ready(c)) main() else if (!running) finish() }
}
