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

/* MainActivity: settings, the extended mode panel, presets, the adult option and the bug report (moved out of MainActivity.kt unchanged) */

/** First start: every picture straight to the gallery, or each one looked at first and saved with Save. Settings changes it later. */
internal fun MainActivity.askSaveMode() {
    val opts = arrayOf(tr(R.string.pick_what_to_save_save_button),
                       tr(R.string.save_every_picture_automatically))
    AlertDialog.Builder(this).setTitle(tr(R.string.saving_pictures))
        .setItems(opts) { _, i -> prefs.edit().putBoolean("auto_save", i == 1).apply(); Logs.add("save mode chosen: ${if (i == 1) "auto" else "pick"}")
            status(tr(R.string.you_can_change_this_in_settings)) }
        .setCancelable(false).show()
}

// ---------------------------------------------------------------- settings
/**
 * Everything the app keeps besides the model packs, with its size, in one place: imported models, LoRAs, the LoRA-baked copies of
 * the engine's weights (made again when needed), unfinished downloads -- on the Models screen, under the packs.
 */
internal fun MainActivity.storageCard(): View {
    val list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    val head = note(tr(R.string.searching))
    fun size(f: File): Long = if (f.isFile) f.length() else (f.listFiles() ?: emptyArray()).sumOf { size(it) }
    fun gbs(b: Long) = if (b >= (1L shl 30)) "%.1f GB".format(b / 1073741824.0) else "${b shr 20} MB"
    fun ask(title: String, b: Long, what: () -> Unit) { AlertDialog.Builder(this).setTitle(tr(R.string.delete_from_this_phone, title))
        .setMessage(gbs(b)).setPositiveButton(tr(R.string.delete)) { _, _ -> what() }.setNegativeButton(tr(R.string.cancel), null).show() }
    lateinit var fill: () -> Unit
    fun row(label: String, b: Long, onTap: () -> Unit) = TextView(this).apply {
        text = "$label\n${gbs(b)}"; textSize = 14f; setPadding(px(14), px(11), px(14), px(11)); setTextColor(INK); background = round(FIELD, 12)
        setOnClickListener { if (!running) onTap() } }
    fill = {
        list.removeAllViews(); head.text = tr(R.string.searching)
        thread {
            // (section title, rows) gathered off the UI thread: sizing a 23 GB folder walks its files
            val secs = ArrayList<Pair<String, List<Triple<String, Long, () -> Unit>>>>()
            val custom = Custom.list(models.root).map { m ->
                val b = size(m.dir); val using = Advanced.custom(m.kind) == m.name
                Triple("${m.name}  ·  ${ModelKind.family(m.kind)}" + if (using) "  ·  " + tr(R.string.in_use_2) else "", b, {
                    AlertDialog.Builder(this).setTitle(m.name).setItems(arrayOf(
                        if (using) tr(R.string.use_the_built_in_model) else tr(R.string.use_this_model),
                        tr(R.string.delete))) { _, i ->
                        if (i == 0) { Advanced.setCustom(m.kind, if (using) "" else m.name); fill() }
                        else ask(m.name, b) { if (using) Advanced.setCustom(m.kind, ""); m.dir.deleteRecursively(); Logs.add("model deleted"); fill() }
                    }.show(); Unit })
            }
            if (custom.isNotEmpty()) secs.add((tr(R.string.imported_models) + (if (!Advanced.on) "  ·  " + tr(R.string.used_in_extended_mode) else "")) to custom)
            for (k in ModelKind.entries.filter { it != ModelKind.YUE }) {
                val kind = k.id
                val ls = loras(kind).map { d -> val b = size(d)
                    Triple(d.name + if (loraList(kind).any { it.first == d.name }) "  ·  " + tr(R.string.in_use_2) else "", b,
                        { ask(d.name, b) { setLoraList(kind, loraList(kind).filterNot { it.first == d.name }); d.deleteRecursively(); Logs.add("lora deleted"); fill() } }) }
                if (ls.isNotEmpty()) secs.add("LoRA · ${k.family}" to ls)
            }
            // the LoRA-baked weights: beside the bundle's engine and beside each imported model's
            val baked = (ModelKind.entries.map { File(models.root, "${it.id}_bundle") } + Custom.list(models.root).map { it.dir })
                .map { File(it, "engine_lora") }.filter { it.isDirectory }.map { d -> val b = size(d)
                    Triple((d.parentFile?.name ?: "") + "  ·  " + tr(R.string.lora_merged_into_the_weights_made), b,
                        { ask(d.parentFile?.name ?: "", b) { d.deleteRecursively(); Logs.add("baked lora deleted"); fill() } }) }
            if (baked.isNotEmpty()) secs.add(tr(R.string.prepared_copies) to baked)
            // only what the app itself left: the .part of a file the model list names, its own download folder, conversions it marked
            val parts = models.offered().flatMap { p -> models.filesOf(p.id, chip!!) }.map { File(models.root, it.path + ".part") }.filter { it.isFile }.distinct()
            val left = listOfNotNull(File(Custom.root(models.root), ".download").takeIf { it.isDirectory }) + parts + Custom.unfinished(models.root)
            val lb = left.sumOf { size(it) }
            if (lb > 0) secs.add(tr(R.string.unfinished_downloads_and_conversions) to
                listOf(Triple(tr(R.string.d_items).format(left.size), lb, { ask(tr(R.string.unfinished_files), lb) { left.forEach { it.deleteRecursively() }; Logs.add("unfinished deleted"); fill() } })))
            val free = android.os.StatFs(models.root.absolutePath).availableBytes
            runOnUiThread {
                head.text = tr(R.string.free_on_this_phone_s).format(gbs(free)) +
                    if (secs.isEmpty()) "\n" + tr(R.string.nothing_besides_the_model_packs) else ""
                for ((t, rows) in secs) {
                    list.addView(note(t, 12f).apply { setPadding(0, px(12), 0, px(2)) })
                    for ((l, b, tap) in rows) list.addView(row(l, b, tap), LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = px(6) })
                }
            }
        }
    }
    fill()
    return card(note(tr(R.string.imported_models_loras_and_other_files)), head, list,
        note(tr(R.string.tap_an_item_to_use_or)))
}

internal fun MainActivity.settings() {
    var dlg: AlertDialog? = null
    lateinit var adult: Button
    fun label() { adult.text = if (prefs.getBoolean("adult", false)) tr(R.string.adult_content_on)
                               else tr(R.string.adult_content_off) }
    adult = button("") {
        val on = prefs.getBoolean("adult", false)
        if (on || prefs.getBoolean("adult_checked", false)) { prefs.edit().putBoolean("adult", !on).apply(); Logs.add("adult setting ${!on}"); label(); releaseBlurred() }
        else adultDialog { label(); releaseBlurred() }
    }
    label()
    // off by default: a picture is looked at first and saved with the Save button
    lateinit var auto: Button
    fun autoLabel() { auto.text = tr(if (prefs.getBoolean("auto_save", false)) R.string.auto_save_on else R.string.auto_save_off) }
    auto = button("") { prefs.edit().putBoolean("auto_save", !prefs.getBoolean("auto_save", false)).apply(); autoLabel() }
    autoLabel()
    // language: the phone's, or one picked here. The screen is rebuilt at once; not while something is running.
    // (no "system" chip: the first screen always picks one, and a phone language the app lacks fell back to English anyway)
    val langs = listOf("en", "ko", "zh", "zh-TW", "ja", "ru", "es", "pt-BR")
    val (langRow, _) = chips(listOf("English", "한국어", "简体中文", "繁體中文", "日本語", "Русский", "Español", "Português"), langs.indexOf(Lang.now()).coerceAtLeast(0), quiet = true, perRow = 4, textSp = 11.5f) { i ->   // "Português" was cut at 14 sp and 12.5 sp, four a row (8 Gen 2)
        if (running || langs[i] == Lang.code) return@chips
        Lang.set(this, langs[i]); Logs.add("language ${langs[i].ifEmpty { "system" }}"); dlg?.dismiss(); route()
    }
    // the stack spaces its children itself, so the edition without the setting leaves it out of the list rather than out of the view
    // the advanced mode: switching it rebuilds the main screen, so the badge appears or goes at once
    val advBtn = button(tr(if (Advanced.on) R.string.advanced_on else R.string.advanced_off)) {
        if (running) return@button
        Advanced.on = !Advanced.on; Logs.add("advanced mode ${Advanced.on}"); dlg?.dismiss(); route()
    }
    // the diagnosis log in Settings, not in the extended mode: anyone whose pictures come out broken should be able to turn it on
    lateinit var diagBtn: Button
    fun diagLabel() { diagBtn.text = if (Advanced.diagLog) tr(R.string.diagnosis_log_on_slower_for_a) else tr(R.string.diagnosis_log_off) }
    diagBtn = button("") { Advanced.diagLog = !Advanced.diagLog; diagLabel() }; diagLabel()
    val box = column(*listOfNotNull(note(tr(R.string.language)), langRow, auto, adult.takeIf { BuildConfig.ADULT_OPTION }, advBtn,
        button(tr(R.string.send_the_log)) { bugReport() },
        diagBtn,
        button(tr(R.string.license_and_notices)) { notices() },
        button(tr(R.string.models)) { if (!running) { dlg?.dismiss(); setup() } }).toTypedArray())
    dlg = AlertDialog.Builder(this).setView(ScrollView(this).apply { addView(column(text(tr(R.string.settings), 20f, true), box).apply { setBackgroundColor(BG) }) }).setPositiveButton("OK", null).show()
}

/**
 * Presets (kept in the app's preferences as JSON, "presets"): everything on screen in one -- the model, steps, ratio, count, the
 * model's LoRAs and strengths, the prompt and the negative, and in extended mode the fine-tune and the sampler settings. A tap
 * applies one (what it has: the older settings-only and prompt-only presets are kept and apply their part), a long press
 * deletes it, the last line saves what is on screen under a name.
 */
internal fun MainActivity.presets(promptEt: EditText) {
    val key = "presets"
    // the two earlier lists, merged in once
    if (!prefs.contains(key)) {
        val merged = org.json.JSONArray()
        for (k in listOf("presets_settings", "presets_prompt")) runCatching { org.json.JSONArray(prefs.getString(k, "[]")) }.getOrNull()?.let { a -> for (i in 0 until a.length()) merged.put(a.get(i)) }
        prefs.edit().putString(key, merged.toString()).apply()
    }
    val arr = runCatching { org.json.JSONArray(prefs.getString(key, "[]")) }.getOrDefault(org.json.JSONArray())
    val names = (0 until arr.length()).map { arr.getJSONObject(it).optString("name") }
    var dlg: AlertDialog? = null
    val box = column()
    fun LinearLayout.put(v: View) = addView(v, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = px(8) })
    names.forEachIndexed { i, n -> box.put(button(n) {
        val o = arr.getJSONObject(i); dlg?.dismiss()
        val e = prefs.edit()
        if (o.has("prompt")) e.putString("prompt", o.optString("prompt")) else e.putString("prompt", promptEt.text.toString())
        if (o.has("negative") && Advanced.on) Advanced.negative = o.optString("negative")
        if (o.has("model")) {
            val m = o.optString("model")
            e.putString("model", m).putInt("steps", o.optInt("steps", 8)).putString("ratio_name", o.optString("ratio", "2:3")).putInt("count", o.optInt("count", 1))
            o.optJSONArray("loras")?.let { arr2 -> setLoraList(m, (0 until arr2.length()).map { i2 -> arr2.getJSONObject(i2).let { j2 -> j2.getString("name") to j2.optInt("pct", 100) } }) }
                ?: o.optString("lora").takeIf { it.isNotEmpty() }?.let { setLoraList(m, listOf(it to o.optInt("lora_pct", 100))) }
            if (Advanced.on && o.has("finetune")) Advanced.setCustom(m, o.optString("finetune"))
            if (Advanced.on && m == "anima") o.optJSONObject("sampling")?.let { t ->
                Advanced.sampler = t.optString("sampler", Advanced.sampler); Advanced.scheduler = t.optString("scheduler", Advanced.scheduler)
                Advanced.cfg = t.optDouble("cfg", Advanced.cfg.toDouble()).toFloat(); Advanced.shift = t.optDouble("shift", Advanced.shift.toDouble()).toFloat() }
        }
        e.apply(); Logs.add("preset applied"); route()
    }.apply { setOnLongClickListener {
        AlertDialog.Builder(this@presets).setTitle(tr(R.string.delete_from_this_phone, n)).setPositiveButton(tr(R.string.delete)) { _, _ ->
            arr.remove(i); prefs.edit().putString(key, arr.toString()).apply(); dlg?.dismiss(); presets(promptEt) }.setNegativeButton(tr(R.string.cancel), null).show(); true } }) }
    if (names.isEmpty()) box.put(note(tr(R.string.none_saved_yet)))
    box.put(button(tr(R.string.read_from_a_picture)) {
        dlg?.dismiss(); startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("image/png"), MainActivity.PICK_META) })
    box.put(primary(tr(R.string.save_the_current_one)) {
        val nameEt = field(EditText(this).apply { hint = tr(R.string.name); setSingleLine() })
        AlertDialog.Builder(this).setView(column(nameEt)).setPositiveButton("OK") { _, _ ->
            val n = nameEt.text.toString().trim().ifEmpty { "${names.size + 1}" }
            val m = prefs.getString("model", "anima")!!
            val o = org.json.JSONObject().put("name", n)
            o.put("prompt", promptEt.text.toString()).put("negative", if (Advanced.on) Advanced.negative else "")
            o.put("model", m).put("steps", prefs.getInt("steps", 8)).put("ratio", prefs.getString("ratio_name", "2:3")).put("count", prefs.getInt("count", 1))
            o.put("loras", org.json.JSONArray().also { arr2 -> loraList(m).forEach { (n2, p2) -> arr2.put(org.json.JSONObject().put("name", n2).put("pct", p2)) } })
            if (Advanced.on) o.put("finetune", Advanced.custom(m))
            if (Advanced.on && m == "anima") o.put("sampling", org.json.JSONObject()
                .put("sampler", Advanced.sampler).put("scheduler", Advanced.scheduler).put("cfg", Advanced.cfg.toDouble()).put("shift", Advanced.shift.toDouble()))
            arr.put(o); prefs.edit().putString(key, arr.toString()).apply(); Logs.add("preset saved"); dlg?.dismiss(); presets(promptEt)
        }.setNegativeButton(tr(R.string.cancel), null).show()
    })
    dlg = AlertDialog.Builder(this).setTitle(tr(R.string.presets)).setView(ScrollView(this).apply { addView(box) }).show()
    dlg.window?.setBackgroundDrawable(round(CARD, 20))
}

/** Extended mode: the negative prompt under the prompt (used when CFG is above 1, which the extended settings set). */
internal fun MainActivity.negativeBox() = field(EditText(this).apply { negEt = this
    hint = tr(R.string.negative_prompt_used_when_cfg_1)
    setText(Advanced.negative); minLines = 2
    addTextChangedListener(object : android.text.TextWatcher {
        override fun afterTextChanged(e: android.text.Editable?) { Advanced.negative = e?.toString() ?: "" }
        override fun beforeTextChanged(a: CharSequence?, b: Int, c: Int, d: Int) {}
        override fun onTextChanged(a: CharSequence?, b: Int, c: Int, d: Int) {} }) })

/** The extended settings as a column; [rebuild] redraws whatever holds it (after a reset). */
internal fun MainActivity.extendedPanel(rebuild: () -> Unit): LinearLayout {
    Advanced.kind = Advanced.kindOf(loraModel)
    val samplers = Advanced.samplers()
    val (samplerRow, _) = chips(samplers.map { samplerLabel(it) }, samplers.indexOf(Advanced.sampler), quiet = true, perRow = 2) { Advanced.sampler = samplers[it] }
    val scheds = Advanced.schedulers()
    val (schedRow, _) = chips(scheds.map { Advanced.schedLabel(it) }, scheds.indexOf(Advanced.scheduler), quiet = true, perRow = 3) { Advanced.scheduler = scheds[it] }
    // a slider with a label that shows its value: v = from + step * position
    fun slider(label: (Float) -> String, from: Float, step: Float, count: Int, now: Float, save: (Float) -> Unit): Pair<TextView, SeekBar> {
        val tv = note(label(now))
        val sb = SeekBar(this).apply { max = count; progress = ((now - from) / step).roundToInt().coerceIn(0, count)
            progressTintList = android.content.res.ColorStateList.valueOf(ACCENT); thumbTintList = android.content.res.ColorStateList.valueOf(ACCENT)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(b: SeekBar?, v: Int, u: Boolean) { val f = from + step * v; tv.text = label(f); save(f) }
                override fun onStartTrackingTouch(b: SeekBar?) {}
                override fun onStopTrackingTouch(b: SeekBar?) {} }) }
        return tv to sb
    }
    val (shiftTv, shiftSb) = slider({ tr(R.string.adv_shift, "%.1f".format(it)) }, 1f, 0.5f, 18, Advanced.shift) { Advanced.shift = it }
    val (cfgTv, cfgSb) = slider({ tr(R.string.adv_cfg, "%.1f".format(it)) }, 1f, 0.5f, 18, Advanced.cfg) { Advanced.cfg = it }
    lateinit var heatBtn: Button
    fun heatLabel() { heatBtn.text = if (Advanced.showHeat) tr(R.string.temperatures_shown)
                                      else tr(R.string.temperatures_hidden) }
    heatBtn = button("") { Advanced.showHeat = !Advanced.showHeat; heatLabel(); if (!running) route() }; heatLabel()
    lateinit var stopBtn: Button
    fun stopLabel() { stopBtn.text = if (Advanced.stopBlock) tr(R.string.stop_after_the_current_block_the)
                      else tr(R.string.stop_at_the_end_of_the) }
    stopBtn = button("") { Advanced.stopBlock = !Advanced.stopBlock; stopLabel() }; stopLabel()
    val graphBtn = button(if (Advanced.nodeGraph) tr(R.string.node_graph_on)
                          else tr(R.string.node_graph_off)) { if (!running) { Advanced.nodeGraph = !Advanced.nodeGraph; route() } }
    // YuE2 makes songs with its own fixed sampling: nothing here applies to it
    if (loraModel == "yue")
        return column(note(tr(R.string.this_model_samples_in_its_own)),
            graphBtn, heatBtn, stopBtn)
    val ownLoop = loraModel == "klein" || loraModel == "klein9" || loraModel == "sdxl"
    return column(*listOfNotNull(note(tr(R.string.adv_help)),
        when (loraModel) {
            "moss" -> note(tr(R.string.these_apply_to_moss_image_turbo))
            "krea" -> note(tr(R.string.these_apply_to_krea_2_the))
            "klein", "klein9", "sdxl" -> null          // no per-model line: the Anima wording would be wrong for them
            else -> note(tr(R.string.these_apply_to_anima_a_turbo))
        },
        // Klein and SDXL run their own sampling loops: only CFG and the negative apply to them
        *(if (ownLoop) emptyArray<View>() else arrayOf(note(tr(R.string.adv_sampler)), samplerRow, note(tr(R.string.adv_scheduler)), schedRow, shiftTv, shiftSb)),
        graphBtn, cfgTv, cfgSb, note(tr(R.string.adv_cfg_help)), heatBtn, stopBtn,
        button(tr(R.string.adv_reset)) {
            Advanced.resetSampling()
            rebuild()
        }).toTypedArray())
}

/**
 * Extended mode: a handle at the foot of the main screen -- a tab with a chevron on a hairline -- that opens and closes the
 * extended settings in place (the state is kept).
 */
internal fun MainActivity.extendedDrawer(): View {
    val holder = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }        // no padding of its own: the handle sits right under the buttons
    fun fill() {
        holder.removeAllViews()
        val open = prefs.getBoolean("ext_open", false)
        extRefill = { fill() }
        val handle = object : View(this) {
            val line = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply { color = MUTED; strokeWidth = px(1).toFloat(); style = android.graphics.Paint.Style.STROKE }
            val tab = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply { color = CARD }
            val chev = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply { color = ACCENT; strokeWidth = px(2).toFloat(); style = android.graphics.Paint.Style.STROKE
                strokeCap = android.graphics.Paint.Cap.ROUND; strokeJoin = android.graphics.Paint.Join.ROUND }
            override fun onDraw(c: android.graphics.Canvas) {
                val w = width.toFloat(); val h = height.toFloat(); val cx = w / 2; val base = h - px(1); val top = px(2).toFloat()
                val half = px(46).toFloat(); val slope = px(14).toFloat()
                val path = android.graphics.Path().apply { moveTo(cx - half - slope, base); lineTo(cx - half, top); lineTo(cx + half, top); lineTo(cx + half + slope, base); close() }
                c.drawPath(path, tab)
                c.drawLine(0f, base, cx - half - slope, base, line); c.drawLine(cx + half + slope, base, w, base, line)
                c.drawLine(cx - half - slope, base, cx - half, top, line); c.drawLine(cx - half, top, cx + half, top, line); c.drawLine(cx + half, top, cx + half + slope, base, line)
                val my = (top + base) / 2; val dx = px(9).toFloat(); val dy = px(4).toFloat() * (if (open) 1 else -1)
                c.drawLine(cx - dx, my - dy, cx, my + dy, chev); c.drawLine(cx, my + dy, cx + dx, my - dy, chev)
            }
        }.apply { contentDescription = tr(R.string.advanced_settings); isClickable = true
            setOnClickListener { prefs.edit().putBoolean("ext_open", !open).apply(); fill()
                if (!open) holder.post { generateSequence(holder.parent) { (it as? View)?.parent }.filterIsInstance<ScrollView>().firstOrNull()?.smoothScrollBy(0, px(360)) } } }
        holder.addView(handle, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, px(26)))
        if (open) holder.addView(extendedPanel { fill() }.apply { setPadding(0, px(14), 0, 0) })
    }
    fill()
    return holder
}

/** Birth year + "I am an adult" + one confirmation, before flagged pictures are shown and saved. */
internal fun MainActivity.adultDialog(done: () -> Unit) {
    val adultAge = if (java.util.Locale.getDefault().country == "KR") 19 else 18
    val yearEt = EditText(this).apply { hint = tr(R.string.year_of_birth); inputType = InputType.TYPE_CLASS_NUMBER; setSingleLine() }
    val chk = check(tr(R.string.i_am_an_adult))
    val msg = text("", 13f)
    val d = AlertDialog.Builder(this).setTitle(tr(R.string.show_adult_content))
        .setView(column(field(yearEt), chk, msg)).setPositiveButton(tr(R.string.next), null).setNegativeButton(tr(R.string.cancel), null).create()
    d.setOnShowListener {
        d.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
            val y = yearEt.text.toString().toIntOrNull()
            val now = java.util.Calendar.getInstance().get(java.util.Calendar.YEAR)
            when {
                y == null || y < 1900 || y > now -> msg.text = tr(R.string.enter_your_year_of_birth)
                now - y < adultAge -> msg.text = tr(R.string.this_is_available_to_adults)
                !chk.isChecked -> msg.text = tr(R.string.tick_i_am_an_adult)
                else -> { d.dismiss()
                    AlertDialog.Builder(this).setTitle(tr(R.string.confirm))
                        .setMessage(tr(R.string.pictures_the_classifier_rates_as))
                        .setPositiveButton(tr(R.string.turn_on)) { _, _ -> prefs.edit().putBoolean("adult", true).putBoolean("adult_checked", true).apply(); Logs.add("adult setting true"); done() }
                        .setNegativeButton(tr(R.string.cancel), null).show() }
            }
        }
    }
    d.show()
}

internal fun MainActivity.bugReport() {
    val f = Logs.report(this, models.root)
    val uri = FileProvider.getUriForFile(this, "$packageName.share", f)
    startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_STREAM, uri)
        .putExtra(Intent.EXTRA_SUBJECT, "Kroma log").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION), tr(R.string.send_the_log)))
}

/** a sampler's name on screen */
internal fun samplerLabel(n: String) = when (n) {
    "turbo8" -> tr(R.string.turbo_8_steps)
    "turbo4" -> tr(R.string.turbo_4_steps)
    "er_sde" -> "ER-SDE"; "euler" -> "Euler"; "euler_ancestral" -> "Euler a"; "dpmpp_2m" -> "DPM++ 2M"; else -> n
}

/**
 * A Kroma picture's settings (PngMeta's JSON) back on the screen: the model, the prompt (as written, before a translation), the
 * negative, the seed, the steps, the size, the LoRAs and the sampling. A picture saved before model_id was written
 * is matched by its model's name.
 */
internal fun MainActivity.applyPictureMeta(text: String) {
    val o = runCatching { org.json.JSONObject(text) }.getOrNull() ?: return
    val name = o.optString("model")
    val m = o.optString("model_id").ifEmpty { when {
        name.startsWith("Anima") -> "anima"; name.startsWith("Krea") -> "krea"; name.contains("Klein 9B") -> "klein9"; name.contains("Klein") -> "klein"
        name.startsWith("SDXL") -> "sdxl"; name.startsWith("MOSS") -> "moss"; else -> "" } }
    val e = prefs.edit(); val got = ArrayList<String>()
    if (m.isNotEmpty()) { e.putString("model", m); got.add(m) }
    (o.optString("prompt_original").ifEmpty { o.optString("prompt") }).takeIf { it.isNotEmpty() }?.let { e.putString("prompt", it); got.add("prompt") }
    if (o.has("steps")) { e.putInt("steps", o.optInt("steps")); got.add("steps") }
    if (o.has("width") && o.has("height")) { e.putInt("size_w", o.optInt("width")).putInt("size_h", o.optInt("height")); got.add("${o.optInt("width")}x${o.optInt("height")}") }
    e.apply()
    if (m.isNotEmpty()) o.optJSONArray("loras")?.let { a -> setLoraList(m, (0 until a.length()).map { i -> a.getJSONObject(i).let { j -> j.getString("name") to j.optInt("pct", 100) } }); got.add("LoRA") }
        ?: o.optString("lora").takeIf { it.isNotEmpty() && m.isNotEmpty() }?.let { setLoraList(m, listOf(it to (o.optDouble("lora_strength", 100.0)).toInt())); got.add("LoRA") }
    // the sampling where the extended settings are on (they are what it is read from there)
    if (Advanced.on && m in setOf("klein", "klein9") && o.optString("finetune").let { f -> f.isNotEmpty() && Custom.list(models.root, m).any { it.name == f } }) Advanced.setCustom(m, o.optString("finetune"))
    if (Advanced.on && m in setOf("anima", "krea", "moss", "klein", "klein9", "sdxl")) { val was = Advanced.kind; Advanced.kind = m
        try { o.optString("sampler").takeIf { it.isNotEmpty() }?.let { Advanced.sampler = it }; o.optString("scheduler").takeIf { it.isNotEmpty() }?.let { Advanced.scheduler = it }
              if (o.has("cfg")) Advanced.cfg = o.optDouble("cfg").toFloat(); if (o.has("shift")) Advanced.shift = o.optDouble("shift").toFloat()
              if (o.has("negative")) Advanced.negative = o.optString("negative") } finally { Advanced.kind = was } }
    if (o.has("seed")) { pendingSeed = o.optLong("seed"); got.add("seed ${o.optLong("seed")}") }
    Logs.add("picture settings read: " + got.joinToString())
    route()
    status(tr(R.string.settings_read_from_the_picture_s).format(got.joinToString(", ")))
}
