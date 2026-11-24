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

/* MainActivity: LoRAs and imported models: the lists, the dialogs, converting and downloading them (moved out of MainActivity.kt unchanged) */

// ---------------------------------------------------------------- LoRA
// <models>/lora/<model>/<name>/: a LoRA file converted on this phone ([LoraImport]). One at a time per model, with a strength.
internal fun MainActivity.loraRoot(model: String) = File(models.root, "lora/$model")

internal fun MainActivity.loras(model: String) = (loraRoot(model).listFiles { f -> f.isDirectory && !f.name.startsWith(".") } ?: emptyArray()).sortedBy { it.name.lowercase() }

/** the LoRAs chosen for a model and their strengths in hundredths, in the order they were picked (an older single choice comes over) */
internal fun MainActivity.loraList(model: String): List<Pair<String, Int>> {
    val raw = prefs.getString("loras_$model", null)
    if (raw == null) return prefs.getString("lora_$model", null)?.let { listOf(it to prefs.getInt("lora_pct_$model", 100)) } ?: emptyList()
    val a = runCatching { org.json.JSONArray(raw) }.getOrNull() ?: return emptyList()
    return (0 until a.length()).map { a.getJSONObject(it).let { o -> o.getString("name") to o.optInt("pct", 100) } }
        .filter { File(loraRoot(model), it.first).isDirectory }
}

internal fun MainActivity.setLoraList(model: String, v: List<Pair<String, Int>>) {
    val a = org.json.JSONArray(); v.forEach { (n, p) -> a.put(org.json.JSONObject().put("name", n).put("pct", p)) }
    prefs.edit().putString("loras_$model", a.toString()).remove("lora_$model").remove("lora_pct_$model").apply()
}

internal fun MainActivity.loraOf(model: String) = loraList(model).firstOrNull()?.let { File(loraRoot(model), it.first) }?.takeIf { it.isDirectory }

/** a LoRA strength (kept in hundredths) as people write it: 1.0, 0.65, 0.7 */
internal fun MainActivity.strengthText(pct: Int) = strengthOf(pct)

internal fun MainActivity.loraLabel() { val v = loraList(loraModel)
    loraBtn?.visibility = if (loraModel == "yue") View.GONE else View.VISIBLE      // YuE2 comes as it is: no LoRAs, no fine-tunes
    yueLenBtn?.visibility = if (loraModel == "yue") View.VISIBLE else View.GONE
    imgStartBtn?.visibility = if (loraModel != "yue" && loraModel != "moss") View.VISIBLE else View.GONE
    // Klein takes the start image as its reference (editing), not as img2img: no strength or mask for it
    imgStartBtn?.text = startLabel(); startThumbsRefresh()
    denoiseBtn?.visibility = if (loraModel != "yue" && loraModel != "moss" && !loraModel.startsWith("klein") && imgStart().isFile) View.VISIBLE else View.GONE
    maskBtn?.visibility = denoiseBtn?.visibility ?: View.GONE; fitBtn?.visibility = denoiseBtn?.visibility ?: View.GONE
    loraBtn?.text = when {
        v.isEmpty() -> tr(R.string.lora_off)
        v.size == 1 -> tr(R.string.lora_on, v[0].first, strengthText(v[0].second))
        else -> tr(R.string.lora_d_together).format(v.size)
    } }

/** a LoRA's info.json (what the trainer wrote, plus the tag and note kept here) */
internal fun MainActivity.loraInfo(d: File) = runCatching { org.json.JSONObject(File(d, "info.json").readText()) }.getOrDefault(org.json.JSONObject())

internal fun MainActivity.loraInfoPut(d: File, k: String, v: String) {
    val o = loraInfo(d); if (v.isEmpty()) o.remove(k) else o.put(k, v)
    runCatching { File(d, "info.json").writeText(o.toString()) }
}

/** what a LoRA is called in the list: set in its ⓘ sheet, or the folder's name without the trainer's step / epoch counters and
 *  the model's own name (every one here is for this model) */
internal fun MainActivity.loraTitle(d: File): String {
    loraInfo(d).optString("label").takeIf { it.isNotBlank() }?.let { return it }
    var t = d.name
    for (r in listOf("[_-]?c\\d+[_-]st\\d+$", "[_-]st\\d+$", "[_-]step\\d+$", "[_-]epoch[_-]?\\d+$", "[_-]E\\d+$", "[_-]0*\\d{4,}$", "[_-]\\d+e-\\d+s$"))
        t = t.replace(Regex(r, RegexOption.IGNORE_CASE), "")
    t = t.replace(Regex("(?i)[_ -]*\\(?krea[_ ]?2\\)?[_ -]*|(?i)[_-]k2(?=[_-]|$)|(?i)[_-]krea(?=[_-]|$)|(?i)^krea[_-]?2?[_-]"), " ")
    t = t.replace('_', ' ').replace(Regex("\\s+"), " ").trim(' ', '-', '.')
    return t.ifEmpty { d.name }
}

internal fun MainActivity.loraTag(d: File) = loraInfo(d).optString("tag").ifEmpty { tr(R.string.not_sorted) }

/** the words a LoRA wants in the prompt: what the ⓘ sheet says, or the tag it was trained on most */
internal fun MainActivity.loraTrigger(d: File) = loraInfo(d).let { o ->
    if (o.has("trig")) o.optString("trig") else o.optJSONArray("triggers")?.optString(0).orEmpty()
}.trim()

/**
 * A LoRA's trigger words follow it into the prompt and out again. The prompt is treated as a comma-separated list, which is
 * how these are written; a word that is not its own item is left alone rather than cut out of the middle of a sentence.
 */
internal fun MainActivity.promptTrigger(d: File, on: Boolean, promptEt: EditText?) {
    val et = promptEt ?: return
    val words = loraTrigger(d).split(',').map { it.trim() }.filter { it.isNotEmpty() }
    if (words.isEmpty()) return
    var parts = et.text.toString().split(',').map { it.trim() }.filter { it.isNotEmpty() }.toMutableList()
    // a trigger goes in front ("trigger, the prompt"), in the LoRA's own order, as trainers caption it
    if (on) parts = (words.filter { w -> parts.none { it.equals(w, true) } } + parts).toMutableList()
    else for (w in words) { val at = parts.indexOfFirst { it.equals(w, true) }; if (at >= 0) parts.removeAt(at) }
    val now = parts.joinToString(", ")
    if (now != et.text.toString()) { et.setText(now); prefs.edit().putString("prompt", now).apply() }
}

/**
 * What one LoRA is: the trainer's title, rank and alpha, how many Linears went in, its trigger words (a tap puts one in the
 * prompt), and the two things kept here -- a group it belongs to and a note. Deleting it from the phone is at the foot.
 */
internal fun MainActivity.loraSheet(d: File, promptEt: EditText?, after: () -> Unit) {
    val o = loraInfo(d); var dlg: AlertDialog? = null
    val box = column(note(tr(R.string.name)))
    val labelEt = field(EditText(this).apply { setText(loraTitle(d)); setSingleLine(); textSize = 17f })
    box.addView(labelEt)
    box.addView(note(d.name + (o.optString("title").takeIf { it.isNotEmpty() && it != d.name }?.let { "  ·  $it" } ?: ""), 12f))
    fun line2(k: String, v: String) { if (v.isNotEmpty()) box.addView(note("$k  $v", 14f)) }
    line2(tr(R.string.model), ModelKind.family(d.parentFile?.name ?: "krea"))   // lora/<model id>/<name>
    line2(tr(R.string.rank), o.optInt("rank").takeIf { it > 0 }?.let { r -> "$r" + (o.optDouble("alpha").takeIf { !it.isNaN() && it > 0 }?.let { aa -> " · alpha %.0f".format(aa) } ?: "") } ?: "")
    line2(tr(R.string.layers), o.optInt("pairs").takeIf { it > 0 }?.toString() ?: "")
    line2(tr(R.string.trained),
        listOfNotNull(o.optInt("steps").takeIf { it > 0 }?.let { "$it steps" }, o.optString("by").ifEmpty { null }, o.optString("date").ifEmpty { null }).joinToString(" · "))
    line2(tr(R.string.size), gb(d.walkTopDown().filter { it.isFile }.sumOf { it.length() }))
    // the trigger words: a tap adds one to the prompt
    box.addView(note(tr(R.string.trigger_words_which_follow_it_into)))
    val trigEt = field(EditText(this).apply { setText(loraTrigger(d)); setSingleLine() })
    box.addView(trigEt)
    val trg = o.optJSONArray("triggers")
    if (trg != null && trg.length() > 0) {
        box.addView(note(tr(R.string.trigger_words_tap_to_add)))
        box.addView(note(tr(R.string.most_used_tags_tap_to_add), 12f))
        val words = (0 until trg.length()).map { trg.getString(it) }
        val (r, _) = chips(words, -1, quiet = true, perRow = 2) { i ->
            val cur = trigEt.text.toString().trim()
            if (!cur.split(',').any { it.trim().equals(words[i], true) })
                trigEt.setText(if (cur.isEmpty()) words[i] else "$cur, " + words[i])
        }
        box.addView(r)
    }
    box.addView(note(tr(R.string.group)))
    val tagEt = field(EditText(this).apply { setText(o.optString("tag")); setSingleLine(); hint = tr(R.string.not_sorted) })
    box.addView(tagEt)
    box.addView(note(tr(R.string.note)))
    val noteEt = field(EditText(this).apply { setText(o.optString("note")); minLines = 2; gravity = Gravity.TOP })
    box.addView(noteEt)
    dlg = AlertDialog.Builder(this).setView(ScrollView(this).apply { addView(box) })
        .setPositiveButton("OK") { _, _ ->
            val was = loraTrigger(d)
            loraInfoPut(d, "tag", tagEt.text.toString().trim()); loraInfoPut(d, "note", noteEt.text.toString().trim())
            loraInfoPut(d, "trig", trigEt.text.toString().trim())
            labelEt.text.toString().trim().let { lb -> loraInfoPut(d, "label", if (lb == loraTitle(d) && !loraInfo(d).has("label")) "" else lb) }
            // in use and its trigger changed: the prompt follows
            if (loraList(d.parentFile?.name ?: "krea").any { it.first == d.name } && was != loraTrigger(d)) {
                promptEt?.let { et -> val keep = loraTrigger(d)
                    was.split(',').map { w -> w.trim() }.filter { w -> w.isNotEmpty() }.forEach { w ->
                        val ps = et.text.toString().split(',').map { x -> x.trim() }.filter { x -> x.isNotEmpty() && !x.equals(w, true) }
                        et.setText(ps.joinToString(", ")) }
                    if (keep.isNotEmpty()) promptTrigger(d, true, et) }
            }
            after() }
        .setNeutralButton(tr(R.string.delete)) { _, _ ->
            AlertDialog.Builder(this).setTitle(tr(R.string.delete_from_this_phone, d.name))
                .setPositiveButton(tr(R.string.delete)) { _, _ ->
                    val m = d.parentFile?.name ?: "krea"; setLoraList(m, loraList(m).filterNot { it.first == d.name }); d.deleteRecursively(); Logs.add("lora deleted"); loraLabel(); after() }
                .setNegativeButton(tr(R.string.cancel), null).show() }
        .setNegativeButton(tr(R.string.cancel), null).show()
    dlg.window?.setBackgroundDrawable(round(CARD, 20))
    dlg.getButton(AlertDialog.BUTTON_NEUTRAL)?.setTextColor(Color.rgb(0xE5, 0x5B, 0x5B))
}

/**
 * The LoRAs for the model on screen. With many of them a flat list is useless, so: a search box, the chosen ones at the top,
 * and the rest in groups that fold away (the group is set per LoRA in its ⓘ sheet). A tap puts one in or takes it out --
 * several add up, their ranks put side by side into one folder before a run -- and each chosen one has its own strength.
 */
internal fun MainActivity.loraDialog(promptEt: EditText? = null) {
    val model = loraModel; var dlg: AlertDialog? = null
    val list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    val search = field(EditText(this).apply { hint = tr(R.string.search); setSingleLine() })
    val folded = HashSet<String>()
    fun fill() {
        list.removeAllViews()
        val chosen = loraList(model); val q = search.text.toString().trim().lowercase()
        val all = loras(model)
        fun matches(d: File) = q.isEmpty() || d.name.lowercase().contains(q) || loraTitle(d).lowercase().contains(q) || loraInfo(d).let { o ->
            o.optString("title").lowercase().contains(q) || o.optString("tag").lowercase().contains(q) || o.optString("note").lowercase().contains(q) }
        fun add(v: View, gap: Boolean = true) = list.addView(v, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { if (gap) topMargin = px(6) })
        fun row(d: File): View {
            val at = chosen.indexOfFirst { it.first == d.name }; val on = at >= 0
            val name = TextView(this).apply {
                text = (if (on) "${at + 1}. " else "") + loraTitle(d)
                textSize = 15f; setPadding(px(14), px(13), px(6), px(13)); setTextColor(if (on) ACCENT else INK); setSingleLine(); ellipsize = android.text.TextUtils.TruncateAt.END
                setOnClickListener {
                    val v = loraList(model).toMutableList()
                    if (on) v.removeAll { it.first == d.name } else v.add(d.name to 100)
                    setLoraList(model, v); promptTrigger(d, !on, promptEt); loraLabel(); fill()
                }
            }
            return LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
                background = round(FIELD, 12).apply { if (on) setStroke(px(1) + 1, ACCENT) }
                addView(name, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                if (on) addView(field(EditText(this@loraDialog).apply {
                    setText(strengthText(chosen[at].second)); setSingleLine(); gravity = Gravity.CENTER
                    inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL or InputType.TYPE_NUMBER_FLAG_SIGNED
                    addTextChangedListener(object : android.text.TextWatcher {
                        override fun beforeTextChanged(c: CharSequence?, x: Int, y: Int, z: Int) {}
                        override fun onTextChanged(c: CharSequence?, x: Int, y: Int, z: Int) {}
                        override fun afterTextChanged(e: android.text.Editable?) {
                            val pct = e.toString().toDoubleOrNull()?.let { Math.round(it.coerceIn(-10.0, 10.0) * 100).toInt() } ?: return
                            setLoraList(model, loraList(model).map { if (it.first == d.name) it.first to pct else it }); loraLabel()
                        }
                    })
                }), LinearLayout.LayoutParams(px(76), ViewGroup.LayoutParams.WRAP_CONTENT))
                addView(TextView(this@loraDialog).apply {
                    text = "\u24D8"; textSize = 19f; setTextColor(MUTED); setPadding(px(10), px(12), px(14), px(12))
                    setOnClickListener { loraSheet(d, promptEt) { loraLabel(); fill() } }
                })
            }
        }
        // none, then the chosen ones, then the groups
        add(TextView(this).apply {
            text = tr(R.string.no_lora); textSize = 15f; setPadding(px(14), px(13), px(14), px(13))
            setTextColor(if (chosen.isEmpty()) ACCENT else INK)
            background = round(FIELD, 12).apply { if (chosen.isEmpty()) setStroke(px(1) + 1, ACCENT) }
            setOnClickListener { all.filter { d -> chosen.any { it.first == d.name } }.forEach { promptTrigger(it, false, promptEt) }
                setLoraList(model, emptyList()); loraLabel(); fill() }
        }, false)
        val used = chosen.mapNotNull { c -> all.firstOrNull { it.name == c.first } }
        if (used.isNotEmpty()) { add(note(tr(R.string.in_use))); used.forEach { add(row(it)) } }
        val rest = all.filter { d -> used.none { it.name == d.name } && matches(d) }
        for ((g, items) in rest.groupBy { loraTag(it) }.toSortedMap().mapValues { e -> e.value.sortedBy { loraTitle(it).lowercase() } }) {
            val shut = g in folded && q.isEmpty()
            add(TextView(this).apply {
                text = (if (shut) "▸  " else "▾  ") + g + "   " + items.size
                textSize = 14f; setTextColor(MUTED); setPadding(px(10), px(10), px(10), px(10))
                setOnClickListener { if (shut) folded.remove(g) else folded.add(g); fill() }
            })
            if (!shut) items.forEach { add(row(it)) }
        }
        if (all.isEmpty()) add(note(tr(R.string.no_lora_on_this_phone_yet)))
    }
    search.addTextChangedListener(object : android.text.TextWatcher {
        override fun beforeTextChanged(c: CharSequence?, x: Int, y: Int, z: Int) {}
        override fun onTextChanged(c: CharSequence?, x: Int, y: Int, z: Int) {}
        override fun afterTextChanged(e: android.text.Editable?) = fill()
    })
    fill(); loraRefill = { fill() }
    val v = column(text("LoRA", 20f, true), search, list,
        button(tr(R.string.add_a_lora_file)) { startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"), PICK_LORA) },
        button(tr(R.string.find_a_lora)) { dlg?.dismiss(); loraBrowser() },
        button(loraModeLabel()) { pickLoraMode { dlg?.dismiss(); loraDialog(promptEt) } },
        note(tr(R.string.lora_help))).apply { setBackgroundColor(BG) }
    dlg = AlertDialog.Builder(this).setView(ScrollView(this).apply { addView(v) }).setPositiveButton("OK", null).setOnDismissListener { loraRefill = null }.show()
}

/**
 * A fine-tune picked from the phone's files: converted into the engine's files by kconv, reading the document where it is
 * (through /proc/<pid>/fd, no copy), then chosen for its model. The Generate button stops it.
 */
internal fun MainActivity.importModel(uri: android.net.Uri) {
    if (running) return
    val (shown, size) = runCatching { contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME, android.provider.OpenableColumns.SIZE), null, null, null)?.use { c ->
        if (c.moveToFirst()) (c.getString(0) ?: "model") to c.getLong(1) else null } }.getOrNull() ?: ("model" to 0L)
    if (!Custom.isModelFile(shown)) { status(tr(R.string.pick_a_safetensors_or_gguf_file)); return }
    // what the file is stored as, from its header, before the format it is converted to is asked ([ModelSrc]); a format the
    // converter does not read is refused here rather than after a long conversion fails
    val fmt = runCatching { contentResolver.openInputStream(uri)?.use { ModelSrc.fromStream(shown, it) } }.getOrNull() ?: ""
    if (fmt.isNotEmpty() && fmt != "gguf" && !ModelSrc.readable(fmt)) { status(tr(R.string.s_is_not_read_by_the).format(ModelSrc.label(fmt))); return }
    Logs.run("model import"); Logs.add("model import: ${size shr 20}MB, $fmt")
    pickFormat(fmt) { importModelAs(uri, shown, size) }
}

private fun MainActivity.importModelAs(uri: android.net.Uri, shown: String, size: Long) {
    // the converted files are about the checkpoint's size in fp16 (a bf16 or fp16 file: the same, an fp32 one: half)
    val free = android.os.StatFs(models.root.absolutePath).availableBytes
    if (free < Custom.outBytes(size) + (1L shl 30)) { status(tr(R.string.not_enough_space_d_gb_needed).format((Custom.outBytes(size) shr 30) + 1, free shr 30)); return }
    val pfd = runCatching { contentResolver.openFileDescriptor(uri, "r") }.getOrNull()
    if (pfd == null) { status(tr(R.string.could_not_open_the_file)); return }
    modelJob(Custom.freeName(models.root, shown), { runCatching { pfd.close() } }) { "/proc/${android.os.Process.myPid()}/fd/${pfd.fd}" }
}

/**
 * Getting a fine-tune ready, on a thread with the Generate button as Stop: [source] gives a path the converter can read (a
 * download goes there first), then kconv converts it into custom/[name] and it is chosen for its model. [after] runs at the end.
 */
internal fun MainActivity.modelJob(name: String, after: () -> Unit = {}, defaults: Advanced.Defaults? = null, source: () -> String) {
    running = true; stopRequested = false; goBtn.text = tr(R.string.stop_2); bar.isIndeterminate = true; bar.visibility = View.VISIBLE
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    val t0 = System.currentTimeMillis()
    thread {
        val msg = try {
            val src = source(); if (stopRequested) throw InterruptedException()
            status(tr(R.string.converting_s).format(name))
            val kind = Custom.convert(applicationInfo.nativeLibraryDir, models.root, src, name, { stopRequested }) { d, n ->
                runOnUiThread { bar.isIndeterminate = false; bar.max = n; bar.progress = d; grid.visibility = View.VISIBLE; grid.set(d, n) }
                status(tr(R.string.converting_s_block_d_of_d).format(name, d, n)) }
            defaults?.let { Custom.writeDefaults(File(Custom.root(models.root), name), it) }
            Advanced.setCustom(kind, name); Logs.add("model ready: $kind in ${(System.currentTimeMillis() - t0) / 1000}s")
            tr(R.string.s_is_ready_and_chosen_for)
                .format(name, ModelKind.family(kind))
        } catch (e: InterruptedException) { tr(R.string.stopped_2) }
          catch (e: Throwable) { if (stopRequested) tr(R.string.stopped_2) else {
              Logs.add("model failed: ${e.message?.take(200)}"); tr(R.string.could_not_get_it_ready_s).format(e.message?.take(160) ?: "") } }
        runCatching { after() }
        runOnUiThread { running = false; goBtn.text = tr(R.string.generate); bar.isIndeterminate = false; bar.progress = 0; grid.visibility = View.GONE
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); route(); status(msg) }
    }
}

/** a model found online: downloaded into custom/.download (needs its size twice over for a moment), converted, the download deleted */
internal fun MainActivity.downloadModel(h: LoraBrowse.Hit) {
    if (running) return
    val free = android.os.StatFs(models.root.absolutePath).availableBytes shr 20
    if (free < h.sizeMb + (Custom.outBytes(h.sizeMb.toLong() shl 20) shr 20) + 1024) { status(tr(R.string.not_enough_space_d_gb_needed).format((h.sizeMb + (Custom.outBytes(h.sizeMb.toLong() shl 20) shr 20)) / 1024 + 1, free / 1024)); return }
    val tmp = File(Custom.root(models.root), ".download/${h.file}").apply { parentFile?.mkdirs() }
    Logs.run("model download"); Logs.add("model download from ${if (h.repo != null) "huggingface" else "civitai"}, ${h.sizeMb}MB")
    modelJob(Custom.freeName(models.root, h.file), { tmp.delete() }, h.defaults) {
        LoraBrowse.models = true
        LoraBrowse.download(h, tmp, { stopRequested }) { got, total ->
            val all = maxOf(total, h.sizeMb.toLong() shl 20)
            runOnUiThread { bar.isIndeterminate = false; bar.max = (all shr 20).toInt(); bar.progress = (got shr 20).toInt() }
            status(tr(R.string.downloading_s_d_d_mb)
                .format(h.file, got shr 20, all shr 20)) }
        tmp.absolutePath
    }
}

/** At the foot of the extended settings: bringing a model in; the format it is converted to is asked first, in the step itself. */
internal fun MainActivity.importSection(): View {
    val box = stack(0, 10, arrayOf(note(tr(R.string.bring_in_a_model))))
    fun LinearLayout.put(v: View) = addView(v, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = px(10) })
    box.put(note(tr(R.string.an_anima_krea_2_moss_image)))
    box.put(button(tr(R.string.import_a_model_file)) {
        if (!running) startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"), PICK_MODEL) })
    if (browseKinds().isNotEmpty())
        box.put(button(tr(R.string.find_a_model_online)) { if (!running) loraBrowser(true) })
    return box
}

private val LORA_MODES = listOf("run", "bake", "bake_once")
private fun MainActivity.loraModeNames() = arrayOf(
    tr(R.string.as_they_run_nothing_written),
    tr(R.string.baked_copy_kept_fast_again_next),
    tr(R.string.baked_copy_removed_after))
internal fun MainActivity.loraModeLabel() = tr(R.string.lora_method_s)
    .format(loraModeNames()[LORA_MODES.indexOf(Advanced.loraMode).coerceAtLeast(0)].substringBefore(" ("))

/** Runtime LoRAs (the default) cost a little each step; baking costs once (Krea 2: minutes and a copy of the blocks) and then
 *  runs at the model's own speed. One path or the other: what the chosen one cannot do stops with the reason ([Engine.loraError]). */
internal fun MainActivity.pickLoraMode(then: () -> Unit) {
    AlertDialog.Builder(this).setTitle(tr(R.string.lora_method))
        .setSingleChoiceItems(loraModeNames(), LORA_MODES.indexOf(Advanced.loraMode).coerceAtLeast(0)) { d, i -> Advanced.loraMode = LORA_MODES[i]; d.dismiss(); then() }
        .setNegativeButton(tr(R.string.cancel), null).show()
}

/** A model's files with what each is stored as: the best one kconv reads preselected, the ones it does not read shown but refused. */
internal fun MainActivity.chooseSrc(title: String, srcs: List<ModelSrc.Src>, then: (ModelSrc.Src) -> Unit) {
    val all = srcs; val srcs = ModelSrc.offered(all)
    if (srcs.size == 1) { then(srcs[0]); return }
    val best = srcs.firstOrNull(); var at = 0
    fun size(mb: Int) = if (mb >= 1024) "%.1f GB".format(mb / 1024.0) else "$mb MB"
    val items = srcs.map { s ->
        val sub = "${ModelSrc.label(s.fmt)}  ·  ${size(s.mb)}" + if (s === best) "  ·  " + tr(R.string.recommended) else ""
        android.text.SpannableString(s.path.substringAfterLast('/') + "\n" + sub).apply {
            val a = length - sub.length; setSpan(android.text.style.RelativeSizeSpan(0.82f), a, length, 0)
            setSpan(android.text.style.ForegroundColorSpan(0xFF9AA0A6.toInt()), a, length, 0) }
    }.toTypedArray<CharSequence>()
    if (best == null) { val srcs = all; status(tr(R.string.none_of_its_files_is_in)
        .format(srcs.joinToString { ModelSrc.label(it.fmt) })); return }
    AlertDialog.Builder(this).setTitle(title).setSingleChoiceItems(items, at) { _, i -> at = i }
        .setPositiveButton("OK") { _, _ -> val c = srcs[at]
            if (ModelSrc.readable(c.fmt)) then(c) else status(tr(R.string.s_is_not_read_by_the).format(ModelSrc.label(c.fmt))) }
        .setNegativeButton(tr(R.string.cancel), null).show()
}

/** The format a model brought in is converted to (fp16 / Q8 / Q4), asked once the file's own format [src] is known ([ModelSrc]):
 * nothing finer than the source is offered. */
internal fun MainActivity.pickFormat(src: String, then: () -> Unit) {
    val all = listOf("f16", "q8", "q4"); val fmts = ModelSrc.outs(src)
    val labels0 = arrayOf(tr(R.string.fp16_as_made),
        tr(R.string.q8_half_the_size),
        tr(R.string.q4_a_quarter_lower_quality))
    val labels = fmts.map { labels0[all.indexOf(it)] }.toTypedArray()
    val title = tr(R.string.convert_to) + if (src.isEmpty()) "" else "  ·  " + tr(R.string.source_s).format(ModelSrc.label(src))
    AlertDialog.Builder(this).setTitle(title)
        .setSingleChoiceItems(labels, fmts.indexOf(Advanced.convertFormat).coerceAtLeast(0)) { d, i -> Advanced.convertFormat = fmts[i]; d.dismiss(); then() }
        .setNegativeButton(tr(R.string.cancel), null).show()
}

/** Brings one LoRA file in, on the calling thread: [fetch] puts it in the cache, then it is converted and selected. */
internal fun MainActivity.importLora(name: String, page: String? = null, words: List<String> = emptyList(), fetch: (File) -> Unit) {
    val tmp = File(cacheDir, "lora_in.safetensors")
    var made: File? = null   // the LoRA's folder once conversion starts: a failure removes it, or it showed up as a LoRA
    try {
        fetch(tmp); status(tr(R.string.converting_the_lora))
        // the file says which model it is for; the model on screen only decides when the file does not
        val model = LoraImport.detect(tmp) ?: loraModel
        val out = File(loraRoot(model), name); out.deleteRecursively(); made = out
        val r = when (model) {
            "sdxl" -> LoraImport.convertSdxl(tmp, out)
            "anima" -> LoraImport.convertAnima(tmp, out)
            "moss" -> LoraImport.convertMoss(tmp, out)
            "klein", "klein9" -> LoraImport.convertKlein(tmp, out)
            else -> LoraImport.convert(tmp, out)   // Krea 2
        }
        // Anima's modulation Linears count as in only with a bundle that computes the modulations on the CPU
        val mods = r.adaln > 0 && Adaln.present(File(models.root, "anima_bundle"))
        val used = r.pairs + (if (mods) r.adaln else 0); val left = r.skipped + (if (mods) 0 else r.adaln)
        Logs.add("lora converted for $model: $used layers, $left left out${if (r.adaln > 0) " (${r.adaln} modulation layers ${if (mods) "in" else "out"})" else ""}")
        // the trigger words: the file's own (ss_tag_frequency) when it has them, else the ones its Civitai page lists
        if (words.isNotEmpty() && loraInfo(out).optJSONArray("triggers") == null) runCatching { File(out, "info.json").writeText(loraInfo(out).put("triggers", org.json.JSONArray(words)).put("triggers_from", "civitai").toString()) }
        Logs.add("lora triggers: " + (loraInfo(out).optJSONArray("triggers")?.let { t -> (0 until t.length()).joinToString(", ") { t.optString(it) } } ?: "none") + if (words.isNotEmpty()) " (Civitai: ${words.joinToString(", ")})" else "")
        setLoraList(model, listOf(name to 100))
        status(ModelKind.family(model) + " · " + tr(R.string.lora_added, name, used) + (if (left > 0) tr(R.string.lora_left_out, left) else ""))
    } catch (e: Throwable) {
        made?.deleteRecursively()
        Logs.add("lora convert failed: ${e.message}")
        status(tr(R.string.failed) + when (e.message) {
            LoraImport.UNSUPPORTED -> tr(R.string.lora_plain_only)
            LoraBrowse.LOGIN -> tr(R.string.lora_needs_login)
            else -> e.message ?: e.javaClass.simpleName })
        // about half of the popular files are like this: the page is one tap away, and the browser's download goes in with "Add a LoRA file"
        if (e.message == LoraBrowse.LOGIN) page?.let { url -> runOnUiThread {
            AlertDialog.Builder(this).setMessage(tr(R.string.lora_needs_login))
                .setPositiveButton(tr(R.string.open_the_page)) { _, _ -> runCatching { startActivity(Intent(Intent.ACTION_VIEW, android.net.Uri.parse(url))) } }
                .setNegativeButton(tr(R.string.cancel), null).show() } }
    } finally { tmp.delete(); runOnUiThread { loraLabel(); loraRefill?.invoke() } }
}

/** the models with fine-tunes to find online, the installed ones */
private fun MainActivity.browseKinds() = listOf("anima", "krea", "moss", "klein", "klein9", "sdxl").map { it to ModelKind.family(it) }.filter { k -> models.installed(chip!!).any { it.id == k.first } || (k.first == "moss" && MossGen.present(models.root)) || (k.first == "klein" && KleinGen.present(models.root)) || (k.first == "klein9" && KleinGen.present(models.root, "klein9_bundle")) }

/**
 * LoRAs for the model on screen from Civitai's public listing: a search box, three sort orders, and a tap to download
 * and convert. Entries the site marks as adult are not requested.
 */
internal fun MainActivity.loraBrowser(forModels: Boolean = false) {
    // the 18+ question is Civitai's alone: before it is answered the browser opens on Hugging Face, and it is asked when Civitai is
    // picked (it closed the whole browser, Hugging Face included)
    lateinit var srcRow: LinearLayout
    fun civitaiGate(then: () -> Unit) {
        if (prefs.getBoolean("civitai_18", false)) { then(); return }
        AlertDialog.Builder(this).setTitle("Civitai")
            .setMessage(tr(R.string.civitai_is_a_separate_website_its))
            .setPositiveButton(tr(R.string.i_am_18_or_older)) { _, _ ->
                prefs.edit().putBoolean("civitai_18", true).apply(); then() }
            .setNegativeButton(tr(R.string.cancel)) { _, _ -> srcRow.getChildAt(1)?.performClick() }
            .setOnCancelListener { srcRow.getChildAt(1)?.performClick() }.show()
    }
    val kinds = browseKinds()
    // the model browsed: the one on screen when it can be, chosen at the top of the models' list
    var model = if (forModels) kinds.firstOrNull { it.first == loraModel }?.first ?: kinds.first().first else loraModel
    var dlg: AlertDialog? = null; var hf = !prefs.getBoolean("civitai_18", false)
    val list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    val info = note(""); var cursor: String? = null; var sort = 0; var asked = 0
    val queryEt = field(EditText(this).apply { hint = tr(R.string.search); setSingleLine(); imeOptions = android.view.inputmethod.EditorInfo.IME_ACTION_SEARCH })
    lateinit var more: Button
    fun pick(h: LoraBrowse.Hit) {
        if (running) return
        LoraBrowse.models = forModels
        // a Hugging Face repository: its .safetensors files are listed now, and one is chosen when there are several
        if (h.repo != null && h.url.isEmpty()) {
            info.text = tr(R.string.searching)
            thread {
                LoraBrowse.models = forModels
                val fs = runCatching { LoraBrowse.hfFiles(h.repo) }.getOrElse { emptyList() }
                runOnUiThread {
                    info.text = ""
                    when {
                        fs.isEmpty() -> info.text = tr(R.string.no_usable_safetensors_file_in_s).format(h.repo)
                        forModels -> chooseSrc(h.name, fs) { c -> LoraBrowse.hfChoose(h, c.path, c.mb, c.fmt); pick(h) }
                        fs.size == 1 -> { LoraBrowse.hfChoose(h, fs[0].path, fs[0].mb); pick(h) }
                        else -> AlertDialog.Builder(this).setTitle(h.name).setItems(fs.map { "${it.path}  ·  ${it.mb} MB" }.toTypedArray()) { _, i ->
                            LoraBrowse.hfChoose(h, fs[i].path, fs[i].mb); pick(h) }.show()
                    }
                }
            }
            return
        }
        // a Civitai version offering several files: the best kconv reads is preselected, the others can be taken instead
        if (forModels && h.repo == null && h.srcs.isNotEmpty() && !h.srcAsked) {
            info.text = tr(R.string.searching)
            thread {
                val ps = LoraBrowse.probe(h.srcs)
                runOnUiThread { info.text = ""; chooseSrc(h.name, ps) { c -> h.srcAsked = true; h.file = c.path; h.url = c.url; h.sizeMb = c.mb; h.fmt = c.fmt; pick(h) } }
            }
            return
        }
        if (forModels) {
            AlertDialog.Builder(this).setTitle(h.name)
                .setMessage(tr(R.string.download_s_d_mb_by_s).format(h.file, h.sizeMb, h.creator) + "\n" + h.page)
                .setPositiveButton(tr(R.string.download_it)) { _, _ -> pickFormat(h.fmt) { dlg?.dismiss(); downloadModel(h) } }
                .setNegativeButton(tr(R.string.cancel), null).show()
            return
        }
        AlertDialog.Builder(this).setTitle(h.name)
            .setMessage(tr(R.string.lora_download_ask, h.sizeMb, h.creator) + "\n" + h.page)
            .setPositiveButton(tr(R.string.download_it)) { _, _ ->
                dlg?.dismiss(); Logs.add("lora download from ${if (h.repo != null) "hugging face" else "civitai"}, ${h.sizeMb}MB")
                val name = h.name.replace(Regex("[^\\p{L}\\p{N} ._-]"), "_").trim().take(40).ifEmpty { "lora" }
                thread { importLora(name, h.page, h.words) { tmp -> LoraBrowse.download(h, tmp, { false }) { got, total -> status(tr(R.string.lora_downloading, got shr 20, maxOf(total shr 20, h.sizeMb.toLong()))) } } }
            }.setNegativeButton(tr(R.string.cancel), null).show()
    }
    fun load(fresh: Boolean) {
        if (fresh) { cursor = null; list.removeAllViews() }
        val mine = ++asked; val q = queryEt.text.toString(); val cur = cursor
        info.text = tr(R.string.searching); more.visibility = View.GONE
        thread {
            LoraBrowse.models = forModels
            val r = runCatching { if (hf || model == "moss" || model == "klein" || model == "klein9") LoraBrowse.searchHf(model, sort, q) to null else LoraBrowse.search(model, sort, q, cur) }
            runOnUiThread {
                if (mine != asked) return@runOnUiThread                       // a newer search has been started since
                r.onFailure { info.text = tr(R.string.failed) + (it.message ?: it.javaClass.simpleName) }
                r.onSuccess { (hits, next) ->
                    val rows = hits.map { h ->
                        TextView(this).apply {
                            text = h.name + "\n" + if (h.repo != null) "${h.creator}  ·  ${h.downloads}" else tr(R.string.lora_hit_line, h.downloads, h.sizeMb, h.creator); textSize = 14f
                            setPadding(px(14), px(11), px(14), px(11)); setTextColor(INK); background = round(FIELD, 12); setOnClickListener { pick(h) }
                        }.also { list.addView(it, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = px(6) }) }
                    }
                    // which ones the author put behind a login: the listing does not say, so each download address is asked
                    // (no file is fetched). With the user's API key set, those download too and are not marked.
                    thread {
                        val pool = java.util.concurrent.Executors.newFixedThreadPool(4)
                        hits.zip(rows).forEach { (h, row) -> pool.execute {
                            if (mine != asked || h.repo != null) return@execute
                            if (LoraBrowse.needsLogin(h) == true) runOnUiThread {
                                row.text = h.name + "\n" + tr(R.string.lora_hit_line, h.downloads, h.sizeMb, h.creator) + "  ·  " + tr(R.string.lora_login_needed)
                                row.setTextColor(MUTED)
                            }
                        } }
                        pool.shutdown()
                    }
                    cursor = next; more.visibility = if (next != null) View.VISIBLE else View.GONE
                    info.text = if (list.childCount == 0) tr(R.string.nothing_found) else ""
                }
            }
        }
    }
    more = button(tr(R.string.more)) { load(false) }
    val (sortRow, _) = chips(listOf(tr(R.string.sort_downloads), tr(R.string.sort_rating), tr(R.string.sort_newest)), 0, quiet = true) { sort = it; load(true) }
    queryEt.setOnEditorActionListener { _, _, _ -> load(true); true }
    lateinit var keyBtn: Button
    fun keyLabel() { keyBtn.text = tr(if (LoraBrowse.apiKey.isNullOrBlank()) R.string.civitai_key_off else R.string.civitai_key_on) }
    keyBtn = button("") {
        val et = field(EditText(this).apply { hint = tr(R.string.civitai_key_hint); setSingleLine()
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD; setText(LoraBrowse.apiKey ?: "") })
        AlertDialog.Builder(this).setTitle(tr(R.string.civitai_key)).setView(column(note(tr(R.string.civitai_key_help)), et))
            .setPositiveButton("OK") { _, _ ->
                val k = et.text.toString().trim().ifEmpty { null }
                LoraBrowse.apiKey = k; prefs.edit().apply { if (k == null) remove("civitai_key") else putString("civitai_key", k) }.apply()
                Logs.add("civitai key ${if (k == null) "removed" else "set"}"); keyLabel(); load(true)
            }.setNegativeButton(tr(R.string.cancel), null).show()
    }
    keyLabel()
    srcRow = chips(listOf("Civitai", "Hugging Face"), if (hf) 1 else 0, quiet = true) { i ->
        if (i == 1) { hf = true; keyBtn.visibility = View.GONE; load(true) }
        else civitaiGate { hf = false; keyBtn.visibility = View.VISIBLE; load(true) } }.first
    fun titleOf() = (kinds.firstOrNull { it.first == model }?.second ?: "Krea 2") + " · " + if (forModels) tr(R.string.models_2) else "LoRA"
    val titleTv = text(titleOf(), 20f, true)
    // MOSS-Image and Klein: Hugging Face alone
    fun sourceRows() { val civ = model != "moss" && model != "klein" && model != "klein9"; srcRow.visibility = if (civ) View.VISIBLE else View.GONE; keyBtn.visibility = if (civ && !hf) View.VISIBLE else View.GONE }
    sourceRows()
    val (kindRow, _) = chips(kinds.map { it.second }, kinds.indexOfFirst { it.first == model }.coerceAtLeast(0), quiet = true, perRow = 3) { model = kinds[it].first; titleTv.text = titleOf(); sourceRows(); load(true) }
    if (!forModels) kindRow.visibility = View.GONE
    val help = if (forModels) tr(R.string.fine_tunes_of_this_model_the) else tr(R.string.lora_browse_help)
    val v = column(titleTv, note(help), kindRow, srcRow, keyBtn, queryEt, sortRow, list, info, more).apply { setBackgroundColor(BG) }
    dlg = AlertDialog.Builder(this).setView(ScrollView(this).apply { addView(v) }).setPositiveButton("OK", null).setOnDismissListener { asked++ }.show()
    load(true)
}
