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

/* MainActivity: making pictures and videos, and what happens to them after (moved out of MainActivity.kt unchanged) */

/** paint the inpaint mask over the start picture (at most 1024 a side; the mask is placed in the frame with the picture) */
internal fun MainActivity.maskDialog() {
    val src = android.graphics.BitmapFactory.decodeFile(imgStart().absolutePath) ?: return
    val k = minOf(1f, 1024f / maxOf(src.width, src.height))
    val pic = if (k < 1f) android.graphics.Bitmap.createScaledBitmap(src, Math.round(src.width * k), Math.round(src.height * k), true).also { src.recycle() } else src
    val old = imgMask().takeIf { it.isFile }?.let { android.graphics.BitmapFactory.decodeFile(it.absolutePath) }
    val mask = android.graphics.Bitmap.createBitmap(pic.width, pic.height, android.graphics.Bitmap.Config.ARGB_8888).also { m ->
        val c = android.graphics.Canvas(m); c.drawColor(android.graphics.Color.BLACK)
        old?.let { c.drawBitmap(it, null, android.graphics.Rect(0, 0, m.width, m.height), null); it.recycle() } }
    val mv = MaskView(this, pic, mask).apply { brush = maxOf(pic.width, pic.height) / 16f }
    val sizes = listOf(32f, 16f, 8f)
    lateinit var sizeB: Button; lateinit var eraseB: Button
    fun labels() { sizeB.text = tr(R.string.brush_s).format(
        listOf(tr(R.string.small), tr(R.string.medium), tr(R.string.large))[sizes.indexOfFirst { maxOf(pic.width, pic.height) / it == mv.brush }.coerceAtLeast(0)])
        eraseB.text = if (mv.erase) tr(R.string.eraser) else tr(R.string.paint) }
    sizeB = button("") { val i = sizes.indexOfFirst { maxOf(pic.width, pic.height) / it == mv.brush }; mv.brush = maxOf(pic.width, pic.height) / sizes[(i + 1) % sizes.size]; labels() }
    eraseB = button("") { mv.erase = !mv.erase; labels() }
    labels()
    val box = column(note(tr(R.string.paint_over_what_to_redraw)), mv,
        line(sizeB, eraseB, button(tr(R.string.clear)) { mv.clear() }))
    val dlg = AlertDialog.Builder(this).setView(box)
        .setPositiveButton("OK") { _, _ ->
            val px = IntArray(mask.width * mask.height).also { mask.getPixels(it, 0, mask.width, 0, 0, mask.width, mask.height) }
            if (px.none { android.graphics.Color.red(it) > 8 }) imgMask().delete()
            else imgMask().outputStream().use { mask.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
            route() }
        .setNeutralButton(tr(R.string.no_mask)) { _, _ -> imgMask().delete(); route() }
        .setNegativeButton(tr(R.string.cancel), null).show()
    dlg.window?.setBackgroundDrawable(round(CARD, 20))
    dlg.getButton(AlertDialog.BUTTON_POSITIVE)?.apply { setTextColor(ACCENT); setTypeface(typeface, Typeface.BOLD) }
}

/** The picture size of a ratio for a model (the preset a ratio chip fills the size sliders with): sides in 64s, at least 256,
 *  the area at most the model's largest (Krea: its bundle's 4096 image tokens, 1024²; the others 2048²). Fixed: Anima (its RoPE
 *  tables), SDXL (the width 1024, the engine's last level), YuE. */
internal fun MainActivity.sizeOf(model: String, ratio: Int): Pair<Int, Int> {
    val (_, w0, h0) = RATIOS[ratio]
    if (!sizeScalable(model)) return w0 to h0
    return fitArea(w0, h0, sizeCap(model)).let { if (model == "sdxl") sdxlFit(it) else it }
}
/** The largest area a model takes (Krea: its bundle's 4096 image tokens; the others 2048x2048). */
internal fun MainActivity.sizeCap(model: String) = when (model) { "krea" -> 1024.0 * 1024
    "sdxl" -> 1024.0 * 1536      // its activation buffers (and 1.5 MP is past what SDXL was trained for)
    "anima" -> AnimaGen.budget(File(models.root, "anima_bundle")) * 256.0     // its blocks' token rows (16x16 pixels each)
    else -> 2048.0 * 2048 }
/** Sides in 64s, at least 256, scaled down together (the ratio kept) when the area is over the cap. */
internal fun fitArea(w: Int, h: Int, cap: Double): Pair<Int, Int> {
    val s = if (w.toDouble() * h > cap) Math.sqrt(cap / (w.toDouble() * h)) else 1.0
    fun side(v: Int) = ((v * s / 64).toInt() * 64).coerceAtLeast(256)
    return side(w) to side(h)
}
/** The size the screen asks for: the two size sliders where the model takes any size (free ratio, area at most the model's cap), else the ratio's. */
internal fun MainActivity.curSize(model: String, ratio: Int): Pair<Int, Int> =
    (if (!sizeScalable(model) || sizeW <= 0 || sizeH <= 0) sizeOf(model, ratio) else fitArea(sizeW, sizeH, sizeCap(model))).let { if (model == "sdxl") sdxlFit(it) else it }
/** SDXL's engine pads each level's latent width to 32 (the picture's width to 256 at the top) and its attention holds at most 6144 tokens at
 *  the middle level (1024 x 1536's; 1216 x 832 padded to 128s broke on an 8 Gen 2): the height cut to fit */
internal fun sdxlFit(s: Pair<Int, Int>): Pair<Int, Int> {
    val w0 = (s.first / 8 + 31) / 32 * 32; val w1 = (w0 / 2 + 31) / 32 * 32
    val h1max = 6144 / w1; val hmax = h1max * 16 / 64 * 64
    return s.first to minOf(s.second, hmax).coerceAtLeast(256)
}
internal fun sizeScalable(model: String) = model != "yue"

internal fun MainActivity.generate(jobs: List<Job>) {
    if (uiPreview) { status(tr(R.string.ui_preview_this_phone_cannot_run)); return }
    if (models.installed(chip!!).none { it.isModel }) { status(tr(R.string.no_model_on_this_phone_yet_2)); return }
    val count = jobs.size
    running = true; stopRequested = false; graphView?.locked = true; Engine.abort(false)
    stopSong(); songBtn?.visibility = View.GONE
    goBtn.text = tr(R.string.stop); extraBtns.forEach { it.isEnabled = false }
    Logs.run("generate ${jobs.size} picture(s)"); Logs.add("memory ${availMb()}MB available")
    // which build made it and on what: a report then says by itself which app and engine libraries ran
    Logs.add("app ${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE}) ${BuildConfig.BUILD_TYPE} · installed " +
        java.text.SimpleDateFormat("MM-dd HH:mm", java.util.Locale.US).format(java.util.Date(runCatching { packageManager.getPackageInfo(packageName, 0).lastUpdateTime }.getOrDefault(0L))) +
        " · chip ${chip?.tag} ${chip?.arch ?: "-"} · ${android.os.Build.MODEL}")
    Logs.armDiag(models.root)
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); GenService.start(this)
    bar.progress = 0; bar.visibility = View.VISIBLE; lockTv.visibility = View.VISIBLE; askTv.visibility = View.GONE
    status(tr(R.string.preparing))
    // the DSP refusing an open is waited for a few times (Engine.dspRetry): the wait on the screen, a stop ends it
    Engine.onWait = { sec -> if (sec > 0 && !stopRequested) status(tr(R.string.waiting_for_the_npu, sec)); !stopRequested }
    Engine.onBake = { d, n ->
        runOnUiThread { grid.visibility = if (d < n) View.VISIBLE else View.GONE; grid.set(d, n) }
        if (d < n) status(tr(R.string.merging_the_lora_into_the_weights).format(d + 1, n))
    }
    thread {
        var made = 0
        // an Anima generator serves the pictures of a run for as long as they share size and LoRA: see AnimaGen.close
        var agKept: AnimaGen? = null; var agKey = ""
        try {
            val root = models.root
            val qtok = QwenTok(File(root, "tok"))
            val trDone = HashMap<String, String>(); trSource = null; trText = null
            for (k in 0 until count) {
                if (stopRequested) break
                // a picture costs several percent of battery, and a phone charging from a computer's USB port loses
                // more than it gains: a run of pictures stops itself before the phone does
                val bat = getSystemService(android.os.BatteryManager::class.java).getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)
                if (bat in 1..14) { Logs.add("stopped: battery $bat%")
                    status(tr(R.string.battery_at_stopped_charge_the, bat)); break }
                val (prompt0, steps, seedIn, w, h) = jobs[k]
                // a prompt in Korean, Japanese, Chinese or Russian goes to the model in English (Translator: PLaMo), once a run
                val prompt = if (jobs[k].model == "yue" || Translator.lang(prompt0) == null) prompt0 else trDone.getOrPut(prompt0) {
                    status(tr(R.string.translating_the_prompt_plamo))
                    val en = Translator.translate(this, root, prompt0, Logs::add)?.also { trSource = prompt0; trText = it }
                    // the tag models (Anima, SDXL): the translation's short items snapped to real Danbooru tags (TagSnap: EmbeddingGemma)
                    if (en != null && (jobs[k].model == "anima" || jobs[k].model == "sdxl")) TagSnap.snap(root, en, Logs::add) else en ?: prompt0 }
                if (prompt !== prompt0) Safety.refuse(prompt, jobs[k].model)?.let { Logs.add("translated prompt refused by the filter"); status(it); break }
                val seed = seedIn ?: (System.nanoTime() and 0x7FFFFFFFFFFFL)
                lastSeed = seed; prefs.edit().putLong("last_seed", seed).apply()
                val tRun = System.currentTimeMillis()
                Logs.add("gen ${jobs[k].model} ${w}x$h steps $steps seed $seed (${k + 1}/$count) · weights ${weightFmt(jobs[k].model)} · encoder ${models.teFmt(jobs[k].model)}" +
                    (Advanced.custom(jobs[k].model).takeIf { Advanced.on && it.isNotEmpty() }?.let { " · fine-tune $it" } ?: "")); stepNow = -1; stepsNow = steps
                if (jobs[k].model == "yue") { if (makeSong(root, prompt, seed, k, count, tRun)) made++; continue }
                if (jobs[k].model == "klein" || jobs[k].model == "klein9") { if (makeKlein(root, qtok, prompt, steps, seed, w, h, k, count, tRun, jobs[k].loras, jobs[k].model)) made++; continue }
                if (jobs[k].model == "sdxl") { if (makeSdxl(root, prompt, steps, seed, w, h, k, count, tRun, jobs[k].loras)) made++; continue }
                if (jobs[k].model == "moss") { if (makeMoss(root, prompt, steps, seed, w, h, k, count, tRun, jobs[k].loras)) made++; continue }
                val anima = jobs[k].model == "anima"
                val sel = jobs[k].loras.mapNotNull { (n, p) -> File(root, "lora/${jobs[k].model}/$n").takeIf { it.isDirectory }?.let { it to p } }
                // one goes in as it is; several are first put side by side in one folder, their strengths folded in
                val lora = when {
                    sel.isEmpty() -> null
                    sel.size == 1 -> sel[0].first
                    else -> {
                        status(tr(R.string.preparing_the_lora))
                        val mdir = File(root, "lora/${jobs[k].model}/.merged")
                        val w0 = LoraImport.mergeLoras(sel.map { it.first to it.second / 100f }, mdir, anima, if (anima) LoraImport.ANIMA_RANK else 512)
                        Logs.add("loras merged: ${sel.size} -> ${if (w0 < 0) "cached" else "$w0 matrices"}")
                        mdir.takeIf { it.isDirectory }
                    }
                }
                val pct = if (sel.size == 1) sel[0].second / 100f else 1f
                // the engines bake the chosen ones exactly, one after another, from their own folders
                val loraSet = if (sel.isEmpty()) null else LoraSet.write(File(root, "lora/${jobs[k].model}/.set.txt"), sel.map { it.first to it.second / 100f })
                if (lora != null) Logs.add("lora " + sel.joinToString(", ") { "${it.first.name} ${it.second}" })
                val kg = if (anima) null else KreaGen(root, models.bundle, { Logs.add(it) }, { status(it) }, { stopRequested }, w, h, lora, pct, loraSet)
                val key = "$w x $h " + sel.joinToString("+") { "${it.first.name}:${it.second}" }
                if (!anima || key != agKey) { agKept?.close(); agKept = null }
                val ag = if (!anima) null else agKept ?: AnimaGen(root, File(root, "anima_bundle"), { Logs.add(it) }, { status(it) }, { stopRequested }, w, h,
                    lora?.let { status(tr(R.string.preparing_the_lora)); LoraImport.activateAnima(it, File(root, "lora/anima/.active"), pct) }, loraSet = loraSet).also { agKept = it; agKey = key }
                val t0 = System.currentTimeMillis(); var t1 = t0
                val preview = { x0: FloatArray -> runCatching { KreaGen.preview(x0, h / 8, w / 8) }.getOrNull()?.let { b -> runOnUiThread { img.setImageBitmap(b) }; graphView?.picture = b }; Unit }
                // the bar and the status move with every block, not once a
                // step: a Krea step is over a minute
                var run0 = 0          // img2img starts part way: the bar counts the steps that run
                var lastDone = 0      // a percent that falls back is logged with what reported it, to find where it comes from
                val onUnit = { i: Int, u: Int, n: Int ->
                    stepNow = i; stepsNow = steps
                    val done = (i - run0) * n + u; val total = (steps - run0) * n
                    if (done < lastDone) Logs.add("progress fell back: step ${i + 1} unit $u/$n, $done < $lastDone of $total (${Thread.currentThread().name})")
                    lastDone = done
                    // time left from the pace of the pieces themselves: the text encoding before them is a
                    // one-off, and counting it made the first minutes read several minutes too long
                    if (done == 1) { t1 = System.currentTimeMillis(); runOnUiThread { bar.isIndeterminate = false; bar.max = total * 100 } }
                    val left = if (done > 1) (System.currentTimeMillis() - t1) * (total - done) / (done - 1) / 1000 else 0L
                    // something on screen has to move the whole time or it looks hung: the percent and the
                    // seconds change with every piece, the bar with them
                    val line = (if (count > 1) tr(R.string.picture, k + 1, count) else "") +
                        tr(R.string.step_about_left, i + 1, steps, done * 100 / total, hhmm(left))
                    if (!stopRequested) status(line)
                    runOnUiThread { bar.setProgress(done * 100, true) }
                    graphView?.let { it.active = "sampler"; it.progress = done.toFloat() / total; it.locked = true }
                    if (u % 8 == 0 || u == n) GenService.update(this, line, done, total)
                }
                runOnUiThread { bar.progress = 0; bar.isIndeterminate = true }
                // img2img: the start picture's latent, and the step the run starts from (the change asked for, in steps)
                val dn = denoisePct()
                val ready = imgStart().takeIf { it.isFile }?.let { f ->
                    if (!KreaGen.canEncode(root, w, h)) { Logs.add("img2img: no encoder for ${w}x$h, text to image instead"); return@let null }
                    status(tr(R.string.reading_the_start_image))
                    android.graphics.BitmapFactory.decodeFile(f.absolutePath)?.let { b -> Img2Img.prepare(root, b, w, h, imgFit(), imgMask()).also { b.recycle() } }
                }
                val init = ready?.latent; val mask = ready?.mask
                // outpaint: the added area is drawn from noise (started from a blurred fill, Anima read the blobs as new things: a room, a leg)
                val from = if (init == null || (imgFit() && mask != null)) 0 else Math.round(steps * (100 - dn) / 100f).coerceIn(0, steps - 1)
                if (init != null) Logs.add("img2img: change $dn%, from step ${from + 1}/$steps" + (if (mask != null) ", mask ${"%.0f".format(mask.average() * 100)}%" else "") + if (imgFit()) ", fit" else "")
                run0 = from
                val x = if (anima) ag!!.generate(prompt, qtok, steps, seed, preview, onUnit, Advanced.tuning("anima"), init, from, mask) { _, _ -> }
                        else kg!!.generate(prompt, qtok, steps, seed, preview, onUnit, init, from, mask, Advanced.tuning("krea").takeIf { Advanced.on }) { _, _ -> }
                val doneSteps = ag?.doneSteps ?: kg!!.doneSteps
                // a stopped run keeps the denoised estimate once it is past half way; earlier it is still a blur
                val xs = if (!stopRequested) x else (ag?.lastX0 ?: kg?.lastX0)?.takeIf { doneSteps * 2 >= steps }
                if (xs == null) { status(tr(R.string.stopped_nothing_saved_under_half)); break }
                status(tr(R.string.decoding)); graphView?.let { it.active = "vae"; it.progress = -1f }
                val bmp = KreaGen.decode(root, xs, h / 8, w / 8)
                val meta = org.json.JSONObject().apply {
                    put("model", modelName(chip!!, jobs[k].model, weightFmt(jobs[k].model))); put("app", "Kroma"); put("ai_generated", true)
                    put("prompt", prompt); put("seed", seed); put("steps", if (stopRequested) doneSteps else steps); put("width", w); put("height", h)
                    if (lora != null) { put("lora", lora.name); put("lora_strength", pct.toDouble()) }
                    if (init != null) put("denoise", dn / 100.0)
                    if (anima || Advanced.on) Advanced.tuning(if (anima) "anima" else "krea").let { t -> put("sampler", t.sampler); put("scheduler", t.scheduler); put("cfg", t.cfg); put("shift", t.shift)
                        if (t.usesCfg && t.negative.isNotEmpty()) put("negative", t.negative) }
                    Advanced.custom(jobs[k].model).takeIf { Advanced.on && it.isNotEmpty() }?.let { put("finetune", it) }
                }.toString()
                if (!deliver(bmp, prompt, "kroma_$seed", meta, jobs[k].model)) continue
                made++
                Logs.add("done in ${hhmm((System.currentTimeMillis() - tRun) / 1000)}")
                // the first two finished pictures only: asking every time would be a nuisance
                // only the trial chips (8s Gen 3, 8s Gen 4, 7+ Gen 3) are asked: their log is what tells whether they work at all
                if (chip?.animaOnly == true) prefs.getInt("log_asks", 0).let { n -> if (n < 2) { prefs.edit().putInt("log_asks", n + 1).apply(); runOnUiThread { askTv.visibility = View.VISIBLE } } }
                status(tr(R.string.done_in_seed, hhmm((System.currentTimeMillis() - tRun) / 1000), seed) + trNote() + if (pending.isEmpty()) "" else "\n" + tr(R.string.not_saved_yet))
            }
        } catch (e: InterruptedException) {
            status(tr(R.string.stopped))
        } catch (e: Throwable) {
            if (stopRequested) { Logs.add("stopped: ${e.message}"); status(tr(R.string.stopped)); return@thread }
            Logs.add("ERROR ${e.stackTraceToString().take(3000)}")
            status(tr(R.string.error) + (e.message ?: e.javaClass.simpleName) + tr(R.string.nsettings_send_the_log))
        } finally {
            agKept?.close(); Engine.dropBaked(); Engine.vdecDrop(); Logs.diagEnd()
            GenService.stop(this, if (made > 0) tr(if (prefs.getBoolean("auto_save", false)) R.string.picture_s_saved else R.string.picture_s_ready, made) else null)
            runOnUiThread { val then = afterRun; afterRun = null; if (made > 0 && !stopRequested) then?.let { ticker.post { it() } }
                running = false; graphView?.let { if (it.active != "upscale") { it.active = null; it.progress = -1f; it.locked = false } }; goBtn.text = tr(R.string.generate); bar.visibility = View.INVISIBLE; lockTv.visibility = View.GONE; window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                buttons(); showMemory() }
        }
    }
}

/** One FLUX.2 Klein picture (the built-in models are distilled to 4 steps, one pass; a fine-tune's CFG and negative from the extended settings). True when it was saved. */
internal fun MainActivity.makeKlein(root: File, qtok: QwenTok, prompt: String, steps0: Int, seed: Long, w: Int, h: Int, k: Int, count: Int, tRun: Long, loras: List<Pair<String, Int>> = emptyList(), id: String = "klein"): Boolean {
    val b = File(root, if (id == "klein9") "klein9_bundle" else "klein_bundle"); val gh = h / 16; val gw = w / 16; val steps = steps0; var t1 = 0L
    val vae = File(b, "vae").takeIf { it.isDirectory } ?: File(root, "moss_bundle/vae")
    runOnUiThread { bar.progress = 0; bar.isIndeterminate = true }
    status(tr(R.string.preparing))
    val custom = Custom.engineDir(root, id)
    // the extended settings (CFG, negative) only when a fine-tune draws: the built-in models are distilled (cfg 1, one pass, no negative)
    val kt = Advanced.KLEIN_TURBO
    val tune = if (custom != null) Advanced.tuning(id) else Advanced.Tuning(kt.sampler, kt.scheduler, kt.shift.toDouble(), kt.cfg.toDouble(), "")
    val x0 = KleinGen(custom ?: File(b, "engine")) { Logs.add(it) }.use { g ->
        val mine = loras.mapNotNull { (n, p) -> File(root, "lora/$id/$n").takeIf { it.isDirectory }?.let { it to p / 100f } }
        if (mine.isNotEmpty()) { g.loraSet = LoraSet.write(File(b, ".set.txt"), mine); Logs.add("lora " + mine.joinToString(", ") { "${it.first.name} ${(it.second * 100).toInt()}" }) }
        val t0 = System.currentTimeMillis()
        Logs.add("  klein: cfg ${tune.cfg}" + if (tune.usesCfg) ", negative ${tune.negative.length} chars" else "")
        val ids = KleinGen.promptIds(qtok, prompt); val ctx = g.context(File(b, "kte"), ids)
        val negIds = if (tune.usesCfg) KleinGen.promptIds(qtok, tune.negative) else null; val negCtx = negIds?.let { g.context(File(b, "kte"), it) }
        Logs.add("  text ${ids.size} tokens · ${(System.currentTimeMillis() - t0) / 1000.0}s · " + ctx.let { c -> var bad = 0; var mx = 0f; for (v in c) { if (!v.isFinite()) bad++ else if (kotlin.math.abs(v) > mx) mx = kotlin.math.abs(v) }; "ctx max|x| %.2f · NaN/inf $bad of ${c.size}".format(mx) })
        val preview = { x: FloatArray -> runCatching { MossGen.preview(x, gh, gw) }.getOrNull()?.let { p -> runOnUiThread { img.setImageBitmap(p) }; graphView?.picture = p }; Unit }
        val nb = g.blocks; val total = steps * nb
        // a start image is Klein's reference picture (editing: the prompt says what to change)
        // the reference pictures share about a megapixel (their tokens join the picture's in every attention)
        val files = imgRefs()
        if (files.isNotEmpty()) status(tr(R.string.reading_the_reference_pictures))
        val refs = files.mapNotNull { f -> android.graphics.BitmapFactory.decodeFile(f.absolutePath)?.let { bm -> KleinGen.encodeRef(bm, vae, area = 1024 * 1024 / files.size, log = { Logs.add(it) }) } }
        g.generate(ctx, ids.size, gh, gw, seed, steps, { stopRequested }, preview, refs, negCtx, negIds?.size ?: 0, tune.cfg.toFloat()) { i, u ->
            stepNow = i; stepsNow = steps
            val done = i * nb + u
            if (done == 1) { t1 = System.currentTimeMillis(); runOnUiThread { bar.isIndeterminate = false; bar.max = total * 100 } }
            val left = if (done > 1) (System.currentTimeMillis() - t1) * (total - done) / (done - 1) / 1000 else 0L
            val line = (if (count > 1) tr(R.string.picture, k + 1, count) else "") + tr(R.string.step_about_left, i + 1, steps, done * 100 / total, hhmm(left))
            if (!stopRequested) status(line)
            runOnUiThread { bar.setProgress(done * 100, true) }
            if (u % 8 == 0 || u == nb) GenService.update(this, line, done, total)
        }
    }
    if (stopRequested) { status(tr(R.string.stopped)); return false }
    status(tr(R.string.decoding))
    val tVae = System.currentTimeMillis()
    val bmp = MossGen.decode(x0, gh, gw, vae).also { Logs.add("  vae ${System.currentTimeMillis() - tVae}ms") }
    val meta = org.json.JSONObject().apply {
        put("model", modelName(chip!!, id, weightFmt(id))); put("app", "Kroma"); put("ai_generated", true)
        put("prompt", prompt); put("seed", seed); put("steps", steps); put("width", w); put("height", h)
        if (tune.usesCfg) { put("cfg", tune.cfg); if (tune.negative.isNotEmpty()) put("negative", tune.negative) }
        if (custom != null) Advanced.custom(id).takeIf { it.isNotEmpty() }?.let { put("finetune", it) }
    }.toString()
    if (!deliver(bmp, prompt, "kroma_$seed", meta, id)) return false
    Logs.add("done in ${hhmm((System.currentTimeMillis() - tRun) / 1000)}")
    status(tr(R.string.done_in_seed, hhmm((System.currentTimeMillis() - tRun) / 1000), seed) + trNote())
    return true
}

/**
 * One SDXL picture (SdxlGen: sdxl_bundle/): the text on the CPU, the UNet's Euler loop and the VAE on the DSP. Width 1024, the height from the
 * ratio; cfg 7, at least 20 steps (SDXL base: the step slider's Turbo counts are too few); the negative from the extended settings. True when saved.
 */
internal fun MainActivity.makeSdxl(root: File, prompt: String, steps0: Int, seed: Long, w: Int, h0: Int, k: Int, count: Int, tRun: Long, loras: List<Pair<String, Int>> = emptyList()): Boolean {
    val tune = Advanced.tuning("sdxl")      // the defaults (cfg 7) unless the extended settings are on
    val h = h0; val steps = if (steps0 < 15) 25 else steps0; val cfg = tune.cfg.toFloat()   // any size in 64s (the engine pads a width that is not 1024's)
    val neg = tune.negative
    runOnUiThread { bar.progress = 0; bar.isIndeterminate = true }
    status(tr(R.string.preparing))
    val g = SdxlGen(root, { Logs.add(it) }, Custom.engineDir(root, "sdxl"))
    // the runtime LoRAs (LoraImport.buildSdxl): their blobs for this set and strengths, kept in the cache until the set changes
    val sets = loras.mapNotNull { (n, p) -> File(root, "lora/sdxl/$n/sdxl.safetensors").takeIf { it.isFile }?.let { it to p / 100f } }
    if (sets.isEmpty()) Engine.xlLora("") else {
        val ld = File(cacheDir, "sdxl_lora"); val key = File(g.unet, "unet.tab").absolutePath + "|" + sets.joinToString("|") { "${it.first.absolutePath}:${it.second}:${it.first.lastModified()}" }
        val kf = File(ld, "key.txt")
        if (runCatching { kf.readText() }.getOrNull() != key || !File(ld, "lora.w").isFile) {
            status(tr(R.string.converting_the_lora)); val t0 = System.currentTimeMillis(); ld.deleteRecursively()
            val n = LoraImport.buildSdxl(File(g.unet, "unet.tab"), sets, ld); kf.writeText(key)
            Logs.add("sdxl lora: $n Linears from ${sets.size} file(s) · ${System.currentTimeMillis() - t0} ms")
        }
        Engine.xlLora(ld.absolutePath)
    }
    val (ctx, y) = g.encode(prompt, neg, h, w) { stopRequested }
    Logs.add("sdxl ${w}x$h steps $steps cfg $cfg")
    var t1 = 0L
    runOnUiThread { bar.isIndeterminate = false; bar.max = steps * 100 }
    // the bar, the percent and the time left move with every piece of a step (the UNet's runs of layers), not once a step:
    // the pace from the pieces themselves, timed from the first one (the text encoding before it is a one-off)
    // the pieces are uneven (the UNet's runs differ in size, and the step's own work comes between): the time each piece ended in the
    // first whole step is the profile of every later one, so the progress and the time left run evenly (counted evenly, the time left
    // fell inside a step and rose again at the next one)
    var prof: FloatArray? = null; var tStep0 = 0L; var stepSeen = -1; val firstEnds = HashMap<Int, Long>(); var pos1 = 0f
    val onUnit = { s: Int, d: Int, n: Int ->
        stepNow = s; stepsNow = steps
        val now = System.currentTimeMillis()
        if (s != stepSeen) {
            if (stepSeen == 1 && prof == null && firstEnds.size >= n / 2) {     // step 1 (the second) whole: its pieces' share of the step
                val span = (now - tStep0).coerceAtLeast(1L).toFloat(); val p = FloatArray(n + 1); var last = 0f
                for (i in 1..n) { last = maxOf(last, firstEnds[i]?.let { (it - tStep0) / span } ?: last); p[i] = last }; p[n] = 1f; prof = p
            }
            stepSeen = s; tStep0 = now; firstEnds.clear()
        }
        if (s == 1) firstEnds[d] = now
        val frac = prof?.let { p -> p[d.coerceIn(0, n)] } ?: (d.toFloat() / n)
        val pos = s + frac; val total = steps * n
        if (t1 == 0L && pos > 0f) { t1 = now; pos1 = pos }      // timed from the first piece: the text encoding before it is a one-off
        val left = if (t1 > 0 && pos - pos1 > 0.05f) ((now - t1) / (pos - pos1) * (steps - pos) / 1000).toLong() else 0L
        val done = (pos * n).toInt().coerceIn(0, total)
        val line = (if (count > 1) tr(R.string.picture, k + 1, count) else "") + tr(R.string.step_about_left, minOf(s + 1, steps), steps, done * 100 / total, hhmm(left))
        if (!stopRequested) status(line)
        runOnUiThread { bar.max = total * 100; bar.setProgress(done * 100, true) }
        if (d % 4 == 0 || d == n) GenService.update(this, line, done, total)
    }
    val z = g.generate(ctx, y, h, seed, steps, cfg, w, onUnit) { _, _ -> stopRequested }
    if (z == null || stopRequested) { status(tr(R.string.stopped)); return false }
    status(tr(R.string.decoding))
    val tVae = System.currentTimeMillis()
    val bmp = g.decode(z, h, w).also { Logs.add("  vae ${System.currentTimeMillis() - tVae}ms") }
    val meta = org.json.JSONObject().apply {
        put("model", modelName(chip!!, "sdxl", weightFmt("sdxl"))); put("app", "Kroma"); put("ai_generated", true)
        put("prompt", prompt); put("seed", seed); put("steps", steps); put("cfg", cfg); put("width", w); put("height", h); if (neg.isNotEmpty()) put("negative", neg)
    }.toString()
    if (!deliver(bmp, prompt, "kroma_$seed", meta, "sdxl")) return false
    Logs.add("done in ${hhmm((System.currentTimeMillis() - tRun) / 1000)}")
    status(tr(R.string.done_in_seed, hhmm((System.currentTimeMillis() - tRun) / 1000), seed) + trNote())
    return true
}

/**
 * One MOSS-Image picture: moss_bundle/ holds engine/ (kconv's blocks; a converted fine-tune's in its place when one is chosen),
 * qte/ (the text encoder for the DSP), tok/, turbo8/ and turbo4/ (the turbo LoRAs), vae/. True when it was saved.
 */
internal fun MainActivity.makeMoss(root: File, prompt: String, steps0: Int, seed: Long, w: Int, h: Int, k: Int, count: Int, tRun: Long, loras: List<Pair<String, Int>> = emptyList()): Boolean {
    val b = File(root, "moss_bundle"); val eng = Custom.engineDir(root, "moss") ?: File(b, "engine")
    val gh = h / 16; val gw = w / 16; var t1 = 0L
    runOnUiThread { bar.progress = 0; bar.isIndeterminate = true }
    status(tr(R.string.preparing))
    // HaDeS-MOSS-Turbo (the default) or, chosen in extended mode, any shared sampler on the model as released
    val tune0 = Advanced.tuning("moss"); val steps = steps0
    // HaDeS-MOSS-Turbo: its LoRA (moss_bundle/turbo8 or turbo4) and its fixed settings, euler / simple / shift 5 / CFG 1 (no negative)
    val turbo = tune0.sampler.startsWith("turbo")
    val turboDir = File(b, if (tune0.sampler == "turbo4") "turbo4" else "turbo8")
    val tune = if (turbo) Advanced.Tuning("euler", "simple", 5.0, 1.0, "") else tune0
    Logs.add("moss: ${if (turbo) turboDir.name + " · " else ""}${tune.sampler}/${tune.scheduler}, shift ${tune.shift}, cfg ${tune.cfg}")
    val x0 = MossGen(eng) { Logs.add(it) }.use { g ->
        // the turbo's pair first, the user's own after it
        val mine = loras.mapNotNull { (n, p) -> File(root, "lora/moss/$n").takeIf { it.isDirectory }?.let { it to p / 100f } }
        val set = (if (turbo && turboDir.isDirectory) listOf(turboDir to 1f) else emptyList()) + mine
        if (set.isNotEmpty()) g.loraSet = LoraSet.write(File(b, ".set.txt"), set)
        if (mine.isNotEmpty()) Logs.add("lora " + mine.joinToString(", ") { "${it.first.name} ${(it.second * 100).toInt()}" })
        val preview = { x: FloatArray -> runCatching { MossGen.preview(x, gh, gw) }.getOrNull()?.let { p -> runOnUiThread { img.setImageBitmap(p) }; graphView?.picture = p }; Unit }
        // onStep gives (step, blocks done in it): the bar moves with every block
        val nb = 42; val total = steps * nb
        val onStep = { i: Int, u: Int ->
            stepNow = i; stepsNow = steps
            val done = i * nb + u
            if (done == 1) { t1 = System.currentTimeMillis(); runOnUiThread { bar.isIndeterminate = false; bar.max = total * 100 } }
            val left = if (done > 1) (System.currentTimeMillis() - t1) * (total - done) / (done - 1) / 1000 else 0L
            val line = (if (count > 1) tr(R.string.picture, k + 1, count) else "") + tr(R.string.step_about_left, i + 1, steps, done * 100 / total, hhmm(left))
            if (!stopRequested) status(line)
            runOnUiThread { bar.setProgress(done * 100, true) }
            if (u % 8 == 0 || u == nb) GenService.update(this, line, done, total)
        }
        val tok = File(b, "tok"); val qte = File(b, "qte")
        val (hid, t) = MossText.hidden(qte, MossText.ids(tok, prompt)) { Logs.add(it) }
        val neg = if (tune.usesCfg) MossText.hidden(qte, MossText.ids(tok, tune.negative)) { Logs.add(it) } else null
        g.generateFlow(hid, t, neg?.first, neg?.second ?: 0, gh, gw, seed, Sched.sigmas(steps, tune.shift, tune.scheduler), tune.sampler,
            tune.cfg.toFloat(), { stopRequested }, preview, onStep)
    }
    if (stopRequested) { status(tr(R.string.stopped)); return false }
    status(tr(R.string.decoding))
    val tVae = System.currentTimeMillis()
    val bmp = MossGen.decode(x0, gh, gw, File(b, "vae")).also { Logs.add("  vae ${System.currentTimeMillis() - tVae}ms") }
    val meta = org.json.JSONObject().apply {
        put("model", modelName(chip!!, "moss", weightFmt("moss"))); put("app", "Kroma"); put("ai_generated", true)
        put("prompt", prompt); put("seed", seed); put("steps", steps); put("width", w); put("height", h)
        Advanced.custom("moss").takeIf { Advanced.on && it.isNotEmpty() }?.let { put("finetune", it) }
    }.toString()
    if (!deliver(bmp, prompt, "kroma_$seed", meta, "moss")) return false
    Logs.add("done in ${hhmm((System.currentTimeMillis() - tRun) / 1000)}")
    status(tr(R.string.done_in_seed, hhmm((System.currentTimeMillis() - tRun) / 1000), seed) + trNote())
    return true
}

/** One YuE2 song (Yue.make), saved to Music/Kroma and played. The bar runs over the four parts by their usual share of the time. */
internal fun MainActivity.makeSong(root: File, prompt: String, seed: Long, k: Int, count: Int, tRun: Long): Boolean {
    // a length always goes to yue-synth as a target: it nudges the model to end near it and cuts only at 130% (the model's soft end).
    // "The song's own length" targets 4 minutes: a song the model never ended ran 8800 frames (6 min, two hours on an 8 Gen 2)
    // and the system closed the app. Lyrics with no letters (",") are a song with no words, which the model rarely
    // ends by itself: 2 minutes unless a length was chosen
    val lyr = prompt.substringAfter('\n', "")
    val words = lyr.replace(Regex("""\[[^\]]*\]"""), "").any { it.isLetter() }
    // A song with words ends when its lyrics do: the length chips are for one without (the model rarely ends that by itself).
    // With words: only the far guard (nudged from 3:24, cut at 5:12); without: the length chosen, never ending before 70% of it
    val secs = if (words) 240 else yueSeconds().takeIf { it > 0 } ?: 120
    val floor = if (words) 0 else yueSeconds() * 7 / 10
    Logs.add(if (words) "yue: lyrics decide the length (guard at ${secs}s)" else "yue: instrumental, ${secs}s aimed at, no end before ${floor}s")
    val f = File(File(root.parentFile, "music").apply { mkdirs() }, "kroma_yue_$seed.mp3")
    // the share of the whole each part takes on an 8 Gen 2 (score 15%, semantic 30%, NAR 20%, VAE 35%)
    val share = mapOf("score" to (0 to 15), "semantic" to (15 to 30), "nar" to (45 to 20), "vae" to (65 to 35))
    val names = mapOf("score" to tr(R.string.writing_the_score),
        "semantic" to tr(R.string.composing),
        "nar" to tr(R.string.rendering_the_sound),
        "vae" to tr(R.string.decoding_the_audio))
    runOnUiThread { bar.isIndeterminate = false; bar.max = 10000; bar.progress = 0 }
    Logs.add("yue seed $seed")
    val ok = Yue.make(applicationInfo.nativeLibraryDir, root, prompt, secs, seed, f, { Logs.add(it) }, { st, a, b ->
        share[st]?.let { (s0, w) ->
            val pct = s0 + if (b > 0) w * a.coerceAtMost(b) / b else 0
            val line = (if (count > 1) tr(R.string.picture, k + 1, count) else "") + (names[st] ?: st) + (if (b > 0 && st != "vae") " ${a.coerceAtMost(b)}/$b" else "") + " · $pct%"
            if (!stopRequested) status(line)
            runOnUiThread { bar.setProgress(pct * 100, true) }
            GenService.update(this, line, pct, 100)
        }
    }, stop = { stopRequested }, floor = floor)
    if (!ok) { status(tr(R.string.stopped)); return false }
    runCatching {
        val cv = android.content.ContentValues().apply {
            put(MediaStore.Audio.Media.DISPLAY_NAME, f.name); put(MediaStore.Audio.Media.MIME_TYPE, "audio/mpeg")
            put(MediaStore.Audio.Media.RELATIVE_PATH, "Music/Kroma")
        }
        contentResolver.insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, cv)?.let { u -> contentResolver.openOutputStream(u)?.use { o -> f.inputStream().use { it.copyTo(o) } } }
    }.onFailure { Logs.add("song not put in Music: ${it.message}") }
    // no autoplay (it could not be turned off): the song waits behind a play button
    runOnUiThread { songFile = f; songBtn?.apply { text = tr(R.string.play_the_song); visibility = View.VISIBLE } }
    Logs.add("done in ${hhmm((System.currentTimeMillis() - tRun) / 1000)}")
    status(tr(R.string.done_s_seed_d_saved_to_2).format(hhmm((System.currentTimeMillis() - tRun) / 1000), seed))
    return true
}

/** "Translated by PLaMo" and the translation, under the run's result (PLaMo's license asks for it to be shown), or nothing */
internal fun MainActivity.trNote(): String = trText?.let { "\n" + tr(R.string.translated_by_plamo_s).format(it) } ?: ""
/** The content filter between a finished picture and the user. True when the picture was shown and saved as is. */
internal fun MainActivity.deliver(bmp: Bitmap, prompt: String, name: String, meta0: String, model: String): Boolean {
    // the picture's text keeps the prompt as written too, and who translated it
    val meta1 = trSource?.let { src -> runCatching { org.json.JSONObject(meta0).put("prompt_original", src).put("translated_by", "PLaMo 3 Translate 0.8B").toString() }.getOrNull() } ?: meta0
    // the model's id and the LoRAs on it, for reading the settings back from the picture (applyPictureMeta)
    val meta = runCatching { org.json.JSONObject(meta1).apply { put("model_id", model)
        loraList(model).takeIf { it.isNotEmpty() }?.let { l -> put("loras", org.json.JSONArray().also { a -> l.forEach { (n, p) -> a.put(org.json.JSONObject().put("name", n).put("pct", p)) } }) } }.toString() }.getOrDefault(meta1)
    // an adult who turned adult content on, and a prompt with no word for a minor: the picture is shown whatever the classifier says,
    // so it is not asked. With a minor's word it still runs: a flagged picture is discarded then, for everyone
    val ask = !(showsAdult() && !Safety.youth(prompt))
    // no message of its own while the classifier runs: the screen keeps the decoding line; what it finds is told below
    val p = if (ask) Safety.nsfw(models.root, bmp) else 0f
    Logs.add(if (ask) "classifier %.3f".format(p) else "classifier skipped (adult setting on, no minor's word)"); lastScore = p
    if (p >= Safety.THRESHOLD) {
        if (Safety.youth(prompt)) {
            Logs.add("picture discarded by the filter")
            runOnUiThread { img.setImageBitmap(null); lastBitmap = null; lastFile = null; buttons() }
            status(tr(R.string.picture_discarded_minor, p) + Safety.minorBasis(model))
            return false
        }
        if (!showsAdult()) {
            Logs.add("picture blurred by the filter")
            runOnUiThread { img.setImageBitmap(Safety.blur(bmp)); lastBitmap = null; lastFile = null; buttons() }
            // kept in memory only, so that an adult who turns the setting on now gets this picture rather than a blank
            blurred = Triple(bmp, name, meta)
            // the edition without the setting does not point at one
            val can = BuildConfig.ADULT_OPTION
            status(tr(R.string.picture_blurred_adult, p, Safety.THRESHOLD) + (if (can) tr(R.string.adult_setting_can_turn_off) else "") + tr(R.string.classifier_can_be_wrong) + Safety.adultBasis(model))
            return false
        }
    }
    val f = keep(bmp, name, meta)
    blurred = null
    runOnUiThread { img.setImageBitmap(bmp); lastBitmap = bmp; lastFile = f; lastMeta = meta; addShot(f, meta); if (!running) buttons() }
    graphView?.let { it.picture = bmp; it.active = if (afterRun != null) "upscale" else null; it.progress = -1f }
    return true
}

/** After the adult setting is turned on: the picture blurred last is shown as it is and can be saved. */
internal fun MainActivity.releaseBlurred() {
    val (bmp, name, meta) = blurred ?: return
    if (!showsAdult()) return
    blurred = null
    thread {
        val f = keep(bmp, name, meta)
        Logs.add("blurred picture released by the adult setting")
        runOnUiThread { img.setImageBitmap(bmp); lastBitmap = bmp; lastFile = f; lastMeta = meta; if (!running) buttons() }
        status(tr(R.string.picture_shown_as_is))
    }
}

/**
 * A finished picture goes to the app's cache as a PNG, not to the gallery: the user looks first and saves what is worth
 * keeping (Settings has "save automatically" for those who want every picture). What is never saved is gone when the
 * app next starts.
 */
internal fun MainActivity.keep(bmp: Bitmap, name: String, meta: String?): File {
    val png = java.io.ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }.toByteArray()
    // a name of its own: pictures of one seed (another LoRA, another prompt) all wrote kroma_<seed>.png, and the shots and the
    // pictures waiting to be saved, which keep the file, all showed the last one
    val dir = File(cacheDir, "pictures").apply { mkdirs() }
    var f = File(dir, "$name.png"); var k = 2; while (f.exists()) f = File(dir, "${name}_${k++}.png")
    f.writeBytes(meta?.let { m -> PngMeta.withText(png, m) } ?: png)
    if (!(prefs.getBoolean("auto_save", false) && export(f) != null)) synchronized(pending) { pending.add(f) }
    return f
}

/** Save / Share / Upscale follow what there is: a picture on screen, and pictures not in the gallery yet. */
internal fun MainActivity.buttons() {
    if (!saveBtnReady) return
    val n = synchronized(pending) { pending.size }
    extraBtns.forEach { it.isEnabled = lastFile != null && !running }
    reportTv?.visibility = if (lastFile != null && BuildConfig.REPORT_LINK) View.VISIBLE else View.GONE   // the store edition only
    val shownWaits = lastFile?.let { f -> synchronized(pending) { f in pending } } == true
    saveBtn.isEnabled = shownWaits && !running
    saveBtn.text = if (lastFile != null && !shownWaits) tr(R.string.saved) else tr(R.string.save)
    saveAllBtn.visibility = if (n > 1) View.VISIBLE else View.GONE; saveAllBtn.isEnabled = !running; saveAllBtn.text = tr(R.string.save_n, n)
    if (pagerReady) { pager.visibility = if (shots.size > 1) View.VISIBLE else View.GONE; pagerTv.text = "${shotAt + 1} / ${shots.size}" }
}

/** a finished picture joins this session's list and is the one on screen (the list keeps the last 50) */
internal fun MainActivity.addShot(f: File, meta: String?) {
    shots.add(f to meta); if (shots.size > 50) shots.removeAt(0); shotAt = shots.size - 1; buttons()
}

/** page to picture i of this session's list */
internal fun MainActivity.showShot(i: Int) {
    if (i < 0 || i >= shots.size || i == shotAt) return
    val (f, meta) = shots[i]
    val bmp = android.graphics.BitmapFactory.decodeFile(f.absolutePath) ?: return
    shotAt = i; img.setImageBitmap(bmp); lastBitmap = bmp; lastFile = f; lastMeta = meta; buttons()
}

internal fun MainActivity.saveShown() {
    val f = lastFile ?: return
    if (synchronized(pending) { pending.remove(f) }) { val ok = export(f) != null; Logs.add("saved 1 to the gallery: $ok"); status(tr(R.string.saved_to_gallery, if (ok) 1 else 0)) }
    buttons()
}

internal fun MainActivity.saveAll() {
    val files = synchronized(pending) { pending.toList().also { pending.clear() } }
    val done = files.count { export(it) != null }
    Logs.add("saved $done of ${files.size} to the gallery"); status(tr(R.string.saved_to_gallery, done)); buttons()
}

/**
 * Reporting a picture the model should not have made, without leaving the app (app stores ask this of apps that generate
 * content). Nothing is sent by the app itself: a short text -- model, size, steps, seed, the filter's score, and the prompt
 * only if the user ticks it -- goes to whatever the user picks in the share sheet, meant for the project's issue tracker.
 */
internal fun MainActivity.reportPicture() {
    val meta = lastMeta ?: return
    val withPrompt = check(tr(R.string.report_include_prompt))
    AlertDialog.Builder(this).setTitle(tr(R.string.report_this_picture))
        .setView(column(note(tr(R.string.report_what_is_sent)), withPrompt))
        .setPositiveButton(tr(R.string.next)) { _, _ ->
            val o = org.json.JSONObject(meta); if (!withPrompt.isChecked) o.remove("prompt")
            o.put("filter_score", "%.3f".format(lastScore)); o.put("device", "${Build.MANUFACTURER} ${Build.MODEL} ${Build.SOC_MODEL}")
            Logs.add("picture reported")
            startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_SUBJECT, "Kroma content report")
                .putExtra(Intent.EXTRA_TEXT, "Kroma content report\n" + o.toString(2) + "\n\n" + REPO_URL + "/issues"), tr(R.string.report_this_picture)))
        }.setNegativeButton(tr(R.string.cancel), null).show()
}

/** A picture from the cache into Pictures/Kroma. */
internal fun MainActivity.export(f: File): Uri? = runCatching {
    val cv = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, f.name); put(MediaStore.Images.Media.MIME_TYPE, "image/png")
        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Kroma"); put(MediaStore.Images.Media.IS_PENDING, 1)
    }
    val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv) ?: return null
    contentResolver.openOutputStream(uri)?.use { o -> f.inputStream().use { it.copyTo(o) } }
    cv.clear(); cv.put(MediaStore.Images.Media.IS_PENDING, 0); contentResolver.update(uri, cv, null, null)
    uri
}.onFailure { Logs.add("save to the gallery failed: ${it.message}") }.getOrNull()

internal fun MainActivity.upscale() {
    val src = lastBitmap ?: return
    if (running) return
    running = true; graphView?.locked = true; stopRequested = false; Engine.abort(false); goBtn.text = tr(R.string.stop); extraBtns.forEach { it.isEnabled = false }; bar.visibility = View.VISIBLE
    GenService.start(this)   // with the screen off the system would freeze the app's process
    thread {
        try {
            val up = Upscale.esrgan2x(src, File(models.root, "onnx/esr"), { stopRequested }) { d, n ->
                runOnUiThread { bar.max = n; bar.progress = d }; graphView?.let { it.active = "upscale"; it.progress = d.toFloat() / n }; if (d == n || d % 10 == 0) status(tr(R.string.upscaling, d, n)) }
            val f = keep(up, "kroma_x2_${System.currentTimeMillis() / 1000}", lastMeta)
            runOnUiThread { img.setImageBitmap(up); lastBitmap = up; lastFile = f; addShot(f, lastMeta); buttons() }
            graphView?.let { it.picture = up; it.active = null; it.progress = -1f }
            status(tr(R.string.upscaled_to, up.width, up.height))
        } catch (e: InterruptedException) { status(tr(R.string.stopped))
        } catch (e: Throwable) { Logs.add("upscale ERROR ${e.stackTraceToString().take(2000)}"); status(tr(R.string.error) + e.message)
        } finally { GenService.stop(this, null); graphView?.let { it.active = null; it.progress = -1f; it.locked = false }; runOnUiThread { running = false; goBtn.text = tr(R.string.generate); bar.visibility = View.INVISIBLE; buttons() } }
    }
}

/** the picture's weight format for its metadata: the format.txt of the blocks it ran on (a fine-tune's, else the built-in model's;
 *  none: fp16). It said fp16 for every picture, Q8 and Q4 ones included */
internal fun MainActivity.weightFmt(id: String): String {
    val c = Advanced.custom(id); val root = models.root
    val dir = if (c.isNotEmpty() && File(root, "custom/$c/engine").isDirectory) File(root, "custom/$c/engine") else File(root, "${id}_bundle/engine")
    return when (runCatching { File(dir, "format.txt").readText().trim().take(2) }.getOrDefault("")) { "q8" -> "Q8"; "q4" -> "Q4"; else -> "fp16" }
}

/** Releases the song's player and puts the button back to Play. */
internal fun MainActivity.stopSong() {
    runCatching { songPlayer?.release() }; songPlayer = null
    songBtn?.text = tr(R.string.play_the_song)
}

/** The button under the progress line: plays the last song, and stops it while it plays (it also stops by itself at the end). */
internal fun MainActivity.toggleSong() {
    if (songPlayer != null) { stopSong(); return }
    val f = songFile ?: return
    runCatching {
        songPlayer = android.media.MediaPlayer().apply { setDataSource(f.absolutePath); prepare(); setOnCompletionListener { stopSong() }; start() }
        songBtn?.text = tr(R.string.stop_the_song)
    }.onFailure { Logs.add("song not played: ${it.message}"); stopSong() }
}
