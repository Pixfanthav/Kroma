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

/* MainActivity: the node graph (extended mode) (moved out of MainActivity.kt unchanged) */

// ---------------------------------------------------------------- node graph (extended mode)
internal fun MainActivity.graphFile() = File(models.root, "graphs/main.json")

internal fun MainActivity.graphCard(model: String, prompt: String): View {
    val g = Graph.load(graphFile()) { Graph.starter(graphStart(model, prompt)) }.also { graph = it }
    lateinit var gv: GraphView
    gv = GraphView(this, g, resources.displayMetrics.density, INK, MUTED, FIELD, ACCENT, { n -> editNode(n) { gv.invalidate(); Graph.save(graphFile(), g) } }) { Graph.save(graphFile(), g) }
    gv.setBackgroundColor(CARD); graphView = gv; gv.picture = lastBitmap
    // a long press on the empty canvas: the node menu, the node placed there; Start over and Fit at its end
    // the node menu: a long press where it should go, or the + in the corner (then it lands in the middle of the view)
    fun nodeMenu(x: Float, y: Float) {
        if (!running) {
            val ks = Graph.KINDS
            val items = ks.map { it.title() } + listOf(tr(R.string.fit_to_screen), tr(R.string.start_over))
            AlertDialog.Builder(this).setItems(items.toTypedArray()) { _, i ->
                when {
                    i < ks.size -> { g.add(ks[i].id, x, y); Graph.save(graphFile(), gv.g); gv.invalidate() }
                    i == ks.size -> gv.fit()
                    else -> { gv.g = Graph.starter(graphStart(model, prompt)).also { graph = it; Graph.save(graphFile(), it) }; gv.fit() }
                }
            }.show()
        }
    }
    gv.onLongPress = { x, y -> nodeMenu(x, y) }
    val plus = styled(Button(this), FIELD, INK).apply {
        text = "+"; textSize = 20f; alpha = 0.9f
        minWidth = px(46); minimumWidth = px(46); minHeight = px(46); minimumHeight = px(46); setPadding(0, 0, 0, 0)
        setOnClickListener { val (cx, cy) = gv.centre(); nodeMenu(cx, cy) }
    }
    // the graph alone, as tall as the screen less the title, the Generate line and the drawer handle
    return card(android.widget.FrameLayout(this).apply {
        addView(gv, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, resources.displayMetrics.heightPixels - px(250)))
        addView(plus, android.widget.FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP or Gravity.END).apply { topMargin = px(8); marginEnd = px(8) })
    })
}

/** what the plain screen has set, for a new graph */
internal fun MainActivity.graphStart(model: String, prompt: String): Graph.Start {
    Advanced.kind = Advanced.kindOf(model); val d = Advanced.defaults()
    val (w, h) = curSize(model, RATIOS.indexOfFirst { it.first == prefs.getString("ratio_name", "2:3") }.coerceAtLeast(0))
    return Graph.Start(model, Advanced.custom(Advanced.kind), prompt, Advanced.negative, w, h,
        prefs.getInt("steps", d.steps), "", prefs.getInt("count", 1).coerceIn(1, 99), loraList(model),
        Advanced.sampler, Advanced.scheduler, Advanced.cfg, Advanced.shift)
}

/** the image models the graph can run: installed ones, and MOSS / Klein / SDXL files that are on the phone though no pack lists them */
internal fun MainActivity.graphModels(): List<String> {
    val have = models.installed(chip!!).filter { it.isModel }.map { it.id }
    val extra = listOfNotNull("moss".takeIf { MossGen.present(models.root) }, "klein".takeIf { KleinGen.present(models.root) },
        "klein9".takeIf { KleinGen.present(models.root, "klein9_bundle") }, "sdxl".takeIf { SdxlGen.present(models.root) })
    return have + extra.filter { it !in have }
}

/** a node's values in a dialog (and its deletion) */
internal fun MainActivity.editNode(n: Graph.Node, done: () -> Unit) {
    val g = graph ?: return
    // running covers a picture, a model download and a conversion alike, so the line does not name one of them
    if (running) { status(tr(R.string.the_graph_can_t_be_changed)); return }
    val box = column()
    val fields = LinkedHashMap<String, EditText>()
    val gm = g.nodes.firstOrNull { it.kind == "model" }?.p?.get("model") ?: "anima"
    var strengthField: EditText? = null                                           // the LoRA node's strength, typed as 0.65
    /** a number box with its caption above it, alone on a line or two side by side */
    fun numCol(key: String, label: String, decimal: Boolean = false): View {
        val e = field(EditText(this).apply { setText(n.p[key]); setSingleLine()
            inputType = InputType.TYPE_CLASS_NUMBER or (if (decimal) InputType.TYPE_NUMBER_FLAG_DECIMAL else 0) }); fields[key] = e
        return column(note(label), e)
    }
    fun num(key: String, label: String, decimal: Boolean = false) { box.addView(numCol(key, label, decimal)) }
    fun pick(key: String, label: String, options: List<String>, labels: List<String> = options, perRow: Int = 3) { box.addView(note(label))
        val (r, _) = chips(labels, options.indexOf(n.p[key]).coerceAtLeast(0), quiet = true, perRow = perRow) { n.p[key] = options[it] }; box.addView(r) }
    when (n.kind) {
        "model" -> { pick("model", tr(R.string.model), graphModels())
            // the fine-tunes brought in for either model (extended mode's model choice); none: the bundle's own
            val cs = listOf("") + Custom.list(models.root).map { it.name }
            box.addView(note(tr(R.string.fine_tune_imported)))
            val (r, _) = chips(cs.map { it.ifEmpty { tr(R.string.built_in) } }, cs.indexOf(n.p["custom"] ?: "").coerceAtLeast(0), quiet = true, perRow = 2) { n.p["custom"] = cs[it] }
            box.addView(r) }
        "lora" -> { val names = listOf("") + loras(g.nodes.firstOrNull { it.kind == "model" }?.p?.get("model") ?: "anima").map { it.name }
            pick("name", "LoRA", names.map { it.ifEmpty { "—" } }.also { lst -> if (n.p["name"].isNullOrEmpty()) n.p["name"] = "" }); box.addView(note(tr(R.string.strength_1_0_as_made)))
            val se = field(EditText(this).apply { setText(strengthText(n.p["strength"]?.toIntOrNull() ?: 100)); setSingleLine(); inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL or InputType.TYPE_NUMBER_FLAG_SIGNED })
            box.addView(se); strengthField = se
            // how every LoRA reaches the model (one setting for the app, the same as the LoRA list's)
            var mb: android.widget.Button? = null
            mb = button(loraModeLabel()) { pickLoraMode { mb?.text = loraModeLabel() } }; box.addView(mb) }
        "prompt", "negative" -> { val e = field(EditText(this).apply { setText(n.p["text"]); minLines = 3; gravity = Gravity.TOP }); fields["text"] = e; box.addView(e) }
        "size", "vaeenc" -> { num("w", tr(R.string.width)); num("h", tr(R.string.height))
            // the app's sizes (the plain screen's ratios), a tap filling both
            val gm = g.nodes.firstOrNull { it.kind == "model" }?.p?.get("model") ?: "anima"
            val sizes = RATIOS.indices.map { sizeOf(gm, it) }
            val (r, _) = chips(RATIOS.map { it.first }, -1, quiet = true, perRow = 4) { i -> val (w, h) = sizes.getOrElse(i) { sizes[0] }
                fields["w"]?.setText("$w"); fields["h"]?.setText("$h") }
            box.addView(r)
            box.addView(note(tr(R.string.multiples_of_16_area_at_most)
                .format(Math.sqrt(sizeCap(gm)).toInt(), Math.sqrt(sizeCap(gm)).toInt()))) }
        "sampler" -> {
            box.addView(row(numCol("steps", tr(R.string.steps_2)), numCol("count", tr(R.string.pictures))))
            // the seed and the last one's button on one line
            box.addView(note(tr(R.string.seed_empty_random)))
            val se = field(EditText(this).apply { setText(n.p["seed"]); setSingleLine(); inputType = InputType.TYPE_CLASS_NUMBER }); fields["seed"] = se
            box.addView(lastSeed?.let { ls -> line(se, button(tr(R.string.last)) { se.setText("$ls") }) } ?: se)
            box.addView(row(numCol("denoise", tr(R.string.denoise_with_vae_encode), true)))
            run {
                box.addView(row(numCol("cfg", "CFG", true), numCol("shift", "Shift", true)))
                (if (gm == "moss") Advanced.SAMPLERS_MOSS else Advanced.SAMPLERS).let { sl -> pick("sampler", tr(R.string.sampler), sl, sl.map { samplerLabel(it) }, perRow = 2) }
                (if (gm == "krea") Advanced.SCHEDULERS_KREA else Advanced.SCHEDULERS).let { sl -> pick("scheduler", tr(R.string.scheduler), sl, sl.map { Advanced.schedLabel(it) }) }
            }
        }
        "startimg" -> {
            val f = n.p["path"]?.let { File(it) }?.takeIf { it.isFile }
            f?.let { thumbOf(it, px(220)) }?.let { b -> box.addView(ImageView(this).apply { setImageBitmap(b); adjustViewBounds = true; maxHeight = px(220) }) }
            box.addView(button(if (f == null) tr(R.string.choose_a_picture) else tr(R.string.another_picture)) {
                graphImageNode = n.id; startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("image/*"), PICK_GRAPH_IMAGE) })
            if (f != null) box.addView(button(tr(R.string.clear_2)) { f.delete(); n.p["path"] = ""; graph?.let { Graph.save(graphFile(), it) }; route() })
        }
        "save" -> { val pic = lastBitmap
            if (pic == null) box.addView(note(tr(R.string.the_picture_appears_here_once_one)))
            else { box.addView(android.widget.ImageView(this).apply { setImageBitmap(pic); adjustViewBounds = true }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
                box.addView(row(button(tr(R.string.save)) { saveShown() }, button(tr(R.string.share)) { shareLast() })) } }
        else -> box.addView(note(tr(R.string.no_settings)))
    }
    // the app's colours: the card behind, the accent on OK, a red Delete apart from the others
    val dlg = AlertDialog.Builder(this).setTitle(Graph.kind(n.kind).title()).setView(ScrollView(this).apply { addView(box) })
        .setPositiveButton("OK") { _, _ -> fields.forEach { (k, e) -> n.p[k] = e.text.toString() }; if (n.kind == "lora" && n.p["name"] == "—") n.p["name"] = ""
            strengthField?.text?.toString()?.toDoubleOrNull()?.let { n.p["strength"] = "${Math.round(it.coerceIn(-10.0, 10.0) * 100)}" }; done() }
        .setNeutralButton(tr(R.string.delete)) { _, _ -> g.remove(n); done() }
        .setNegativeButton(tr(R.string.cancel), null).show()
    dlg.window?.setBackgroundDrawable(round(CARD, 20))
    dlg.getButton(AlertDialog.BUTTON_POSITIVE)?.apply { setTextColor(ACCENT); setTypeface(typeface, Typeface.BOLD) }
    dlg.getButton(AlertDialog.BUTTON_NEGATIVE)?.setTextColor(MUTED)
    dlg.getButton(AlertDialog.BUTTON_NEUTRAL)?.setTextColor(Color.rgb(0xE5, 0x5B, 0x5B))
}

/** the graph as one job: its settings go where the plain screen keeps them, then the generator runs */
internal fun MainActivity.runGraph() {
    val g = graph ?: return
    val (p, why) = Graph.plan(g)
    if (p == null) { status(why ?: ""); return }
    if (p.model.startsWith("klein") && p.start.isNotEmpty() && File(p.start).isFile) {
        // Klein: the picture is the reference to edit, whatever the denoise (it has none)
        File(p.start).copyTo(imgStart(), true); File(cacheDir, "img_ref2.img").delete(); File(cacheDir, "img_ref3.img").delete(); imgMask().delete()
    } else if (p.start.isNotEmpty() && File(p.start).isFile && p.denoise < 1.0) {
        // img2img: the start image and the denoise, where the plain screen keeps them
        File(p.start).copyTo(imgStart(), true); prefs.edit().putInt("denoise", Math.round(p.denoise * 100).toInt()).apply()
    } else imgStart().delete()
    // the size: sides in 16s, the area within the model's cap (the plain screen's sliders: Klein / MOSS 2048², SDXL 1.5 MP and its padding rule)
    if ((p.w % 16 != 0 || p.h % 16 != 0 || p.w.toDouble() * p.h > sizeCap(p.model) || p.model == "sdxl" && sdxlFit(p.w to p.h) != (p.w to p.h))) {
        val c = sizeCap(p.model); val side = Math.sqrt(c).toInt()
        status(tr(R.string.size_sides_in_multiples_of_16).format(side, side,
            if (p.model == "sdxl") tr(R.string.sdxl_d_d_fits).format(sdxlFit(p.w to p.h).first, sdxlFit(p.w to p.h).second) else "")); return }
    if ((p.model == "krea" || p.model == "anima") && p.start.isNotEmpty() && p.denoise < 1.0 && !KreaGen.canEncode(models.root, p.w, p.h)) {
        status(tr(R.string.vae_encode_this_phone_has_no).format(p.w, p.h)); return }
    if (p.model !in graphModels()) { status(tr(R.string.is_not_runnable_in_this, p.model)); return }
    // Anima takes any size whose sides are multiples of 16 and whose tokens fit its blocks' rows (AnimaGen.accepts; the RoPE table is computed)
    if (p.model == "anima" && !AnimaGen.accepts(File(models.root, "anima_bundle"), p.w, p.h)) {
        val n = AnimaGen.budget(File(models.root, "anima_bundle"))
        status(tr(R.string.anima_size_rule, n)); return }
    Safety.refuse(p.prompt, p.model)?.let { Logs.add("prompt refused by the filter"); status(it); return }
    val tuned = p.model in setOf("anima", "krea", "moss", "klein", "klein9", "sdxl")
    if (tuned) Advanced.setCustom(p.model, p.custom.takeIf { c -> Custom.list(models.root, p.model).any { it.name == c } } ?: "")
    if (tuned) { Advanced.kind = p.model; Advanced.sampler = p.sampler; Advanced.scheduler = p.scheduler; Advanced.cfg = p.cfg.toFloat(); Advanced.shift = p.shift.toFloat(); Advanced.negative = p.negative }
    // the plain screen shows what the graph ran (both ways: a graph starts from the plain screen, the plain screen ends as the graph)
    prefs.edit().putString("prompt", p.prompt).putInt("steps", p.steps).putInt("count", p.count).putString("model", p.model)
        .apply { RATIOS.indices.firstOrNull { sizeOf(p.model, it) == (p.w to p.h) }?.let { putString("ratio_name", RATIOS[it].first) }
                 if (sizeScalable(p.model)) { putInt("size_w", p.w); putInt("size_h", p.h) } }.apply()
    setLoraList(p.model, p.loras)
    afterRun = if (p.upscale) ({ upscale() }) else null
    Logs.add("graph run: ${p.model} ${p.w}x${p.h} ${p.steps} steps, ${p.sampler}/${p.scheduler}, cfg ${p.cfg}, lora ${if (p.loras.isEmpty()) "none" else p.loras.joinToString(", ") { "${it.first} ${it.second}" }}, upscale ${p.upscale}")
    generate(List(p.count) { Job(p.prompt, p.steps, p.seed?.plus(it), p.w, p.h, p.model, p.loras) })
}
