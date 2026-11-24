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

/**
 * One screen at a time, built in code: the first-run agreement, the model download for this phone's chip,
 * then generation. Krea 2 Turbo only: prompt, steps, seed, aspect ratio, repeat count.
 */
class MainActivity : Activity() {
    lateinit var models: Models
    val prefs by lazy { getSharedPreferences("kroma", MODE_PRIVATE) }
    /** debug builds only: the screens on a phone that cannot run the models (to look at the layout and the wording), as an 8 Gen 2 */
    internal val uiPreview by lazy { BuildConfig.DEBUG && Chip.detect() == null && prefs.getBoolean("ui_preview", false) }
    val chip by lazy { Chip.detect() ?: Chip.GEN2.takeIf { uiPreview } }
    @Volatile internal var running = false
    @Volatile internal var trSource: String? = null; @Volatile internal var trText: String? = null   // this run's prompt as written and as PLaMo translated it (Translator)
    @Volatile internal var stopRequested = false
    /** the step the picture being made is at (0 on), -1 before its first; and its steps: the stop button's choice */
    @Volatile internal var stepNow = -1
    @Volatile internal var stepsNow = 0
    internal var lastBitmap: Bitmap? = null
    internal var lastFile: File? = null                 // the picture on screen, as a PNG in the app's cache: what Share sends
    internal val pending = ArrayList<File>()            // finished pictures not in the gallery yet: Save puts them there
    internal val shots = ArrayList<Pair<File, String?>>() // this session's pictures (and their PNG text), paged under the picture
    internal var shotAt = -1                            // which of them is on screen
    internal lateinit var pager: LinearLayout; internal lateinit var pagerTv: TextView; internal lateinit var saveAllBtn: Button
    internal lateinit var saveBtn: Button
    internal lateinit var tempTv: TextView
    internal var graph: Graph.G? = null                 // the node graph on screen (extended mode), null on the plain screen
    internal var graphView: GraphView? = null           // its view: which node is at work, and the last picture in the Image node
    internal var afterRun: (() -> Unit)? = null         // run once after a generation that made pictures (the graph's upscale)
    // the seed of the latest picture, kept across launches: "Fixed" puts it in the seed box
    internal var lastSeed: Long? = null
        get() = field ?: prefs.getLong("last_seed", -1L).takeIf { it >= 0 }
    internal var reportTv: TextView? = null             // "Report this picture": shown under a finished picture
    internal var lastScore = -1f                        // the content filter's score for the picture on screen
    internal var loraBtn: Button? = null                // "LoRA: <name> · 80%" on the main screen; opens the LoRA list
    internal var graphImageNode = 0                     // the Start image node a picked picture goes to
    internal var yueLenBtn: View? = null                // YuE2: the song's length (a label and its chips)
    internal fun yueSeconds() = prefs.getInt("yue_seconds", 0)   // 0: as long as the song the model writes
    internal var songPlayer: android.media.MediaPlayer? = null   // the last song, while it plays
    internal var songFile: File? = null                          // the last song made
    internal var songBtn: Button? = null                         // play / stop the last song (hidden until there is one)
    internal var imgStartBtn: Button? = null            // Krea 2 / Anima: img2img's start picture
    internal var denoiseBtn: Button? = null             // and how much of it is redrawn
    /** the start picture(s) as small thumbnails under their button: what was picked is seen */
    internal var startThumbs: LinearLayout? = null
    /** fills [startThumbs] from the files the current model takes; hidden when there are none */
    internal fun startThumbsRefresh() {
        val box = startThumbs ?: return; box.removeAllViews()
        val files = if (loraModel == "yue" || loraModel == "moss") emptyList() else imgRefs()
        val hh = if (files.size <= 1) px(280) else px(150)      // a single start picture is looked at, references are a row
        for (f in files) thumbOf(f, hh)?.let { b -> box.addView(ImageView(this).apply {
            setImageBitmap(b); adjustViewBounds = true; maxHeight = hh; maxWidth = if (files.size <= 1) px(520) else px(160); setPadding(0, 0, px(8), 0) }) }
        box.visibility = if (box.childCount > 0) View.VISIBLE else View.GONE
    }
    /** a small copy of a picture file (about [h] pixels high), decoded subsampled */
    internal fun thumbOf(f: File, h: Int): android.graphics.Bitmap? = runCatching {
        val o = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }; android.graphics.BitmapFactory.decodeFile(f.absolutePath, o)
        var s = 1; while (o.outHeight / (s * 2) >= h) s *= 2
        android.graphics.BitmapFactory.decodeFile(f.absolutePath, android.graphics.BitmapFactory.Options().apply { inSampleSize = s })
    }.getOrNull()
    /** a picked photo turned upright: a camera's JPEG keeps its rotation in EXIF, which BitmapFactory ignores (a portrait photo went
     *  in on its side once); rewritten in place as a JPEG only when it has one */
    internal fun upright(f: File) { runCatching {
        val r = android.media.ExifInterface(f.absolutePath).getAttributeInt(android.media.ExifInterface.TAG_ORIENTATION, android.media.ExifInterface.ORIENTATION_NORMAL)
        val m = android.graphics.Matrix()
        when (r) { android.media.ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f); android.media.ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
                   android.media.ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f); android.media.ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
                   android.media.ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
                   android.media.ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
                   android.media.ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
                   else -> return }
        val b = android.graphics.BitmapFactory.decodeFile(f.absolutePath) ?: return
        val u = android.graphics.Bitmap.createBitmap(b, 0, 0, b.width, b.height, m, true); b.recycle()
        f.outputStream().use { o -> u.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, o) }; u.recycle()
        Logs.add("start image turned upright (EXIF $r)")
    }.onFailure { Logs.add("start image upright: ${it.message}") } }
    internal fun imgStart() = File(cacheDir, "img_start.img")
    /** Klein's reference pictures: the start image and up to two more (a multiple pick) */
    internal fun imgRefs() = listOf(imgStart(), File(cacheDir, "img_ref2.img"), File(cacheDir, "img_ref3.img")).filter { it.isFile }
    /** the start picture's button: img2img's start for Krea 2 / Anima, the reference to edit for FLUX.2 Klein (KleinGen.Ref) */
    internal fun startLabel(): String = if (loraModel.startsWith("klein"))
        (if (imgStart().isFile) tr(R.string.reference_pictures_edit_d).format(imgRefs().size)
         else tr(R.string.reference_pictures_edit_none_up_to))
        else if (imgStart().isFile) tr(R.string.start_image_set)
        else tr(R.string.start_image_none_text_to_image)
    internal fun denoisePct() = prefs.getInt("denoise", 60)
    internal var maskBtn: Button? = null                // inpaint: the mask painted on the start picture
    internal var fitBtn: Button? = null                 // cut the picture to the frame, or fit it in and draw the rest (outpaint)
    internal fun imgMask() = File(cacheDir, "img_mask.png")
    internal fun imgFit() = prefs.getBoolean("img_fit", false)
    internal var negEt: EditText? = null                // the extended mode's negative prompt box (shows the current model's negative)
    internal var loraModel = "krea"                     // the model the main screen shows, which is whose LoRAs are listed
    /** rebuilds the extended settings in place (set by the drawer): a model change must not leave the last model's sampler shown */
    internal var extRefill: (() -> Unit)? = null
    internal lateinit var memTv: TextView               // free memory, shown for the model whose speed depends on it
    internal var lastMeta: String? = null
    /** a seed for the seed box when the screen is next built (a picture's settings read back: MainSettings.applyPictureMeta) */
    internal var pendingSeed: Long? = null

    internal lateinit var statusTv: TextView
    internal lateinit var bar: ProgressBar
    /** one cell per block while a model is converted or a LoRA baked into the weights */
    internal lateinit var grid: BlockGrid
    internal lateinit var lockTv: TextView      // shown while generating: a locked screen puts the app on the slow cores
    internal lateinit var askTv: TextView       // after the first couple of pictures on a trial chip: offers to send the log
    internal lateinit var img: ImageView
    internal lateinit var goBtn: Button
    internal var extraBtns = listOf<Button>()
    /** What the system back gesture does on the current screen (null = leave the app). */
    internal var onBack: (() -> Unit)? = null
    @Deprecated("the simple back hook is enough for one activity") override fun onBackPressed() { onBack?.invoke() ?: super.onBackPressed() }

    internal val dp get() = resources.displayMetrics.density
    internal fun px(v: Int) = (v * dp).toInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Logs.init(this); Lang.load(this); models = Models(this)
        LoraBrowse.apiKey = prefs.getString("civitai_key", null)
        // debug builds only: a files/no_filter marker turns the prompt and picture checks off
        Safety.off = BuildConfig.DEBUG && File(filesDir, "no_filter").exists()
        Advanced.init(this); Advanced.defaultsOf = { Custom.defaults(models.root, it) }
        File(cacheDir, "pictures").deleteRecursively()      // pictures of the last session that were never saved
        Thread.setDefaultUncaughtExceptionHandler { _, e -> Logs.add("CRASH ${e.stackTraceToString().take(4000)}"); android.os.Process.killProcess(android.os.Process.myPid()) }
        // only when not granted yet: asking again opened the system's permission screen over the app for a moment on every start (a flicker)
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED) requestPermissions(arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), 1)
        route()
        Hooks.onStart?.invoke(this)
    }

    internal fun route() = when {
        !prefs.contains("lang") -> language()
        !prefs.getBoolean("agreed_terms_2", false) -> gate()
        chip == null -> unsupported()
        !models.ready(chip!!) && !prefs.getBoolean("setup_later", false) -> setup()
        else -> main()
    }

    // ---------------------------------------------------------------- small view helpers
    internal val BG = Color.rgb(0x16, 0x17, 0x1B); internal val CARD = Color.rgb(0x1F, 0x21, 0x27); internal val FIELD = Color.rgb(0x2A, 0x2C, 0x33)
    internal val INK = Color.rgb(0xEC, 0xEC, 0xF0); internal val MUTED = Color.rgb(0x9A, 0x9C, 0xA6); internal val ACCENT = Color.rgb(0xFA, 0x6F, 0x50)   // the icon's orange
    internal fun round(color: Int, r: Int = 14) = android.graphics.drawable.GradientDrawable().apply { setColor(color); cornerRadius = px(r).toFloat() }
    internal fun stack(pad: Int, gap: Int, views: Array<out View>) = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL; setPadding(px(pad), px(pad), px(pad), px(pad))
        views.forEachIndexed { i, v -> addView(v, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { if (i < views.size - 1) bottomMargin = px(gap) }) }
    }
    internal fun column(vararg views: View) = stack(18, 14, views)
    /** A rounded panel that groups related controls. */
    internal fun card(vararg views: View) = stack(16, 10, views).apply { background = round(CARD, 18) }
    /** One scroll view for the app's life: a new screen replaces what is in it (making a new one each time blanked the window for a
     *  frame -- a flicker on every switch -- and threw the scroll position away). The same screen rebuilt keeps its place. */
    internal var rootScroll: ScrollView? = null
    internal var shownKind = ""
    /** set just before [show] by a screen that is rebuilt in place (the main one), so it keeps its scroll position */
    internal var nextKind = ""
    internal fun show(content: View) {
        val kind = nextKind; nextKind = ""
        val sv = rootScroll ?: ScrollView(this).apply { fitsSystemWindows = true; setBackgroundColor(BG) }.also { rootScroll = it; setContentView(it) }
        val y = if (kind.isNotEmpty() && kind == shownKind) sv.scrollY else 0; shownKind = kind
        sv.removeAllViews(); sv.addView(content)
        if (y > 0) sv.post { sv.scrollTo(0, y) }
    }
    internal fun text(s: String, size: Float = 15f, bold: Boolean = false) = TextView(this).apply {
        text = s; textSize = size; setTextColor(INK); if (bold) setTypeface(typeface, Typeface.BOLD); setLineSpacing(0f, 1.2f)
    }
    /** Secondary text: captions, hints, figures. */
    internal fun note(s: String, size: Float = 13f) = text(s, size).apply { setTextColor(MUTED) }
    internal fun styled(b: Button, fill: Int, ink: Int) = b.apply {
        isAllCaps = false; textSize = 15f; setTextColor(ink); background = round(fill); stateListAnimator = null
        minHeight = px(50); minimumHeight = px(50); setPadding(px(18), 0, px(18), 0)
    }
    /** The one main action of a screen. */
    internal fun primary(s: String, on: () -> Unit) = styled(Button(this), ACCENT, Color.rgb(0x1B, 0x12, 0x0F)).apply { text = s; setTypeface(typeface, Typeface.BOLD); setOnClickListener { on() } }
    internal fun button(s: String, on: () -> Unit) = styled(Button(this), FIELD, INK).apply { text = s; setOnClickListener { on() } }
    internal fun field(e: EditText) = e.apply { background = round(FIELD, 12); setPadding(px(14), px(12), px(14), px(12)); setTextColor(INK); setHintTextColor(MUTED); textSize = 15f }
    internal fun check(s: String, on: Boolean = false) = CheckBox(this).apply { text = s; isChecked = on; setTextColor(INK); textSize = 15f; buttonTintList = android.content.res.ColorStateList.valueOf(ACCENT) }
    internal fun progress() = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
        progressTintList = android.content.res.ColorStateList.valueOf(ACCENT); progressBackgroundTintList = android.content.res.ColorStateList.valueOf(FIELD)
    }
    /** Views side by side: the first takes the room that is left, the rest keep their own width. */
    internal fun line(vararg views: View) = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        views.forEachIndexed { i, v -> addView(v, LinearLayout.LayoutParams(if (i == 0) 0 else ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, if (i == 0) 1f else 0f).apply { if (i > 0) marginStart = px(10) }) }
    }
    /** One choice out of a few, as a row of equal chips (the platform dropdown does not follow the app's colours). Returns the row and a reader of the chosen index. */
    internal fun chips(labels: List<String>, start: Int, quiet: Boolean = false, perRow: Int = labels.size, textSp: Float = 14f, onPick: (Int) -> Unit = {}): Pair<LinearLayout, () -> Int> {
        var sel = start.coerceIn(0, maxOf(0, labels.size - 1))
        val views = ArrayList<TextView>()
        fun paint() = views.forEachIndexed { i, v ->
            // quiet: the choice is marked by an outline and the accent as ink, for a row that must not compete with the main button
            v.background = round(if (i == sel && !quiet) ACCENT else FIELD, 12).apply { if (i == sel && quiet) setStroke(px(1) + 1, ACCENT) }
            v.setTextColor(if (i != sel) INK else if (quiet) ACCENT else Color.rgb(0x1B, 0x12, 0x0F))
            v.setTypeface(null, if (i == sel) Typeface.BOLD else Typeface.NORMAL)
        }
        // more labels than fit side by side go on further rows of the same width
        val per = perRow.coerceAtLeast(1); val rows = ArrayList<LinearLayout>()
        val box = LinearLayout(this).apply { orientation = if (labels.size > per) LinearLayout.VERTICAL else LinearLayout.HORIZONTAL }
        labels.forEachIndexed { i, s ->
            if (labels.size > per && i % per == 0) rows.add(LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }.also {
                box.addView(it, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { if (i > 0) topMargin = px(6) }) })
            val v = TextView(this).apply { text = s; textSize = textSp; gravity = Gravity.CENTER; setSingleLine(); setPadding(px(4), px(11), px(4), px(11))
                setOnClickListener { sel = i; paint(); onPick(i) } }
            views.add(v); (rows.lastOrNull() ?: box).addView(v, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { if (i % per > 0) marginStart = px(6) })
        }
        paint(); return box to { sel }
    }
    internal fun row(vararg views: View) = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        dividerDrawable = android.graphics.drawable.GradientDrawable().apply { setSize(px(10), 1) }; showDividers = LinearLayout.SHOW_DIVIDER_MIDDLE
        views.forEach { v -> addView(v, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)) }
    }
    internal fun gb(n: Long) = if (n >= 1_000_000_000L) "%.1f GB".format(n / 1e9) else "%.0f MB".format(n / 1e6)
    internal fun asset(name: String) = assets.open(name).use { it.readBytes().toString(Charsets.UTF_8) }

    // ---------------------------------------------------------------- generation
    internal val RATIOS = listOf(Triple("1:1", 1024, 1024), Triple("2:3", 832, 1216), Triple("3:2", 1216, 832), Triple("3:4", 896, 1152),
        Triple("4:3", 1152, 896), Triple("9:16", 768, 1344), Triple("16:9", 1344, 768))
    internal var sizeW = 0; internal var sizeH = 0      // the two size sliders' (0 = not set yet: the ratio's size, sizeOf); free ratio, see curSize

    internal fun main() {
        onBack = null; graph = null; sizeW = prefs.getInt("size_w", 0); sizeH = prefs.getInt("size_h", 0)
        // no model on this phone yet (downloading was put off): the whole screen anyway, to look around, with a panel at the top that
        // says so and leads to the downloads; drawing says the same
        val noModels = !uiPreview && models.installed(chip!!).none { it.isModel }
        val noModelPanel = if (!noModels) null else card(
            note(tr(R.string.no_model_on_this_phone_yet)),
            primary(tr(R.string.download_models)) { setup() })
        val promptEt = EditText(this).apply { hint = tr(R.string.describe_the_picture_in_english); minLines = 4; gravity = Gravity.TOP
            setText(prefs.getString("prompt", "")); inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE }
        // YuE2: the prompt box takes the style tags and this one the lyrics (with [Verse] / [Chorus] markers), shown only for it
        val lyricsEt = EditText(this).apply {
            hint = tr(R.string.lyrics_in_parts_marked_verse_chorus)
            minLines = 8; gravity = Gravity.TOP; setText(prefs.getString("lyrics", "")); inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE }
        val lyricsBox = field(lyricsEt).apply { visibility = View.GONE }
        // the negative prompt (extended mode): the image models' CFG; YuE2 has none, so it is hidden there
        val negBox = negativeBox().takeIf { Advanced.on }
        // How long the prompt is, counted with the model's own tokenizer as it is typed. The limit is the text rows
        // of the chosen shape (the image rows a shape leaves unused carry text too, so it differs by ratio) less
        // the chat template's closing tokens: what is over it never reaches the picture.
        val lenTv = note("", 12f)
        // the prompt's own translation on demand, before a run: the English the model will get, editable; tapped again, the original back
        var trOrig: String? = null; var trBusy = false; var trSet = false
        val trBtn = android.widget.TextView(this).apply {
            textSize = 13f; setTextColor(ACCENT); setPadding(px(12), px(6), px(12), px(6)); visibility = View.GONE
            background = android.graphics.drawable.GradientDrawable().apply { cornerRadius = px(14).toFloat(); setStroke(px(1), ACCENT) } }
        fun trLabel() { trBtn.text = if (trOrig != null) tr(R.string.original)
                        else tr(R.string.translate) }
        var limit = 123
        val cfg = runCatching { File(models.bundle, "krea.txt").readText().trim().split(" ").filter { it.isNotEmpty() }.map { it.toInt() } }.getOrNull()
        val suffix = runCatching { org.json.JSONObject(File(models.root, "krea/template.json").readText()).getJSONArray("suffix").length() }.getOrDefault(5)
        fun limitFor(w: Int, h: Int) = if (cfg == null) 123 else KreaGen.textRows(cfg[2], (cfg[0] / 16) * (cfg[1] / 16), w, h) - suffix
        val counter = java.util.concurrent.Executors.newSingleThreadExecutor()
        var tok: QwenTok? = null
        fun count() { val text = promptEt.text.toString()
            counter.execute {
                val t = tok ?: runCatching { QwenTok(File(models.root, "tok")) }.getOrNull()?.also { tok = it } ?: return@execute
                val n = t.encode(text).size
                runOnUiThread { if (promptEt.text.toString() == text) {
                    lenTv.text = if (n <= limit) tr(R.string.length, n, limit)
                                 else tr(R.string.length_too_long_the_end, n, limit)
                    lenTv.setTextColor(if (n <= limit) MUTED else ACCENT) } }
            } }
        promptEt.addTextChangedListener(object : android.text.TextWatcher {
            override fun afterTextChanged(e: android.text.Editable?) { count()
                if (!trSet) trOrig = null   // typed over: the translation's original no longer applies
                trBtn.visibility = if (trOrig != null || prefs.getString("model", "") != "yue") View.VISIBLE else View.GONE; trLabel() }
            override fun beforeTextChanged(c: CharSequence?, a: Int, b: Int, d: Int) {}
            override fun onTextChanged(c: CharSequence?, a: Int, b: Int, d: Int) {} })
        count()
        trBtn.setOnClickListener {
            if (trBusy || running) return@setOnClickListener
            trOrig?.let { o -> trSet = true; promptEt.setText(o); promptEt.setSelection(o.length); trSet = false; trOrig = null; trLabel(); return@setOnClickListener }
            val src = promptEt.text.toString(); val root = models.root
            if (src.isBlank()) { status(tr(R.string.write_a_prompt_first)); return@setOnClickListener }
            if (!Translator.present(root)) { status(tr(R.string.the_translation_model_is_not_installed)); return@setOnClickListener }
            trBusy = true; trBtn.alpha = 0.5f; val model = prefs.getString("model", "") ?: ""
            status(tr(R.string.translating_the_prompt_plamo))
            Thread {
                val en = runCatching { Translator.translate(this, root, src, Logs::add)?.let { if (model == "anima" || model == "sdxl") TagSnap.snap(root, it, Logs::add) else it } }.getOrNull()
                runOnUiThread { trBusy = false; trBtn.alpha = 1f
                    if (en == null) { status(tr(R.string.error) + "translation"); return@runOnUiThread }
                    if (promptEt.text.toString() != src) return@runOnUiThread   // typed over meanwhile
                    trSet = true; promptEt.setText(en); promptEt.setSelection(en.length); trSet = false; trOrig = src; trLabel(); trBtn.visibility = View.VISIBLE; status("") }
            }.start()
        }
        trLabel(); trBtn.visibility = if (prefs.getString("model", "") != "yue") View.VISIBLE else View.GONE
        val stepsTv = text("")
        // 4..16; advanced mode 4..50 (a fine-tune that is not a turbo model wants about 30)
        val stepsSb = SeekBar(this).apply { max = if (Advanced.on) 46 else 12; progress = (prefs.getInt("steps", if (prefs.getString("model", "") in setOf("klein", "klein9")) 4 else 8) - 4).coerceAtMost(max)
            progressTintList = android.content.res.ColorStateList.valueOf(ACCENT); thumbTintList = android.content.res.ColorStateList.valueOf(ACCENT) }
        fun steps() = stepsSb.progress + 4
        var warmHook: (() -> Unit)? = null     // Anima's modulations computed ahead (AnimaGen.warmMods): set below, once the model is known
        fun stepLabel() { stepsTv.text = tr(R.string.steps, steps()) }
        stepsSb.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(b: SeekBar?, v: Int, u: Boolean) = stepLabel()
            override fun onStartTrackingTouch(b: SeekBar?) {}
            override fun onStopTrackingTouch(b: SeekBar?) { warmHook?.invoke() } }); stepLabel()
        val seedEt = EditText(this).apply { hint = tr(R.string.seed); inputType = InputType.TYPE_CLASS_NUMBER; setSingleLine() }
        // seed: the box as always (a number is used, empty = a new random seed every picture). "Fixed" puts the last picture's
        // seed in the box and lights up while the box holds one; pressed again it empties the box (random again).
        lateinit var seedBtn: Button
        fun seedLook() { val on = seedEt.text.isNotEmpty(); styled(seedBtn, if (on) ACCENT else FIELD, if (on) Color.rgb(0x1B, 0x12, 0x0F) else INK) }
        seedBtn = button(tr(R.string.fixed)) {
            if (seedEt.text.isNotEmpty()) seedEt.setText("") else lastSeed?.let { seedEt.setText(it.toString()) }
            seedLook() }
        seedEt.addTextChangedListener(object : android.text.TextWatcher {
            override fun afterTextChanged(e: android.text.Editable?) = seedLook()
            override fun beforeTextChanged(a: CharSequence?, b: Int, c: Int, d: Int) {}
            override fun onTextChanged(a: CharSequence?, b: Int, c: Int, d: Int) {} })
        seedLook()
        val sizeTv = note("")
        // the size under the ratio: two sliders, the width and the height (64 .. 4096 in 64s), the ratio free; the area is kept under the
        // model's largest (sizeCap: 2048x2048, Krea 1024x1024) by taking the other side down when one goes up. A ratio chip fills both
        // with its preset. For the models with fixed sizes (Anima's RoPE tables, SDXL's width, Wan) they are hidden
        fun sizeBar() = SeekBar(this).apply { max = (4096 - 256) / 64
            progressTintList = android.content.res.ColorStateList.valueOf(ACCENT); thumbTintList = android.content.res.ColorStateList.valueOf(ACCENT) }
        val sizeSb = sizeBar(); val sizeSbH = sizeBar()
        var sizeMoved: (() -> Unit)? = null
        var sizeModel = "krea"      // the model whose cap the sliders keep to (set by refresh)
        fun syncSizeBars() { val (w, h) = curSize(sizeModel, 0); sizeSb.progress = (w - 256) / 64; sizeSbH.progress = (h - 256) / 64 }
        fun sizeMoved2(widthMoved: Boolean, v: Int) {
            val cap = sizeCap(sizeModel); val cur = curSize(sizeModel, 0)
            var w = cur.first; var h = cur.second
            if (widthMoved) { w = 256 + v * 64; if (w.toDouble() * h > cap) h = ((cap / w) / 64).toInt() * 64 }
            else { h = 256 + v * 64; if (w.toDouble() * h > cap) w = ((cap / h) / 64).toInt() * 64 }
            sizeW = w.coerceAtLeast(256); sizeH = h.coerceAtLeast(256)
            syncSizeBars(); sizeMoved?.invoke()
        }
        for ((sb, isW) in listOf(sizeSb to true, sizeSbH to false)) sb.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(b: SeekBar?, v: Int, u: Boolean) { if (u) sizeMoved2(isW, v) }
            override fun onStartTrackingTouch(b: SeekBar?) {}
            override fun onStopTrackingTouch(b: SeekBar?) { prefs.edit().putInt("size_w", sizeW).putInt("size_h", sizeH).apply() } })
        // the models on this phone, always shown: with one it simply says which model draws
        val installed = (models.installed(chip!!).filter { it.isModel } + listOfNotNull(Models.Pack("yue", "YuE2", true, emptyList()).takeIf { Yue.present(models.root) && models.installed(chip!!).none { p -> p.id == "yue" } },
            Models.Pack("moss", "MOSS-Image Turbo", true, emptyList()).takeIf { MossGen.present(models.root) && models.installed(chip!!).none { p -> p.id == "moss" } },
            Models.Pack("klein", "FLUX.2 Klein 4B", true, emptyList()).takeIf { KleinGen.present(models.root) && models.installed(chip!!).none { p -> p.id == "klein" } },
            Models.Pack("klein9", "FLUX.2 Klein 9B", true, emptyList()).takeIf { KleinGen.present(models.root, "klein9_bundle") && models.installed(chip!!).none { p -> p.id == "klein9" } },
            Models.Pack("sdxl", "SDXL 1.0", true, emptyList()).takeIf { SdxlGen.present(models.root) && models.installed(chip!!).none { p -> p.id == "sdxl" } }))
            .ifEmpty { if (uiPreview || noModels) models.offered().filter { it.isModel }.ifEmpty { listOf(Models.Pack("krea", "Krea 2 Turbo", true, emptyList()), Models.Pack("anima", "Anima Turbo", true, emptyList())) }
                       else emptyList() }
        // the saved model when it can draw, else the first that can (a pack kept without its built-in model and no fine-tune cannot)
        var modelId = installed.firstOrNull { it.id == prefs.getString("model", "") && usable(it) }?.id ?: installed.firstOrNull { usable(it) }?.id ?: installed.first().id
        // a second later (the steps of a newly chosen model are set by then), not while a picture is being made
        val warmRun = Runnable { if (modelId == "anima" && !running) AnimaGen.warmMods(models.root, steps(), Advanced.tuning("anima")) { Logs.add(it) } }
        warmHook = { stepsSb.removeCallbacks(warmRun); stepsSb.postDelayed(warmRun, 1000) }
        var ratioIdx = RATIOS.indexOfFirst { it.first == prefs.getString("ratio_name", "2:3") }.coerceAtLeast(0)
        // the size and the prompt limit both depend on the model as well as the ratio
        var audioMode: ((Boolean) -> Unit)? = null      // YuE2: the image controls hidden (set once they exist, below)
        fun refresh() {
            audioMode?.invoke(modelId == "yue")
            sizeModel = modelId
            if (sizeScalable(modelId) && (sizeW <= 0 || sizeH <= 0)) { val p = sizeOf(modelId, ratioIdx); sizeW = p.first; sizeH = p.second }
            val (w, h) = curSize(modelId, ratioIdx)
            sizeSb.visibility = if (sizeScalable(modelId)) View.VISIBLE else View.GONE; sizeSbH.visibility = sizeSb.visibility
            if (sizeScalable(modelId)) syncSizeBars()
            sizeTv.text = if (modelId == "yue") tr(R.string.a_song_style_above_lyrics_below)
                else "$w×$h"
            limit = when (modelId) {   // the prompt's length limit, in characters
                "anima" -> AnimaGen.CTX - 12
                "yue" -> 3000
                "moss", "klein", "klein9" -> 500
                "sdxl" -> 300
                else -> limitFor(w, h)       // Krea 2: what the picture's tokens leave
            }
            count()
            lyricsBox.visibility = if (modelId == "yue") View.VISIBLE else View.GONE
            negBox?.visibility = if (modelId == "yue") View.GONE else View.VISIBLE
            trBtn.visibility = if (modelId == "yue" && trOrig == null) View.GONE else View.VISIBLE      // songs are not translated
            promptEt.hint = if (modelId == "yue") tr(R.string.style_tags_e_g_pop_female)
                else tr(R.string.describe_the_picture_in_english)
            if (this::memTv.isInitialized) memTv.visibility = if (modelId == "anima") View.VISIBLE else View.GONE
            val changed = loraModel != modelId
            loraModel = modelId; Advanced.kind = Advanced.kindOf(modelId); loraLabel()
            if (changed) { negEt?.let { e -> val n = Advanced.negative; if (e.text.toString() != n) e.setText(n) }; extRefill?.invoke() }
            // MOSS-Image's tokenizer read from the moment it is chosen (seconds of parsing otherwise at the first picture)
            if (changed && modelId == "moss") MossText.warm(File(models.root, "moss_bundle/tok"))
            if (changed) warmHook?.invoke()
        }
        val (ratioRow, ratioAt) = chips(RATIOS.map { it.first }, ratioIdx) { ratioIdx = it
            if (sizeScalable(modelId)) { val p = sizeOf(modelId, it); sizeW = p.first; sizeH = p.second; prefs.edit().putInt("size_w", sizeW).putInt("size_h", sizeH).apply() }
            refresh() }
        sizeMoved = { refresh() }
        // the model: one button, the model and its version ("Krea 2 · Turbo"); a tap picks the model, then the version when it has
        // more than one. In extended mode a fine-tune brought in is one of the versions, and named in place of the built-in one.
        var modelIdx = installed.indexOfFirst { it.id == modelId }.coerceAtLeast(0)
        val modelBtn = button("") {}
        fun modelLabel() { val m = installed[modelIdx]
            val c = if (m.id != "yue" && (Advanced.on || !models.hasBase(m.id))) Advanced.custom(m.id).takeIf { n -> Custom.list(models.root, m.id).any { it.name == n } }.orEmpty() else ""
            val v = c.ifEmpty { versionOf(m) }; modelBtn.text = if (v.isEmpty()) familyOf(m) else familyOf(m) + "  ·  " + v }
        modelBtn.setOnClickListener {
            if (running) return@setOnClickListener
            modelPicker(installed) { i, custom ->
                modelIdx = i; modelId = installed[i].id; if (Advanced.on || !models.hasBase(modelId)) Advanced.setCustom(modelId, custom)
                prefs.edit().putString("model", modelId).apply()       // kept, so a redraw (the Wan buttons, settings) stays on it
                // the sampling settings are kept per model and per fine-tune: the defaults and the open panel follow the one just picked
                Advanced.kind = Advanced.kindOf(modelId)
                // the model's own usual step count
                val usual = when {
                    modelId == "anima" || modelId == "krea" && Advanced.custom("krea").isNotEmpty() -> Advanced.defaults().steps
                    modelId == "klein" || modelId == "klein9" -> Advanced.defaults().steps
                    else -> 8
                }
                stepsSb.progress = (usual - 4).coerceIn(0, stepsSb.max)
                modelLabel(); refresh(); extRefill?.invoke()
            }
        }
        modelLabel()
        // in extended mode a gear beside it: bringing fine-tunes in, deleting them
        val modelRow = if (!Advanced.on) modelBtn else line(modelBtn, styled(Button(this), FIELD, INK).apply {
            text = "\u2699"; textSize = 20f; minWidth = px(50); minimumWidth = px(50); setPadding(0, 0, 0, 0)
            setOnClickListener { if (!running) modelSettings() } })
        val modelAt = { modelIdx }
        refresh()
        // how many pictures one press makes: a stepper with the number in words, not a bare box
        var count = prefs.getInt("count", 1).coerceIn(1, 99)
        // how many pictures one press makes, typed in (1-99); beside it the presets (everything on screen in one)
        val countEt = field(EditText(this).apply { setText("$count"); inputType = InputType.TYPE_CLASS_NUMBER; setSingleLine(); gravity = Gravity.CENTER
            addTextChangedListener(object : android.text.TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun afterTextChanged(e: android.text.Editable?) { count = e.toString().toIntOrNull()?.coerceIn(1, 99) ?: 1; prefs.edit().putInt("count", count).apply() }
            }) })
        val countLabel = note(tr(R.string.pictures), 14f)
        val countRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            addView(countLabel)
            addView(countEt, LinearLayout.LayoutParams(px(70), ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginStart = px(8) })
            addView(button(tr(R.string.presets)) { if (!running) presets(promptEt) },
                LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = px(10) }) }
        statusTv = text("", 13f)          // no time estimate before a run: it differed by phone and heat, and read as a promise
        bar = progress().apply { visibility = View.INVISIBLE }
        grid = BlockGrid(this, ACCENT, FIELD).apply { visibility = View.GONE; setPadding(0, px(6), 0, px(6)) }
        tempTv = note("", 12f).apply { gravity = Gravity.END; visibility = if (Advanced.on && Advanced.showHeat) View.VISIBLE else View.GONE; text = if (Advanced.on && Advanced.showHeat) heat() else "" }
        ticker.removeCallbacks(tick); ticker.post(tick)
        lockTv = note(tr(R.string.much_slower_if_the_screen)).apply { visibility = View.GONE }
        askTv = note(tr(R.string.sending_the_log_helps_even)).apply {
            visibility = View.GONE; setTextColor(ACCENT); setOnClickListener { visibility = View.GONE; bugReport() } }
        img = ImageView(this).apply { adjustViewBounds = true }
        songBtn = button(tr(R.string.play_the_song)) { toggleSong() }.apply { visibility = View.GONE }
        reportTv = note(tr(R.string.report_this_picture), 12f).apply { visibility = View.GONE; paintFlags = paintFlags or android.graphics.Paint.UNDERLINE_TEXT_FLAG; setOnClickListener { reportPicture() } }
        goBtn = primary(tr(R.string.generate)) {
            if (!running && graph != null) { runGraph(); return@primary }
            if (running) { stopRequested = true
                // at once: chosen (the default, and always outside the extended mode), or still preparing (no step yet: the text, the
                // modulations, the engine opening), or a step that would end under half way (a stop there saves nothing, so finishing it is waiting for nothing)
                val blk = !Advanced.on || Advanced.stopBlock || stepNow < 0 || (stepNow + 1) * 2 < stepsNow; if (blk) Engine.abort(true)
                status(if (blk) tr(R.string.stopping_after_this_block) else tr(R.string.stopping_at_the_end_of_this))
                return@primary }
            val model = installed[modelAt().coerceIn(0, installed.size - 1)]
            if (noModels) { status(tr(R.string.no_model_on_this_phone_yet_2)); return@primary }
            if (!usable(model)) { status(tr(R.string.no_model_to_draw_with_bring)); return@primary }       // VAE and encoders only, and no fine-tune
            val style = promptEt.text.toString().trim(); val lyrics = lyricsEt.text.toString().trim()
            // YuE2 reads one text: the style, then the lyrics from their first [section] (Yue.split); lyrics without one get [Verse].
            // No lyrics: a song with no words, given as a section holding a comma (what people typed to get one; it runs, and makeSong
            // caps such a song at 2 minutes since the model rarely ends it)
            val prompt = if (model.id != "yue") style
                         else if (style.isEmpty() && lyrics.isEmpty()) "" else style + "\n" + (if (lyrics.isEmpty()) "[Verse]\n," else if (lyrics.startsWith("[")) lyrics else "[Verse]\n$lyrics")
            if (prompt.isEmpty()) { status(if (model.id == "yue") tr(R.string.write_the_style_or_the_lyrics) else tr(R.string.write_a_prompt_first)); return@primary }
            Safety.refuse(prompt, modelId)?.let { Logs.add("prompt refused by the filter"); status(it); return@primary }
            if (model.id !in setOf("krea", "anima", "moss", "klein", "klein9", "sdxl", "yue")) { status(tr(R.string.is_not_runnable_in_this, model.name)); return@primary }
            prefs.edit().putString("prompt", style).putString("lyrics", lyrics).putInt("steps", steps()).putString("ratio_name", RATIOS[ratioAt()].first).putString("model", model.id).apply()
            val (w, h) = curSize(model.id, ratioAt())
            if (model.id == "anima" && !AnimaGen.accepts(File(models.root, "anima_bundle"), w, h)) { status(tr(R.string.ratio_not_available, model.name)); return@primary }
            val seed0 = seedEt.text.toString().toLongOrNull()
            generate(List(count) { Job(prompt, steps(), seed0?.plus(it), w, h, model.id, loraList(model.id)) })
        }
        saveBtn = button(tr(R.string.save)) { saveShown() }
        saveAllBtn = button("") { saveAll() }.apply { visibility = View.GONE }
        // more pictures than one: the arrows page through this session's pictures (swiping the picture does the same)
        pagerTv = note("", 13f).apply { gravity = Gravity.CENTER }
        pager = row(button("‹") { showShot(shotAt - 1) }, pagerTv, button("›") { showShot(shotAt + 1) }).apply { visibility = View.GONE }
        img.setOnTouchListener(object : View.OnTouchListener {
            var x0 = 0f
            override fun onTouch(v: View, e: android.view.MotionEvent): Boolean {
                when (e.actionMasked) {
                    android.view.MotionEvent.ACTION_DOWN -> x0 = e.x
                    android.view.MotionEvent.ACTION_UP -> { val dx = e.x - x0; if (Math.abs(dx) > px(60)) showShot(shotAt + if (dx < 0) 1 else -1) else v.performClick() }
                }
                return true
            }
        })
        loraBtn = button("") { if (!running) loraDialog(promptEt) }; loraLabel()
        // Krea 2 / Anima: a start picture (img2img) and how much of it is redrawn: the steps before that are skipped, the picture noised to where they end
        imgStartBtn = button(startLabel()) {
            if (running) return@button
            if (imgStart().isFile) { imgRefs().forEach { it.delete() }; imgMask().delete(); route() }
            else startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("image/*")
                .putExtra(Intent.EXTRA_ALLOW_MULTIPLE, loraModel.startsWith("klein")), PICK_IMG_START)
        }
        startThumbs = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, px(4), 0, px(4)); visibility = View.GONE }
        maskBtn = button(if (imgMask().isFile) tr(R.string.mask_painted)
                         else tr(R.string.mask_none_the_whole_picture)) {
            if (!running) maskDialog() }
        fitBtn = button(if (imgFit()) tr(R.string.mode_outpaint)
                         else tr(R.string.mode_inpaint)) {
            if (running) return@button
            prefs.edit().putBoolean("img_fit", !imgFit()).apply(); route() }
        denoiseBtn = button(tr(R.string.change_s).format(strengthOf(denoisePct()))) {
            if (running) return@button
            val o = listOf(30, 45, 60, 75, 90); prefs.edit().putInt("denoise", o[(o.indexOf(denoisePct()) + 1) % o.size]).apply(); route()
        }
        // YuE2: the song's length as chips. "Auto": the model ends the song where its lyrics and score do (nudged only from about
        // three and a half minutes on, so a song it never ends cannot run the phone out of memory); a length: the model is nudged
        // to end near it (the model's soft end), cut with a fade only at 130% of it
        val lens = listOf(0, 30, 60, 120, 180, 240)
        val (lenRow, _) = chips(listOf(tr(R.string.auto_2), tr(R.string.t_30_s), tr(R.string.t_1_min),
            tr(R.string.t_2_min), tr(R.string.t_3_min), tr(R.string.t_4_min)),
            lens.indexOf(yueSeconds()).coerceAtLeast(0), perRow = 3) { prefs.edit().putInt("yue_seconds", lens[it]).apply() }
        yueLenBtn = column(note(tr(R.string.length_of_an_instrumental_a_song)), lenRow).apply { visibility = if (loraModel == "yue") View.VISIBLE else View.GONE }
        // Anima keeps as many of its parts loaded as the phone has memory for: the line says how much there is, and a tap
        // asks the system to drop other apps' background processes
        memTv = note("", 12f).apply { setOnClickListener { freeMemory() } }; showMemory()
        // YuE2 makes a song: the screen keeps the style, the lyrics, the length, the seed and how many; the steps, the ratio and
        // the picture's own buttons go (loraLabel hides LoRA and the start image for it)
        val ratioLine = line(note(tr(R.string.aspect_ratio)), sizeTv)
        audioMode = { on -> val v = if (on) View.GONE else View.VISIBLE
            stepsTv.visibility = v; stepsSb.visibility = v; ratioLine.visibility = v; ratioRow.visibility = v
            sizeSb.visibility = if (on || !sizeScalable(modelId)) View.GONE else View.VISIBLE; sizeSbH.visibility = sizeSb.visibility
            countLabel.text = if (on) tr(R.string.songs) else tr(R.string.pictures) }
        audioMode?.invoke(modelId == "yue")
        val share = button(tr(R.string.share)) { shareLast() }
        val up = button(tr(R.string.upscale_2)) { upscale() }
        // the picture on screen as the next run's start image (img2img) or, for Klein, its reference: no trip through the gallery picker
        val reuse = button(tr(R.string.use_as_start)) {
            val f = lastFile ?: return@button
            if (runCatching { f.copyTo(imgStart(), overwrite = true); File(cacheDir, "img_ref2.img").delete(); File(cacheDir, "img_ref3.img").delete(); imgMask().delete() }.isSuccess) {
                loraLabel(); status(if (loraModel.startsWith("klein")) tr(R.string.this_picture_is_the_reference_now)
                                else tr(R.string.this_picture_is_the_start_image)) }
        }
        extraBtns = listOf(saveBtn, share, up, reuse); extraBtns.forEach { it.isEnabled = false }
        pendingSeed?.let { seedEt.setText(it.toString()); pendingSeed = null; seedLook() }
        if (!models.has("upscale", chip!!)) up.visibility = View.GONE
        statusTv.setTextColor(MUTED); refresh()
        nextKind = "main"; show(column(*listOfNotNull(
            line(LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
                    addView(ImageView(this@MainActivity).apply { setImageDrawable(packageManager.getApplicationIcon(packageName)) }, LinearLayout.LayoutParams(px(34), px(34)).apply { marginEnd = px(10) })
                    val title = text("Kroma", 24f, true); addView(title)
                    // the advanced mode's badge beside the name, as long as the mode is on: its plate stands on the name's baseline
                    // (the row centres its children, which left the badge floating above the letters)
                    if (Advanced.on) Advanced.Badge(this@MainActivity).let { b ->
                        addView(b, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                            marginStart = px(6); gravity = Gravity.BOTTOM
                            bottomMargin = (title.paint.fontMetrics.descent + title.paddingBottom - b.plateBelow()).toInt().coerceAtLeast(0)
                        })
                    } },
                button(tr(R.string.settings)) { settings() }),
            noModelPanel,
            // the prompt box with its length counter in the bottom-left corner
            *(if (nodeScreen) arrayOf(graphCard(installed.first { it.id == modelId }.id, promptEt.text.toString()), slimGo()) else arrayOf(
            card(*listOfNotNull(modelRow, android.widget.FrameLayout(this).apply {
                addView(field(promptEt).apply { setPadding(paddingLeft, paddingTop, paddingRight, px(34)) })
                addView(lenTv, android.widget.FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.START or Gravity.BOTTOM).apply { leftMargin = px(14); bottomMargin = px(10) })
                addView(trBtn, android.widget.FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.END or Gravity.BOTTOM).apply { rightMargin = px(10); bottomMargin = px(6) })
            }, lyricsBox, negBox).toTypedArray()),
            card(stepsTv, stepsSb, line(field(seedEt), seedBtn), countRow, ratioLine, ratioRow, sizeSb, sizeSbH, loraBtn!!, imgStartBtn!!, startThumbs!!, denoiseBtn!!, maskBtn!!, fitBtn!!, yueLenBtn!!))),
            *(if (nodeScreen) listOfNotNull(extendedDrawer()) else listOfNotNull(goBtn, memTv, if (Advanced.on && Advanced.showHeat) line(statusTv, tempTv) else statusTv, bar, songBtn!!, grid, img, lockTv, askTv, reportTv!!, pager, row(saveBtn, share), row(up, reuse), saveAllBtn, extendedDrawer().takeIf { Advanced.on })).toTypedArray()).toTypedArray()))
        if (!prefs.contains("auto_save")) askSaveMode()
    }

    /** node mode: the graph over the screen, Generate a thin line under it that also shows the progress, no other controls */
    internal val nodeScreen get() = Advanced.on && Advanced.nodeGraph
    internal fun status(s: String) = runOnUiThread {
        statusTv.text = s
        if (nodeScreen && this::goBtn.isInitialized) {
            if (running) goBtn.text = if (s.isEmpty()) tr(R.string.stop) else tr(R.string.stop) + "  ·  " + s
            else if (s.isNotEmpty()) android.widget.Toast.makeText(this, s, android.widget.Toast.LENGTH_SHORT).show()
        }
    }
    internal fun shareLast() { lastFile?.let { FileProvider.getUriForFile(this, "$packageName.share", it) }?.let { u -> startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).setType("image/png").putExtra(Intent.EXTRA_STREAM, u).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION), null)) } }
    /** Generate as one thin line (node mode) */
    internal fun slimGo() = goBtn.apply { minHeight = px(40); minimumHeight = px(40); textSize = 14f; setSingleLine(); ellipsize = android.text.TextUtils.TruncateAt.END }

    internal val ticker = android.os.Handler(android.os.Looper.getMainLooper())
    // the temperatures every 0.25 s while a picture is made; between runs, while the app is on screen, them and the free memory every 2 s
    // (both stayed at the run's last value, and the memory at what it was when the screen was built)
    @Volatile private var onScreen = false
    private var idleAt = 0L
    override fun onResume() { super.onResume(); onScreen = true }
    override fun onPause() { super.onPause(); onScreen = false }
    internal val tick = object : Runnable {
        override fun run() {
            val heatOn = Advanced.on && Advanced.showHeat && this@MainActivity::tempTv.isInitialized
            if (running && this@MainActivity::bar.isInitialized) {
                if (heatOn) tempTv.text = heat()
            } else if (onScreen && android.os.SystemClock.uptimeMillis() - idleAt >= 2000) {
                idleAt = android.os.SystemClock.uptimeMillis()
                if (heatOn) tempTv.text = heat()
                showMemory()
            }
            ticker.postDelayed(this, 250)
        }
    }
    /** Extended mode: the NPU's temperature (the hottest nspss thermal zone, where the phone has them readable) and the battery's. */
    // the NPU's zones (type nsp*) found once: walking every thermal zone's name each 0.25 s was most of this
    private val npuZones: List<File> by lazy { runCatching { File("/sys/class/thermal").listFiles()?.filter { z ->
        runCatching { File(z, "type").readText().trim() }.getOrNull()?.startsWith("nsp") == true }?.map { File(it, "temp") } }.getOrNull().orEmpty() }
    internal fun heat(): String {
        val npu = npuZones.mapNotNull { f -> runCatching { f.readText().trim().toInt() / 1000f }.getOrNull() }.maxOrNull()
        val bat = registerReceiver(null, android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED))?.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        return listOfNotNull(npu?.let { "NPU %.0f°C".format(it) }, if (bat >= 0) tr(R.string.battery) + " %.0f°C".format(bat / 10f) else null).joinToString(" · ")
    }
    internal fun hhmm(sec: Long) = if (sec >= 60) tr(R.string.m_s, sec / 60, sec % 60) else tr(R.string.s, sec)

    /** One picture to make. seed null = a random one. */
    data class Job(val prompt: String, val steps: Int, val seed: Long?, val w: Int, val h: Int, val model: String = "krea",
                   val loras: List<Pair<String, Int>> = emptyList())

    internal var loraRefill: (() -> Unit)? = null

    /** A picked .safetensors: copied into the cache, converted for the model on screen, and selected. */
    @Deprecated("the platform's simple result callback is enough for one picker")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (requestCode == PICK_MODEL && resultCode == RESULT_OK && uri != null) { importModel(uri); return }
        if (requestCode == PICK_GRAPH_IMAGE && resultCode == RESULT_OK && uri != null) {
            // a Start image node's picture: a copy in the app's files (the graph keeps its path)
            val g = graph; val n = g?.node(graphImageNode)
            if (n != null) runCatching {
                val f = File(filesDir, "graph_images/${n.id}_${System.currentTimeMillis()}.img").apply { parentFile?.mkdirs() }
                contentResolver.openInputStream(uri)?.use { i -> f.outputStream().use { o -> i.copyTo(o) } }; upright(f)
                n.p["path"]?.let { File(it) }?.takeIf { it.isFile && it.parentFile?.name == "graph_images" }?.delete()
                n.p["path"] = f.absolutePath; Graph.save(graphFile(), g)
            }.onFailure { Logs.add("graph image: ${it.message}") }
            route(); return
        }
        if (requestCode == PICK_META && resultCode == RESULT_OK && uri != null) {
            val png = runCatching { contentResolver.openInputStream(uri)?.use { it.readBytes() } }.getOrNull()
            val text = png?.let { PngMeta.read(it) }
            if (text == null) status(tr(R.string.no_kroma_settings_in_that_picture))
            else applyPictureMeta(text)
            return
        }
        if (requestCode == PICK_IMG_START && resultCode == RESULT_OK && (uri != null || data?.clipData != null)) {
            imgMask().delete()           // a mask belongs to the picture it was painted on
            // Klein takes up to three references (a multiple pick): the first in imgStart, the others after it (imgRefs)
            val uris = data?.clipData?.let { c -> (0 until c.itemCount).map { c.getItemAt(it).uri } } ?: listOfNotNull(uri)
            val dst = listOf(imgStart(), File(cacheDir, "img_ref2.img"), File(cacheDir, "img_ref3.img")); dst.forEach { it.delete() }
            for ((i, u) in uris.take(3).withIndex()) runCatching { contentResolver.openInputStream(u)?.use { inp -> dst[i].outputStream().use { o -> inp.copyTo(o) } }; upright(dst[i]) }
                .onFailure { Logs.add("start image ${i + 1}: ${it.message}") }
            Logs.add("start image(s) set: ${imgRefs().size}")
            route(); return
        }
        if (requestCode != PICK_LORA || resultCode != RESULT_OK || uri == null) return
        val shown = runCatching { contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c -> if (c.moveToFirst()) c.getString(0) else null } }.getOrNull() ?: "lora"
        val name = shown.substringBeforeLast('.').replace(Regex("[^\\p{L}\\p{N} ._-]"), "_").trim().take(40).ifEmpty { "lora" }
        status(tr(R.string.converting_the_lora))
        thread { importLora(name) { tmp -> contentResolver.openInputStream(uri)!!.use { i -> tmp.outputStream().use { o -> i.copyTo(o) } } } }
    }

    /** The last picture the filter blurred (never one it discarded), held in memory only: picture, file name, metadata. */
    @Volatile internal var blurred: Triple<Bitmap, String, String>? = null

    /** Whether pictures the filter flagged are shown as they are: a setting, and only in the edition that has it. */
    internal fun showsAdult() = BuildConfig.ADULT_OPTION && prefs.getBoolean("adult", false)

    internal fun availMb(): Long = android.app.ActivityManager.MemoryInfo().also { getSystemService(android.app.ActivityManager::class.java).getMemoryInfo(it) }.availMem shr 20
    internal fun showMemory() { if (this::memTv.isInitialized) memTv.text = tr(R.string.free_memory, "%.1f GB".format(availMb() / 1024.0)) + (if (BuildConfig.CLOSE_APPS) "  ·  " + tr(R.string.close_other_apps) else "") }

    /**
     * Asks the system to end the background processes of the other apps with a launcher icon. It is the most an ordinary app
     * may do (nothing running in the foreground or as a service is touched), and what it gives depends on the phone.
     */
    internal fun freeMemory() {
        if (running || !BuildConfig.CLOSE_APPS) return
        val before = availMb()
        thread {
            val am = getSystemService(android.app.ActivityManager::class.java)
            val pkgs = packageManager.queryIntentActivities(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0).map { it.activityInfo.packageName }.toSet() - packageName
            pkgs.forEach { runCatching { am.killBackgroundProcesses(it) } }
            Thread.sleep(1500)
            val after = availMb(); Logs.add("memory: ${pkgs.size} apps asked to stop, ${before}MB -> ${after}MB available")
            runOnUiThread { showMemory() }
            status(tr(R.string.memory_freed, "%.1f GB".format(before / 1024.0), "%.1f GB".format(after / 1024.0)) + "\n" + tr(R.string.more_memory_is_faster))
        }
    }

    internal val pagerReady get() = ::pager.isInitialized     /* for the files the activity is split into (isInitialized needs the class) */
    internal val saveBtnReady get() = ::saveBtn.isInitialized     /* for the files the activity is split into (isInitialized needs the class) */

    companion object {
        internal const val PICK_LORA = 41
        internal const val PICK_MODEL = 42
        internal const val PICK_META = 46
        internal const val PICK_IMG_START = 45
        internal const val PICK_GRAPH_IMAGE = 44
        const val REPO_URL = "https://github.com/Pixfanthav/Kroma"     // the public source repository
        // Krea 2 Community License 3.1(b): a distributed model's name starts with "Krea". One name per chip:
        // each chip's set is its own compiled model.
        fun modelName(c: Chip, id: String = "krea", fmt: String = "fp16") = (ModelKind.of(id) ?: ModelKind.KREA).full + " (for ${c.npu} NPU, $fmt)"
    }
}
