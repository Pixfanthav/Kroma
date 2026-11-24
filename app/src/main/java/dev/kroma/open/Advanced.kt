package dev.kroma.open

import android.content.Context
import android.content.SharedPreferences
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Typeface
import android.view.View

/**
 * The advanced mode: off by default, turned on in Settings. When on, a badge stands beside the app's name and the settings
 * below take effect; when off the app runs exactly as it always has, whatever is stored here.
 */
object Advanced {
    /** What the badge says. One place, so the name can change without touching anything else. */
    const val BADGE = "EX"

    val SAMPLERS = listOf("er_sde", "euler", "euler_ancestral", "dpmpp_2m")
    /** MOSS-Image: HaDeS-MOSS-Turbo (DeVilDoNia, Apache-2.0, LoRAs on the SFT: "turbo8" 8-step v0.7, "turbo4" 4-step v0.6; euler, simple,
     *  shift 5, CFG 1 fixed) first; then the shared flow samplers for the model as it was released (SFT, or a fine-tune of it: CFG 3, shift 5, 30-50 steps) */
    val SAMPLERS_MOSS = listOf("turbo8", "turbo4") + SAMPLERS
    fun samplers() = if (kind == "moss") SAMPLERS_MOSS else SAMPLERS
    val SCHEDULERS = Sched.NAMES
    /** Krea 2: its own schedule (the turbo's fixed sigmas, "krea") first, then the shared ones */
    val SCHEDULERS_KREA = listOf("krea") + Sched.NAMES
    fun schedulers() = if (kind == "krea") SCHEDULERS_KREA else SCHEDULERS
    /** a schedule's name on screen: "krea" is Krea 2 Turbo's own fixed schedule (the one it was distilled on) */
    fun schedLabel(n: String) = if (n == "krea") tr(R.string.krea_turbo) else n
    /** the model the sampling settings below are for: "anima", "krea", "moss", "klein", "klein9" or "sdxl" (each keeps its own) */
    @Volatile var kind = "anima"
    /** the settings kind a model on screen uses: its own id, Anima's plain one for the rest */
    fun kindOf(model: String) = if (model in setOf("krea", "moss", "klein", "klein9", "sdxl")) model else "anima"

    private lateinit var p: SharedPreferences
    fun init(c: Context) { p = c.getSharedPreferences("kroma_advanced", Context.MODE_PRIVATE) }

    var on: Boolean get() = p.getBoolean("on", false); set(v) { p.edit().putBoolean("on", v).apply() }
    /** A model's own sampling settings: the built-in Anima is a turbo model; a fine-tune's come from its defaults.txt (written when it
     *  is brought in, from what its page's sample pictures used), else from its name ("turbo" or not). */
    class Defaults(val sampler: String, val scheduler: String, val shift: Float, val cfg: Float, val steps: Int)
    val TURBO = Defaults("er_sde", "simple", 3f, 1f, 10)
    val BASE = Defaults("er_sde", "simple", 3f, 4f, 28)
    /** a fine-tune's defaults.txt, read by the activity (it knows where the models are); null when it has none */
    var defaultsOf: (String) -> Defaults? = { null }
    val KREA_TURBO = Defaults("euler", "krea", 3f, 1f, 8)
    val KREA_BASE = Defaults("euler", "simple", 3f, 3.5f, 28)
    val MOSS_TURBO = Defaults("turbo8", "simple", 5f, 1f, 8)
    /** FLUX.2 Klein: the built-in models are distilled (4 steps, one pass); a base (not distilled) fine-tune wants CFG and more steps */
    val KLEIN_TURBO = Defaults("euler", "simple", 3f, 1f, 4)
    val KLEIN_BASE = Defaults("euler", "simple", 3f, 4f, 25)
    /** SDXL 1.0 as SdxlGen runs it: its own Euler loop, cfg 7, at least 20 steps (only the negative prompt and cfg are read from here) */
    val SDXL_DEFAULT = Defaults("euler", "simple", 3f, 7f, 20)
    fun defaults(): Defaults {
        if (kind == "krea") { val m = custom("krea"); return if (m.isEmpty()) KREA_TURBO else defaultsOf(m) ?: if (m.contains("turbo", true)) KREA_TURBO else KREA_BASE }
        if (kind == "klein" || kind == "klein9") { val m = custom(kind); return if (m.isEmpty()) KLEIN_TURBO else defaultsOf(m) ?: if (m.contains("base", true)) KLEIN_BASE else KLEIN_TURBO }
        if (kind == "sdxl") return SDXL_DEFAULT
        // MOSS-Image: the turbo for the built-in model and a fine-tune alike (its LoRA goes on top of either); Raw sampling when asked
        if (kind == "moss") { val m = custom("moss"); return if (m.isEmpty()) MOSS_TURBO else defaultsOf(m) ?: MOSS_TURBO }
        val m = custom("anima"); if (m.isEmpty()) return TURBO
        return defaultsOf(m) ?: if (m.contains("turbo", true)) TURBO else BASE
    }
    /** the settings below are kept per Anima that draws: the built-in one under the plain keys (as before), a fine-tune's beside them */
    private fun key(n: String) = if (kind == "krea" || kind == "moss" || kind == "klein" || kind == "klein9" || kind == "sdxl") custom(kind).let { if (it.isEmpty()) "$kind.$n" else "$kind.$n@$it" }
                                 else custom("anima").let { if (it.isEmpty()) n else "$n@$it" }
    var sampler: String get() = p.getString(key("sampler"), null)?.takeIf { it in samplers() } ?: defaults().sampler; set(v) { p.edit().putString(key("sampler"), v).apply() }
    var scheduler: String get() = p.getString(key("scheduler"), null)?.takeIf { it in schedulers() } ?: defaults().scheduler; set(v) { p.edit().putString(key("scheduler"), v).apply() }
    var shift: Float get() = if (p.contains(key("shift"))) p.getFloat(key("shift"), 3f) else defaults().shift; set(v) { p.edit().putFloat(key("shift"), v).apply() }
    var cfg: Float get() = if (p.contains(key("cfg"))) p.getFloat(key("cfg"), 1f) else defaults().cfg; set(v) { p.edit().putFloat(key("cfg"), v).apply() }
    var negative: String get() = p.getString(key("negative"), "")!!; set(v) { p.edit().putString(key("negative"), v).apply() }
    /** back to the model's own settings */
    fun resetSampling() { p.edit().remove(key("sampler")).remove(key("scheduler")).remove(key("shift")).remove(key("cfg")).remove(key("negative")).apply() }
    /** The node graph in place of the prompt and settings cards (extended mode). */
    var nodeGraph: Boolean get() = p.getBoolean("node_graph", false); set(v) { p.edit().putBoolean("node_graph", v).apply() }
    /** how chosen LoRAs reach the blocks (Engine's loraMode): "run", "bake" (copy kept) or "bake_once" (copy removed after) */
    var loraMode: String get() = p.getString("lora_mode", "run") ?: "run"; set(v) { p.edit().putString("lora_mode", v).apply() }
    /** what a fine-tune is converted to: "f16" (as the model was made), "q8" (half the size; the 8-bit choice) or "q4" (a quarter, lower quality) */
    var convertFormat: String get() = (p.getString("convert_format", "f16") ?: "f16").let { if (it == "i8") "q8" else it }; set(v) { p.edit().putString("convert_format", v).apply() }
    /** the fine-tune (Custom) that draws for a model ("anima" / "krea"), "" for the bundle's own */
    fun custom(kind: String) = p.getString("custom_$kind", "") ?: ""
    fun setCustom(kind: String, name: String) { p.edit().putString("custom_$kind", name).apply() }
    /** Show the NPU and battery temperatures beside the progress line (off unless asked: a busy NPU reads 90°C, which alarms). */
    var showHeat: Boolean get() = p.getBoolean("show_heat", false); set(v) { p.edit().putBoolean("show_heat", v).apply() }
    /** Stop leaves at the next block, the step thrown away (off: at the step's end, which a picture past half the steps survives) */
    var stopBlock: Boolean get() = p.getBoolean("stop_block", true); set(v) { p.edit().putBoolean("stop_block", v).apply() }
    /** the engine's per-block diagnosis to the log (largest value, NaN / inf after each block; slower): a kdebug file in the engine's
     *  weights folder while on, which the engine checks each step (Anima's path), for a bug report that says which block.
     *  On by default on the chips offered only as a trial (8s, 7+: the log is what the maintainers learn from), off elsewhere; the user's own choice wins */
    var diagLog: Boolean get() = p.getBoolean("diag_log", Chip.detect()?.animaOnly == true); set(v) { p.edit().putBoolean("diag_log", v).apply() }

    /** The sampling settings in effect for one picture: the defaults when the mode is off. */
    class Tuning(val sampler: String, val scheduler: String, val shift: Double, val cfg: Double, val negative: String) {
        val usesCfg get() = cfg > 1.0001
        override fun toString() = "sampler $sampler, scheduler $scheduler, shift $shift, cfg $cfg${if (usesCfg) " (negative ${negative.length} chars)" else ""}"
    }
    val DEFAULT = Tuning("er_sde", "simple", 3.0, 1.0, "")
    /** the settings for model [k] ("anima" / "krea" / "moss") */
    fun tuning(k: String): Tuning { val was = kind; kind = k; try { return tuning() } finally { kind = was } }
    /** off: the model's own settings (a fine-tune only draws in the mode, so that is the built-in turbo one) */
    fun tuning() = if (!on) defaults().let { Tuning(it.sampler, it.scheduler, it.shift.toDouble(), it.cfg.toDouble(), "") } else Tuning(sampler, scheduler, shift.toDouble(), cfg.toDouble(), negative)

    /**
     * The badge, drawn from the same shapes as the design (an SVG of 66 x 26 units): a slanted dark plate, an accent band down the
     * left edge, a row of halftone dots along the bottom growing from left to right (noise settling into a picture), the word in
     * white, and an orange outline.
     */
    class Badge(c: Context) : View(c) {
        private val plate = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0x17, 0x18, 0x1C) }
        private val accent = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0xFF, 0x6B, 0x3D) }
        private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0xFF, 0x6B, 0x3D); style = Paint.Style.STROKE; strokeWidth = 1.2f; strokeJoin = Paint.Join.ROUND }
        private val word = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 13f; textAlign = Paint.Align.CENTER
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD); letterSpacing = 0.04f; isFakeBoldText = true }
        private fun poly(vararg xy: Float) = Path().apply { moveTo(xy[0], xy[1]); var i = 2; while (i < xy.size) { lineTo(xy[i], xy[i + 1]); i += 2 }; close() }
        // The design is 66 units wide for a three-letter word; a longer word widens the plate by its extra width, the slanted ends
        // and the left band staying as they are and the dots spreading along the bottom.
        private val extra = maxOf(0f, word.measureText(BADGE) - word.measureText("PRO"))
        private val wide = 66f + extra
        private fun x(v: Float) = if (v > 30f) v + extra else v              // points of the right half move right
        private val frame = poly(8f, 1f, x(63f), 1f, x(57.9f), 17f, x(52f), 23f, 1f, 23f)
        private val band = poly(8.6f, 1.8f, 11.4f, 1.8f, 5.55f, 20.2f, 7f, 20.2f, 6.3f, 22.3f, 2.08f, 22.3f)
        private val radii = floatArrayOf(0.3f, 0.38f, 0.46f, 0.54f, 0.62f, 0.7f, 0.76f, 0.82f, 0.88f, 0.94f, 1f, 1f, 1f, 1f)
        private val dotStep = (x(48f) - 9f) / (radii.size - 1)
        /** Height in dp: about two thirds of the 24sp title beside it. */
        private val heightDp = 17f
        /** Pixels between the plate's bottom edge and the view's bottom (the outline's margin), for lining it up with text. */
        fun plateBelow() = heightDp * resources.displayMetrics.density * 2f / 26f
        override fun onMeasure(w: Int, h: Int) {
            val hh = (heightDp * resources.displayMetrics.density).toInt(); setMeasuredDimension((hh * (wide + 1f) / 26f).toInt() + 2, hh)
        }
        /** Where the word is drawn so its slanted letters sit in the middle of the plate's open part: the letters' own outline,
         *  slanted as drawn, centred between the accent band's right edge and the plate's right edge at the letters' mid-height
         *  (the plate is slanted and banded, so the middle of its width is not where the eye puts its middle). */
        private val textX: Float by lazy {
            val p = Path(); word.getTextPath(BADGE, 0, BADGE.length, 0f, 17f, p)
            p.transform(android.graphics.Matrix().apply { setSkew(-0.32f, 0f) })
            val b = android.graphics.RectF(); p.computeBounds(b, true)
            val ym = (b.top + b.bottom) / 2
            val left = 11.4f - (ym - 1.8f) * (5.85f / 18.4f)                      // the band's right edge, (11.4, 1.8) to (5.55, 20.2)
            val right = x(63f) - (ym - 1f) * (5.1f / 16f)                       // the plate's right edge, (63, 1) to (57.9, 17)
            (left + right) / 2 - (b.left + b.right) / 2                          // the text drawn centred at 0 has its ink centre at (b.left + b.right) / 2
        }
        override fun onDraw(c: Canvas) {
            val s = height / 26f
            c.save(); c.scale(s, s); c.translate(1f, 1f)
            c.drawPath(frame, plate); c.drawPath(band, accent)
            radii.forEachIndexed { i, r -> c.drawCircle(9f + i * dotStep, 21.3f, r, accent) }
            c.save(); c.skew(-0.32f, 0f); c.drawText(BADGE, textX, 17f, word); c.restore()
            c.drawPath(frame, line)
            c.restore()
        }
    }
}
