package dev.kroma.open

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.MotionEvent
import android.view.View
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.hypot

/**
 * Extended mode's node graph: the picture's recipe as nodes and wires. It runs through the same generator as the plain screen --
 * the graph is turned into one job (model, prompt, negative, LoRA, size, sampler settings) and what follows it (upscale) -- so
 * a graph and the plain controls make the same pictures.
 *
 * Types on the wires: MODEL, COND (text), LATENT (an empty picture of a size, or a sampled one), IMAGE.
 */
object Graph {
    enum class T { MODEL, COND, LATENT, IMAGE }
    class Kind(val id: String, val title: () -> String, val ins: List<Pair<String, T>>, val outs: List<Pair<String, T>>, val params: Map<String, String>)

    val KINDS = listOf(
        Kind("model", { tr(R.string.model) }, emptyList(), listOf("model" to T.MODEL), mapOf("model" to "anima", "custom" to "")),
        Kind("lora", { "LoRA" }, listOf("model" to T.MODEL), listOf("model" to T.MODEL), mapOf("name" to "", "strength" to "100")),
        Kind("prompt", { tr(R.string.prompt) }, emptyList(), listOf("cond" to T.COND), mapOf("text" to "")),
        Kind("negative", { tr(R.string.negative) }, emptyList(), listOf("cond" to T.COND), mapOf("text" to "")),
        Kind("size", { tr(R.string.size_2) }, emptyList(), listOf("latent" to T.LATENT), mapOf("w" to "832", "h" to "1216")),
        // a picture to start from (ComfyUI's Load Image)
        Kind("startimg", { tr(R.string.start_image) }, emptyList(), listOf("image" to T.IMAGE), mapOf("path" to "")),
        // img2img (Krea 2, Anima): the start image as a latent of the size given (cut to it in the middle, ComfyUI's VAE Encode);
        // the Sampler's denoise says how much of it is redrawn
        Kind("vaeenc", { tr(R.string.vae_encode) }, listOf("image" to T.IMAGE), listOf("latent" to T.LATENT), mapOf("w" to "832", "h" to "1216")),
        Kind("sampler", { tr(R.string.sampler) },
            // image (optional): a start picture straight in: img2img for Krea 2 / Anima with the denoise below
            listOf("model" to T.MODEL, "positive" to T.COND, "negative" to T.COND, "latent" to T.LATENT, "image" to T.IMAGE), listOf("latent" to T.LATENT),
            mapOf("steps" to "10", "cfg" to "1.0", "seed" to "", "sampler" to "er_sde", "scheduler" to "simple", "shift" to "3.0", "count" to "1", "denoise" to "1.0")),
        Kind("vae", { tr(R.string.vae_decode) }, listOf("latent" to T.LATENT), listOf("image" to T.IMAGE), emptyMap()),
        Kind("upscale", { tr(R.string.upscale_2_2) }, listOf("image" to T.IMAGE), listOf("image" to T.IMAGE), emptyMap()),
        Kind("save", { tr(R.string.image) }, listOf("image" to T.IMAGE), emptyList(), emptyMap()))
    fun kind(id: String) = KINDS.first { it.id == id }

    class Node(val id: Int, val kind: String, var x: Float, var y: Float, val p: MutableMap<String, String>)
    class Wire(val from: Int, val out: Int, val to: Int, val inp: Int)
    class G(val nodes: MutableList<Node> = mutableListOf(), val wires: MutableList<Wire> = mutableListOf()) {
        fun add(kind: String, x: Float, y: Float) = Node((nodes.maxOfOrNull { it.id } ?: 0) + 1, kind, x, y, kind(kind).params.toMutableMap()).also { nodes.add(it) }
        fun remove(n: Node) { nodes.remove(n); wires.removeAll { it.from == n.id || it.to == n.id } }
        /** one wire into an input: a new one replaces the old */
        fun connect(from: Node, out: Int, to: Node, inp: Int) {
            if (kind(from.kind).outs[out].second != kind(to.kind).ins[inp].second || from === to) return
            wires.removeAll { it.to == to.id && it.inp == inp }; wires.add(Wire(from.id, out, to.id, inp))
        }
        fun node(id: Int) = nodes.firstOrNull { it.id == id }
        fun input(n: Node, inp: Int) = wires.firstOrNull { it.to == n.id && it.inp == inp }?.let { node(it.from) }
        fun json() = JSONObject().put("nodes", JSONArray().apply { nodes.forEach { n -> put(JSONObject().put("id", n.id).put("kind", n.kind).put("x", n.x.toDouble()).put("y", n.y.toDouble()).put("p", JSONObject(n.p as Map<*, *>))) } })
            .put("wires", JSONArray().apply { wires.forEach { w -> put(JSONObject().put("from", w.from).put("out", w.out).put("to", w.to).put("in", w.inp)) } })
    }
    fun parse(o: JSONObject) = G().apply {
        val ns = o.getJSONArray("nodes")
        for (i in 0 until ns.length()) ns.getJSONObject(i).let { j ->
            if (KINDS.none { it.id == j.getString("kind") }) return@let     // a kind this app no longer has (Wan's latent): left out
            val p = j.getJSONObject("p"); val m = kind(j.getString("kind")).params.toMutableMap(); p.keys().forEach { k -> m[k] = p.getString(k) }
            nodes.add(Node(j.getInt("id"), j.getString("kind"), j.getDouble("x").toFloat(), j.getDouble("y").toFloat(), m)) }
        val ws = o.getJSONArray("wires")
        for (i in 0 until ws.length()) ws.getJSONObject(i).let { j -> if (node(j.getInt("from")) != null && node(j.getInt("to")) != null) wires.add(Wire(j.getInt("from"), j.getInt("out"), j.getInt("to"), j.getInt("in"))) }
    }
    /** the plain screen's settings, which a new graph starts from (so turning the graph on keeps what was set) */
    class Start(val model: String, val custom: String, val prompt: String, val negative: String, val w: Int, val h: Int, val steps: Int,
                val seed: String, val count: Int, val loras: List<Pair<String, Int>>, val sampler: String, val scheduler: String, val cfg: Float, val shift: Float)
    /** the graph a first look starts from: everything the plain screen does, wired, with its values */
    fun starter(s: Start) = G().apply {
        val f = { v: Float -> "%.1f".format(java.util.Locale.ROOT, v) }
        val m = add("model", 30f, 40f).apply { p["model"] = s.model; p["custom"] = s.custom }
        var mo: Node = m
        s.loras.forEachIndexed { i, (nm, pc) ->
            val prev = mo; mo = add("lora", 30f + i * 20f, 170f + i * 150f).apply { p["name"] = nm; p["strength"] = "$pc" }.also { connect(prev, 0, it, 0) }
        }
        val pr = add("prompt", 30f, 360f).apply { p["text"] = s.prompt }
        val ng = add("negative", 30f, 540f).apply { p["text"] = s.negative }
        val sz = add("size", 30f, 720f).apply { p["w"] = "${s.w}"; p["h"] = "${s.h}" }
        val sa = add("sampler", 520f, 200f).apply { p["steps"] = "${s.steps}"; p["seed"] = s.seed; p["count"] = "${s.count}"
            p["sampler"] = s.sampler; p["scheduler"] = s.scheduler; p["cfg"] = f(s.cfg); p["shift"] = f(s.shift) }
        val va = add("vae", 520f, 640f); val sv = add("save", 520f, 840f)
        connect(mo, 0, sa, 0); connect(pr, 0, sa, 1); connect(ng, 0, sa, 2); connect(sz, 0, sa, 3); connect(sa, 0, va, 0); connect(va, 0, sv, 0)
    }
    fun load(f: File, fallback: () -> G) = runCatching { parse(JSONObject(f.readText())) }.getOrNull()?.takeIf { it.nodes.isNotEmpty() } ?: fallback()
    fun save(f: File, g: G) = runCatching { f.parentFile?.mkdirs(); f.writeText(g.json().toString(1)) }

    /** What a graph asks for: the job's settings and what follows the picture. Null and a reason when something needed is not wired. */
    class Plan(val model: String, val prompt: String, val negative: String, val loras: List<Pair<String, Int>>, val w: Int, val h: Int,
               val steps: Int, val cfg: Double, val seed: Long?, val sampler: String, val scheduler: String, val count: Int, val upscale: Boolean,
               val custom: String = "", val shift: Double = 3.0, val start: String = "", val denoise: Double = 1.0)
    fun plan(g: G): Pair<Plan?, String?> {
        val save = g.nodes.firstOrNull { it.kind == "save" } ?: return null to tr(R.string.add_an_image_node)
        var img = g.input(save, 0); var upscale = false
        if (img?.kind == "upscale") { upscale = true; img = g.input(img, 0) }
        if (img?.kind != "vae") return null to tr(R.string.the_image_node_needs_a_vae)
        val sa = g.input(img, 0)?.takeIf { it.kind == "sampler" } ?: return null to tr(R.string.vae_decode_needs_a_sampler)
        // every LoRA node between the Sampler and the Model, in the order they are wired (nearest the model first)
        var mn = g.input(sa, 0); val chain = ArrayList<Node>()
        while (mn?.kind == "lora") { chain.add(0, mn); mn = g.input(mn, 0) }
        val loras = chain.mapNotNull { n -> n.p["name"]?.takeIf { it.isNotEmpty() }?.let { it to (n.p["strength"]?.toIntOrNull() ?: 100) } }
        if (mn?.kind != "model") return null to tr(R.string.the_sampler_needs_a_model)
        val pos = g.input(sa, 1)?.p?.get("text")?.trim().orEmpty()
        if (pos.isEmpty()) return null to tr(R.string.the_sampler_needs_a_prompt)
        val sz = g.input(sa, 3)
        // a start image on the Sampler's own input wins over one through VAE encode
        val direct = g.input(sa, 4)?.takeIf { it.kind == "startimg" }?.p?.get("path").orEmpty()
        val start = direct.ifEmpty { if (sz?.kind == "vaeenc") g.input(sz!!, 0)?.takeIf { it.kind == "startimg" }?.p?.get("path").orEmpty() else "" }
        if (sz?.kind == "vaeenc" && start.isEmpty()) return null to tr(R.string.vae_encode_needs_a_start_image)
        return Plan(mn.p["model"] ?: "anima", pos, g.input(sa, 2)?.p?.get("text").orEmpty(), loras,
            sz?.p?.get("w")?.toIntOrNull() ?: 832, sz?.p?.get("h")?.toIntOrNull() ?: 1216,
            sa.p["steps"]?.toIntOrNull()?.coerceIn(1, 50) ?: 10, sa.p["cfg"]?.toDoubleOrNull() ?: 1.0, sa.p["seed"]?.toLongOrNull(),
            sa.p["sampler"] ?: "er_sde", sa.p["scheduler"] ?: "simple", sa.p["count"]?.toIntOrNull()?.coerceIn(1, 99) ?: 1, upscale,
            mn.p["custom"].orEmpty(), sa.p["shift"]?.toDoubleOrNull() ?: 3.0, start, sa.p["denoise"]?.toDoubleOrNull()?.coerceIn(0.05, 1.0) ?: 1.0) to null
    }
}

/** a LoRA strength kept in hundredths, as people write it: 1.0, 0.65, 0.7 */
fun strengthOf(pct: Int) = "%.2f".format(java.util.Locale.ROOT, pct / 100.0).trimEnd('0').let { if (it.endsWith(".")) it + "0" else it }

/** The graph on a canvas: drag a node by its title, tap a node to edit it, tap an output dot then an input dot to wire them, drag
 *  the empty space to pan, pinch (or the + / - of [zoomBy]) to zoom. */
@SuppressLint("ViewConstructor")
class GraphView(ctx: Context, var g: Graph.G, private val dp: Float, private val ink: Int, private val muted: Int, private val card: Int, private val accent: Int,
                private val onEdit: (Graph.Node) -> Unit, private val onChange: () -> Unit) : View(ctx) {
    /** a long press on the empty canvas, with the place in graph units (the node menu opens there) */
    var onLongPress: ((Float, Float) -> Unit)? = null
    private var pressX = 0f; private var pressY = 0f
    private val longPress = Runnable { if (!moved && !pinched && drag == null) { panning = false; onLongPress?.invoke((pressX - panX) / zs, (pressY - panY) / zs) } }
    private val W = 220f; private val HEAD = 30f; private val ROW = 26f; private val R = 7f
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG); private val text = Paint(Paint.ANTI_ALIAS_FLAG)
    private val wire = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private var panX = 0f; private var panY = 0f; private var zoom = 1f
    /** pixels per graph unit at the current zoom */
    private val zs get() = dp * zoom
    private var drag: Graph.Node? = null; private var dragDX = 0f; private var dragDY = 0f; private var panning = false; private var lastX = 0f; private var lastY = 0f
    private var downX = 0f; private var downY = 0f; private var moved = false; private var pinched = false
    private var pending: Pair<Graph.Node, Int>? = null                               // an output tapped or dragged, waiting for its input
    private var pendIn: Pair<Graph.Node, Int>? = null                                // the same the other way round: an input dragged towards an output
    private var wiring = false; private var wireX = 0f; private var wireY = 0f       // a wire being dragged, and where the finger is
    /** the kind of node at work during a run ("sampler", "vae", "upscale", "save"; null when idle), and its progress 0..1 (-1: none) */
    var active: String? = null; set(v) { field = v; postInvalidate() }
    var progress = -1f; set(v) { field = v; postInvalidate() }
    /** the last picture, shown inside the Image node (the live preview during a run, then the result) */
    var picture: android.graphics.Bitmap? = null; set(v) { field = v; postInvalidate() }
    /** while a picture is being made: no wire comes out and no node goes (a free input can still be wired; moving, zooming and looking work) */
    var locked = false
    /** zoom and pan so every node is on screen (the first layout, and the Start over button) */
    fun fit() {
        if (width == 0 || g.nodes.isEmpty()) { fitted = false; return }
        val l = g.nodes.minOf { it.x }; val t = g.nodes.minOf { it.y }; val r = g.nodes.maxOf { it.x + W }; val b = g.nodes.maxOf { it.y + 200f }
        zoom = minOf(width / ((r - l + 40f) * dp), height / ((b - t + 40f) * dp), 1.2f).coerceIn(0.35f, 2.5f)
        panX = (width - (r - l) * zs) / 2 - l * zs; panY = 20f * dp - t * zs + 0f; fitted = true; invalidate()
    }
    private var fitted = false
    override fun onSizeChanged(w: Int, h: Int, ow: Int, oh: Int) { super.onSizeChanged(w, h, ow, oh); if (!fitted) fit() }
    /** where a new node goes: the middle of what is on screen, in graph units */
    fun centre() = ((width / 2f - panX) / zs - W / 2) to ((height / 2f - panY) / zs - 40f)
    private fun thumbH(n: Graph.Node) = picture?.takeIf { n.kind == "save" }?.let { minOf(320f, (W - 20f) * it.height / it.width) + 10f } ?: 0f
    private val color = mapOf(Graph.T.MODEL to Color.rgb(0xB3, 0x9D, 0xDB), Graph.T.COND to Color.rgb(0xF2, 0xC1, 0x4E), Graph.T.LATENT to Color.rgb(0xF0, 0x7F, 0x9A), Graph.T.IMAGE to Color.rgb(0x6F, 0xC3, 0xDF))

    /** zoom by [f] about the point (fx, fy) of the view (the middle when not given), kept between 0.35x and 2.5x */
    fun zoomBy(f: Float, fx: Float = width / 2f, fy: Float = height / 2f) {
        val nz = (zoom * f).coerceIn(0.35f, 2.5f); val r = nz / zoom
        panX = fx - (fx - panX) * r; panY = fy - (fy - panY) * r; zoom = nz; invalidate()
    }
    private val scaler = android.view.ScaleGestureDetector(ctx, object : android.view.ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: android.view.ScaleGestureDetector): Boolean { zoomBy(d.scaleFactor, d.focusX, d.focusY); return true }
    })

    /** rows of a node: its ports, its summary lines starting beside the first port row with no input (the left side is free there) */
    private fun rows(n: Graph.Node) = Graph.kind(n.kind).let { k -> maxOf(k.ins.size, k.outs.size, 1, k.ins.size + minOf(2, summary(n).size)) }
    private fun h(n: Graph.Node) = (HEAD + ROW * rows(n) + thumbH(n)) * zs
    /** the port nearest (x, y) within [near]: the node, which port and whether it is an output. Nearest, not the first found:
     *  zoomed out the ports sit close together and the first one in reach was usually the wrong one. */
    private fun nearestPort(x: Float, y: Float, near: Float, outs: Boolean, ins: Boolean): Triple<Graph.Node, Int, Boolean>? {
        var best: Triple<Graph.Node, Int, Boolean>? = null; var bd = near
        for (n in g.nodes) {
            val k = Graph.kind(n.kind)
            if (outs) k.outs.indices.forEach { i -> val (px, py) = outPt(n, i); val d = hypot(x - px, y - py); if (d < bd) { bd = d; best = Triple(n, i, true) } }
            if (ins) k.ins.indices.forEach { i -> val (px, py) = inPt(n, i); val d = hypot(x - px, y - py); if (d < bd) { bd = d; best = Triple(n, i, false) } }
        }
        return best
    }
    /** dropping a wire on a node's body, not its dot: that node's port of the right type nearest the finger */
    private fun portOnNode(x: Float, y: Float, wantOut: Boolean, t: Graph.T): Pair<Graph.Node, Int>? {
        val n = g.nodes.reversed().firstOrNull { box(it).contains(x, y) } ?: return null
        val k = Graph.kind(n.kind); val list = if (wantOut) k.outs else k.ins
        var best = -1; var bd = Float.MAX_VALUE
        list.forEachIndexed { i, (_, pt) -> if (pt == t) {
            val (px, py) = if (wantOut) outPt(n, i) else inPt(n, i); val d = hypot(x - px, y - py); if (d < bd) { bd = d; best = i } } }
        return if (best < 0) null else n to best
    }
    /** a wire from the output [o] of [a] into the input [i] of [b], unless a run forbids replacing one */
    private fun join(a: Graph.Node, o: Int, b: Graph.Node, i: Int) {
        if (locked && g.wires.any { it.to == b.id && it.inp == i }) return
        g.connect(a, o, b, i); onChange()
    }
    private fun box(n: Graph.Node) = RectF(n.x * zs + panX, n.y * zs + panY, (n.x + W) * zs + panX, n.y * zs + panY + h(n))
    private fun inPt(n: Graph.Node, i: Int) = box(n).let { b -> b.left to b.top + (HEAD + ROW * i + ROW / 2) * zs }
    private fun outPt(n: Graph.Node, i: Int) = box(n).let { b -> b.right to b.top + (HEAD + ROW * i + ROW / 2) * zs }
    /** the node's values in short, under its ports */
    private fun summary(n: Graph.Node): List<String> = when (n.kind) {
        "model" -> listOf((ModelKind.of(n.p["model"]) ?: ModelKind.KREA).full + ((n.p["custom"] ?: "").takeIf { it.isNotEmpty() }?.let { " · $it" } ?: ""))
        "lora" -> listOf((n.p["name"] ?: "").ifEmpty { "—" } + " · " + strengthOf(n.p["strength"]?.toIntOrNull() ?: 100))
        "prompt", "negative" -> listOf((n.p["text"] ?: "").ifEmpty { "—" }.replace('\n', ' '))
        "size", "vaeenc" -> listOf("${n.p["w"]}×${n.p["h"]}")
        "sampler" -> listOf("${n.p["steps"]} · cfg ${n.p["cfg"]} · ${samplerLabel(n.p["sampler"] ?: "")}", "${Advanced.schedLabel(n.p["scheduler"] ?: "")} · shift ${n.p["shift"]} · ×${n.p["count"]}" + ((n.p["denoise"] ?: "1.0").takeIf { (it.toDoubleOrNull() ?: 1.0) < 1.0 }?.let { " · dn $it" } ?: "") + ((n.p["seed"] ?: "").takeIf { it.isNotEmpty() }?.let { " · $it" } ?: ""))
        else -> emptyList()
    }

    override fun onDraw(c: Canvas) {
        text.textSize = 13 * zs; wire.strokeWidth = 2.5f * zs
        for (w in g.wires) {
            val a = g.node(w.from) ?: continue; val b = g.node(w.to) ?: continue
            val (x1, y1) = outPt(a, w.out); val (x2, y2) = inPt(b, w.inp); val dx = maxOf(60 * zs, (x2 - x1) / 2)
            wire.color = color[Graph.kind(a.kind).outs[w.out].second]!!
            c.drawPath(Path().apply { moveTo(x1, y1); cubicTo(x1 + dx, y1, x2 - dx, y2, x2, y2) }, wire)
        }
        val fm = text.fontMetrics; val mid = -(fm.ascent + fm.descent) / 2          // from a line's middle to its baseline
        // the wire being dragged, from its port to the finger
        if (wiring) {
            val p = pending?.let { outPt(it.first, it.second) to Graph.kind(it.first.kind).outs[it.second].second }
                ?: pendIn?.let { inPt(it.first, it.second) to Graph.kind(it.first.kind).ins[it.second].second }
            if (p != null) {
                val (pt, t) = p; wire.color = color[t]!!
                c.drawPath(Path().apply { moveTo(pt.first, pt.second); lineTo(wireX, wireY) }, wire)
                fill.color = color[t]!!; c.drawCircle(wireX, wireY, R * zs * 0.8f, fill)
            }
        }
        for (n in g.nodes) {
            val k = Graph.kind(n.kind); val b = box(n); val busy = active == n.kind
            fill.color = card; c.drawRoundRect(b, 10 * zs, 10 * zs, fill)
            fill.color = if (pending?.first === n || busy) accent else muted; fill.alpha = if (busy) 110 else 60; c.drawRoundRect(RectF(b.left, b.top, b.right, b.top + HEAD * zs), 10 * zs, 10 * zs, fill); fill.alpha = 255
            text.color = ink; text.isFakeBoldText = true; c.drawText(k.title(), b.left + 10 * zs, b.top + HEAD * zs / 2 + mid, text); text.isFakeBoldText = false
            k.ins.forEachIndexed { i, (nm, t) -> val (x, y) = inPt(n, i); fill.color = color[t]!!; c.drawCircle(x, y, R * zs, fill); text.color = muted; c.drawText(nm, x + 12 * zs, y + mid, text) }
            k.outs.forEachIndexed { i, (nm, t) -> val (x, y) = outPt(n, i); fill.color = color[t]!!; c.drawCircle(x, y, R * zs, fill)
                text.color = muted; c.drawText(nm, x - 12 * zs - text.measureText(nm), y + mid, text) }
            val rows = rows(n); val sm = summary(n).take(2)
            sm.forEachIndexed { i, s0 ->
                val r = k.ins.size + i                                                // the row: after the inputs, beside an output if there is one
                val room = b.width() - 20 * zs - (k.outs.getOrNull(r)?.let { (nm, _) -> text.measureText(nm) + 24 * zs } ?: 0f)
                var s = s0; if (text.measureText(s) > room) { while (s.isNotEmpty() && text.measureText("$s…") > room) s = s.dropLast(1); s = "$s…" }
                text.color = ink; c.drawText(s, b.left + 10 * zs, b.top + (HEAD + ROW * r + ROW / 2) * zs + mid, text) }
            val pic = picture
            if (n.kind == "save" && pic != null) {
                val top = b.top + (HEAD + ROW * rows) * zs; val th = (thumbH(n) - 10f) * zs; val tw = th * pic.width / pic.height
                val l = b.left + (b.width() - tw) / 2; c.drawBitmap(pic, null, RectF(l, top, l + tw, top + th), null)
            }
            if (busy) {                                                               // the node at work: an accent outline, and its progress along the foot
                ring.strokeWidth = 2.5f * zs; ring.color = accent; c.drawRoundRect(b, 10 * zs, 10 * zs, ring)
                if (progress >= 0f) { fill.color = accent; c.drawRect(b.left + 8 * zs, b.bottom - 6 * zs, b.left + 8 * zs + (b.width() - 16 * zs) * progress.coerceIn(0f, 1f), b.bottom - 3 * zs, fill) }
            }
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        scaler.onTouchEvent(e)
        if (e.pointerCount > 1) { pinched = true; drag = null; panning = false; removeCallbacks(longPress); return true }      // two fingers: only the zoom
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                lastX = e.x; lastY = e.y; downX = e.x; downY = e.y; moved = false; pinched = false; wiring = false
                // a port first: the nearest one within 26dp (a dot is small, and zoomed out they sit close together)
                val near = 26 * dp
                val hit = nearestPort(e.x, e.y, near, true, true)
                if (hit != null) {
                    val (n, i, isOut) = hit
                    wireX = e.x; wireY = e.y
                    if (isOut) { pending = n to i; pendIn = null; wiring = true }
                    else {
                        val p = pending
                        if (p != null) { join(p.first, p.second, n, i); pending = null }        // the second tap of tap, tap
                        else { pendIn = n to i; wiring = true }                                  // or drag from here towards an output
                    }
                    invalidate(); return true
                }
                drag = g.nodes.reversed().firstOrNull { box(it).contains(e.x, e.y) }
                drag?.let { dragDX = e.x - it.x * zs; dragDY = e.y - it.y * zs } ?: run { panning = true; pressX = e.x; pressY = e.y; postDelayed(longPress, 550) }
            }
            MotionEvent.ACTION_MOVE -> {
                if (pinched) return true
                if (hypot(e.x - downX, e.y - downY) > 8 * dp) { moved = true; removeCallbacks(longPress) }
                if (wiring) { wireX = e.x; wireY = e.y; invalidate(); lastX = e.x; lastY = e.y; return true }
                if (moved) drag?.let { it.x = (e.x - dragDX) / zs; it.y = (e.y - dragDY) / zs; invalidate() }
                if (panning) { panX += e.x - lastX; panY += e.y - lastY; invalidate() }
                lastX = e.x; lastY = e.y
            }
            MotionEvent.ACTION_UP -> {
                removeCallbacks(longPress)
                if (wiring) {
                    val near = 26 * dp
                    if (moved) {
                        // dropped: a port nearby, or anywhere on a node that has one of the right type
                        val from = pending; val into = pendIn
                        if (from != null) {
                            val t = Graph.kind(from.first.kind).outs[from.second].second
                            val tgt = nearestPort(e.x, e.y, near, false, true)?.let { it.first to it.second } ?: portOnNode(e.x, e.y, false, t)
                            if (tgt != null) join(from.first, from.second, tgt.first, tgt.second)
                            pending = null
                        } else if (into != null) {
                            val t = Graph.kind(into.first.kind).ins[into.second].second
                            val src = nearestPort(e.x, e.y, near, true, false)?.let { it.first to it.second } ?: portOnNode(e.x, e.y, true, t)
                            if (src != null) join(src.first, src.second, into.first, into.second)
                        }
                        pendIn = null
                    } else {
                        // a tap: an output waits for its input (tap, tap), a wired input lets go of its wire
                        val into = pendIn
                        if (into != null && !locked) { g.wires.removeAll { it.to == into.first.id && it.inp == into.second }; onChange() }
                        pendIn = null
                    }
                    wiring = false; invalidate(); parent?.requestDisallowInterceptTouchEvent(false); return true
                }
                val n = drag
                if (n != null && !moved && !pinched) onEdit(n)                              // a tap on a node: its settings
                else if (n != null && moved) onChange()
                else if (n == null && !moved && !pinched && pending != null) { pending = null; invalidate() }   // a tap on the empty space drops a half-made wire
                drag = null; panning = false; parent?.requestDisallowInterceptTouchEvent(false)
            }
            MotionEvent.ACTION_CANCEL -> { removeCallbacks(longPress); drag = null; panning = false; parent?.requestDisallowInterceptTouchEvent(false) }
        }
        return true
    }
}
