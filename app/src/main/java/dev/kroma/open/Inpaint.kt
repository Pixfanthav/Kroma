package dev.kroma.open

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import android.view.MotionEvent
import android.view.View
import java.io.File

/**
 * img2img's start picture made ready for the sampler: placed in the W x H frame, encoded, and the mask of what is redrawn.
 *  - cut ([fit] false): scaled to cover the frame, the middle kept (as plain img2img)
 *  - fit ([fit] true, outpaint): scaled to fit inside, the edges stretched out over the rest, and the rest masked to be drawn anew
 * A mask painted on the picture ([maskFile], white = redraw) is placed the same way. The mask goes to the latent grid (8 x 8 pixels
 * a cell) as each cell's mean, then faded out into what stays (see [prepare]).
 */
object Img2Img {
    class Ready(val latent: FloatArray, val mask: FloatArray?)

    fun prepare(root: File, src: Bitmap, W: Int, H: Int, fit: Boolean, maskFile: File?): Ready {
        val k = if (fit) minOf(W.toFloat() / src.width, H.toFloat() / src.height) else maxOf(W.toFloat() / src.width, H.toFloat() / src.height)
        val sw = Math.round(src.width * k).coerceAtLeast(1); val sh = Math.round(src.height * k).coerceAtLeast(1)
        val ox = (W - sw) / 2; val oy = (H - sh) / 2                     // where the scaled picture's corner lands in the frame (negative: cut)
        fun mirror(v: Int, n: Int): Int { val m = Math.floorMod(v, 2 * n); return if (m < n) m else 2 * n - 1 - m }
        fun place(b: Bitmap, mirrored: Boolean = false): IntArray {
            val sc = Bitmap.createScaledBitmap(b, sw, sh, true)
            val all = IntArray(sw * sh).also { sc.getPixels(it, 0, sw, 0, 0, sw, sh) }
            if (sc !== b) sc.recycle()
            // inside the frame's picture area the picture; outside, its edge stretched out (the mask) or the picture mirrored (the start image)
            return IntArray(W * H) { i -> val x = i % W - ox; val y = i / W - oy
                if (mirrored) all[mirror(y, sh) * sw + mirror(x, sw)] else all[y.coerceIn(0, sh - 1) * sw + x.coerceIn(0, sw - 1)] }
        }
        // outpaint: what is added starts as the picture mirrored over the edge and blurred to its colours and layout only, then is redrawn from
        // there (not from noise): the extension continues the picture's own background instead of whatever the prompt suggests
        val px = place(src, fit)
        if (fit && (sw < W || sh < H)) {
            val full = Bitmap.createBitmap(px, W, H, Bitmap.Config.ARGB_8888)
            val small = Bitmap.createScaledBitmap(full, maxOf(1, W / 24), maxOf(1, H / 24), true)
            val blur = Bitmap.createScaledBitmap(small, W, H, true)
            val bp = IntArray(W * H).also { blur.getPixels(it, 0, W, 0, 0, W, H) }
            full.recycle(); small.recycle(); blur.recycle()
            for (i in 0 until W * H) { val x = i % W - ox; val y = i / W - oy; if (x !in 0 until sw || y !in 0 until sh) px[i] = bp[i] }
        }
        val framed = Bitmap.createBitmap(px, W, H, Bitmap.Config.ARGB_8888)
        val z = KreaGen.encode(root, framed, W, H); framed.recycle()
        val LW = W / 8; val LH = H / 8
        val pad = FloatArray(LW * LH); val paint = FloatArray(LW * LH)
        if (fit && (sw < W || sh < H)) {
            for (y in 0 until LH) for (x in 0 until LW) {
                val inside = x * 8 >= ox && x * 8 + 8 <= ox + sw && y * 8 >= oy && y * 8 + 8 <= oy + sh
                if (!inside) pad[y * LW + x] = 1f
            }
        }
        maskFile?.takeIf { it.isFile }?.let { android.graphics.BitmapFactory.decodeFile(it.absolutePath) }?.let { mb ->
            val px = place(mb); mb.recycle()
            for (y in 0 until LH) for (x in 0 until LW) {
                var s = 0
                for (dy in 0 until 8) for (dx in 0 until 8) s += Color.red(px[(y * 8 + dy) * W + x * 8 + dx])
                paint[y * LW + x] = s / (64 * 255f)
            }
        }
        if (pad.none { it > 0f } && paint.none { it > 0.02f }) return Ready(z, null)
        // the edge fades into what stays: twelve cells (96 pixels) of the picture around an outpainted border are redrawn more and more lightly,
        // three cells around a painted area. With Differential Diffusion a cell at v is redrawn over the last v of the steps.
        fun fade(m: FloatArray, r: Int) = FloatArray(m.size) { i -> val x = i % LW; val y = i / LW; var v = m[i]
            for (dy in -r..r) for (dx in -r..r) { val xx = x + dx; val yy = y + dy
                if (xx in 0 until LW && yy in 0 until LH) v = maxOf(v, m[yy * LW + xx] * (1f - Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat() / (r + 1)).coerceAtLeast(0f)) }
            v }
        val a = fade(pad, 12); val b = fade(paint, 3)
        val grown = FloatArray(a.size) { maxOf(a[it], b[it]) }
        return Ready(z, grown)
    }
}

/** The start picture with a mask painted over it by finger: red where it is redrawn. [mask] is white on black at the picture's size. */
@SuppressLint("ViewConstructor")
class MaskView(ctx: Context, private val pic: Bitmap, val mask: Bitmap) : View(ctx) {
    var brush = 48f                // in picture pixels
    var erase = false
    private val shown = RectF()
    private val tinted: Bitmap = Bitmap.createBitmap(mask.width, mask.height, Bitmap.Config.ARGB_8888)
    private val pen = Paint().apply { isAntiAlias = true; strokeCap = Paint.Cap.ROUND; style = Paint.Style.STROKE }
    private var lx = 0f; private var ly = 0f

    init { redrawTint() }
    private fun redrawTint() {
        val c = Canvas(tinted); c.drawColor(0, PorterDuff.Mode.CLEAR)
        // the mask's white as alpha, shown in red
        val px = IntArray(mask.width * mask.height).also { mask.getPixels(it, 0, mask.width, 0, 0, mask.width, mask.height) }
        for (i in px.indices) px[i] = Color.argb(Color.red(px[i]) * 140 / 255, 255, 60, 60)
        tinted.setPixels(px, 0, mask.width, 0, 0, mask.width, mask.height)
    }
    fun clear() { Canvas(mask).drawColor(Color.BLACK); redrawTint(); invalidate() }

    override fun onMeasure(w: Int, h: Int) {
        val width = MeasureSpec.getSize(w)
        setMeasuredDimension(width, minOf((width.toFloat() * pic.height / pic.width).toInt(), (resources.displayMetrics.heightPixels * 0.6f).toInt()))
    }
    override fun onDraw(c: Canvas) {
        val k = minOf(width.toFloat() / pic.width, height.toFloat() / pic.height)
        val w = pic.width * k; val h = pic.height * k
        shown.set((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2)
        c.drawBitmap(pic, null, shown, null); c.drawBitmap(tinted, null, shown, null)
    }
    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        parent?.requestDisallowInterceptTouchEvent(true)
        val k = mask.width / shown.width()
        val x = (e.x - shown.left) * k; val y = (e.y - shown.top) * k
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> { lx = x; ly = y; stroke(x, y, x, y) }
            MotionEvent.ACTION_MOVE -> { stroke(lx, ly, x, y); lx = x; ly = y }
        }
        return true
    }
    private fun stroke(x0: Float, y0: Float, x1: Float, y1: Float) {
        pen.strokeWidth = brush; pen.color = if (erase) Color.BLACK else Color.WHITE
        Canvas(mask).drawLine(x0, y0, x1, y1, pen)
        pen.color = if (erase) Color.TRANSPARENT else Color.argb(140, 255, 60, 60)
        pen.xfermode = PorterDuffXfermode(if (erase) PorterDuff.Mode.CLEAR else PorterDuff.Mode.SRC)
        Canvas(tinted).drawLine(x0, y0, x1, y1, pen); pen.xfermode = null
        invalidate()
    }
}
