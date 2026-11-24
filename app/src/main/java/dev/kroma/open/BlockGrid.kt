package dev.kroma.open

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.view.View

/**
 * A row of cells, one per block of a long job (converting a model, baking a LoRA into the weights): finished cells filled, the one
 * being worked on breathing, so the screen moves even while one block takes many seconds. [set] from any thread.
 */
class BlockGrid(ctx: Context, private val on: Int, private val off: Int) : View(ctx) {
    @Volatile private var done = 0
    @Volatile private var total = 0
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val r = RectF()
    private val dp = ctx.resources.displayMetrics.density

    fun set(d: Int, n: Int) { done = d.coerceIn(0, n); total = n; postInvalidateOnAnimation() }

    override fun onMeasure(w: Int, h: Int) {
        val width = MeasureSpec.getSize(w); val n = total.coerceAtLeast(1)
        val perRow = rowCells(width, n); val rows = (n + perRow - 1) / perRow
        setMeasuredDimension(width, (rows * cell(width, perRow) + (rows - 1) * gap()).toInt().coerceAtLeast(1))
    }

    private fun gap() = 3 * dp
    // up to 30 cells a row, no smaller than 7dp: 28 blocks fit one row, 60 two
    private fun rowCells(width: Int, n: Int): Int { var k = minOf(n, 30); while (k > 1 && cell(width, k) < 7 * dp) k--; return k }
    private fun cell(width: Int, k: Int) = (width - (k - 1) * gap()) / k

    override fun onDraw(c: Canvas) {
        val n = total; if (n <= 0) return
        val perRow = rowCells(width, n); val s = cell(width, perRow); val g = gap()
        if (height < ((n + perRow - 1) / perRow) * (s + g) - g - 1) { requestLayout(); return }
        val t = (System.currentTimeMillis() % 1200) / 1200f; val pulse = 0.35f + 0.65f * (0.5f - 0.5f * Math.cos(2 * Math.PI * t).toFloat())
        for (i in 0 until n) {
            val x = (i % perRow) * (s + g); val y = (i / perRow) * (s + g)
            r.set(x, y, x + s, y + s)
            paint.color = when { i < done -> on; else -> off }
            c.drawRoundRect(r, 2 * dp, 2 * dp, paint)
            if (i == done && done < n) { paint.color = on; paint.alpha = (pulse * 255).toInt(); c.drawRoundRect(r, 2 * dp, 2 * dp, paint); paint.alpha = 255 }
        }
        if (done < n && visibility == VISIBLE) postInvalidateOnAnimation()
    }
}
