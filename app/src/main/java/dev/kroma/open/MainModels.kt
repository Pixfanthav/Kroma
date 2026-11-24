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

/* MainActivity: the model button: families, versions, the picker and its gear (moved out of MainActivity.kt unchanged) */

/** a model pack's family ("Krea 2", "Anima") and version ("Turbo"): the pack's name is the two together */
// a pack id is its kind's id or starts with it (older lists had variants such as "krea_…")
internal fun MainActivity.familyOf(p: Models.Pack) = (ModelKind.of(p.id) ?: ModelKind.entries.firstOrNull { p.id.startsWith(it.id) })?.family ?: p.name

/** the version shown after the family; empty: the family alone (YuE2 has one). The packs whose names are only the family name it here
 *  (they showed "FLUX.2 Klein 4B · FLUX.2 Klein 4B") */
internal fun MainActivity.versionOf(p: Models.Pack) = when {
    p.id.startsWith("klein") -> "Turbo"; p.id.startsWith("sdxl") -> "Base"; p.id.startsWith("yue") -> ""; p.id.startsWith("moss") -> "Turbo"
    else -> p.name.removePrefix(familyOf(p)).trim().ifEmpty { p.name } }

/** a pack on the phone that can draw: its built-in model is here, or a fine-tune brought in for it is */
internal fun MainActivity.usable(p: Models.Pack) = models.hasBase(p.id) || Custom.list(models.root, p.id).isNotEmpty()

/** the model, then its version when the family has more than one: the built-in ones, and in extended mode the fine-tunes
 *  brought in for it ([done] gets the pack and the fine-tune's name, "" for a built-in one) */
internal fun MainActivity.modelPicker(installed: List<Models.Pack>, done: (Int, String) -> Unit) {
    fun styledDlg(b: AlertDialog.Builder) = b.show().also { it.window?.setBackgroundDrawable(round(CARD, 20)) }
    // only the families that can draw (Anima downloaded without its built-in model and with no fine-tune used to be offered);
    // fine-tunes whose model's text encoder and VAE are not on the phone are listed greyed, saying what to download
    val kindOrder = ModelKind.entries.map { it.family }          // the table's order: Krea, Anima, MOSS, Klein, SDXL, the song model last
    val families = installed.filter { usable(it) }.map { familyOf(it) }.distinct().sortedBy { kindOrder.indexOf(it).let { i -> if (i < 0) kindOrder.size else i } }
    val orphans = Custom.list(models.root).map { it.kind }.distinct().filter { k -> installed.none { it.id == k } }
        .map { k -> familyOf(Models.Pack(k, k, true, emptyList())) }.filter { it !in families }.distinct()
    val all = families + orphans.sortedBy { kindOrder.indexOf(it).let { i -> if (i < 0) kindOrder.size else i } }
    if (all.isEmpty()) { status(tr(R.string.no_model_to_draw_with_download)); return }
    val adapter = object : ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, all) {
        override fun isEnabled(position: Int) = position < families.size
        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View = super.getView(position, convertView, parent).also {
            (it as TextView).setTextColor(INK); it.alpha = if (position < families.size) 1f else 0.4f
            if (position >= families.size) it.setOnClickListener { _ -> status(tr(R.string.s_the_fine_tunes_need_its).format(all[position], all[position])) }
        }
    }
    styledDlg(AlertDialog.Builder(this).setTitle(tr(R.string.model))
        .setAdapter(adapter) { _, f ->
            // a pack kept without its built-in model offers only the fine-tunes brought in for it (in any mode)
            val vs = installed.withIndex().filter { familyOf(it.value) == families[f] }.flatMap { (i, p) ->
                (if (models.hasBase(p.id)) listOf(Triple(i, "", versionOf(p))) else emptyList()) +
                (if (p.id != "yue" && (Advanced.on || !models.hasBase(p.id))) Custom.list(models.root, p.id).map { Triple(i, it.name, it.name) } else emptyList()) }
                .distinctBy { installed[it.first].id to it.second }
            if (vs.isEmpty()) { status(tr(R.string.no_model_to_draw_with_bring)); return@setAdapter }
            if (vs.size == 1) { done(vs[0].first, vs[0].second); return@setAdapter }
            styledDlg(AlertDialog.Builder(this).setTitle(families[f])
                .setItems(vs.map { it.third }.toTypedArray()) { _, j -> done(vs[j].first, vs[j].second) })
        })
}

/** the gear beside the model (extended mode): bringing a fine-tune in (a file or online, in the format asked), deleting them */
internal fun MainActivity.modelSettings() {
    val t0 = System.currentTimeMillis()
    var dlg: AlertDialog? = null
    val box = importSection() as LinearLayout
    val all = Custom.list(models.root)
    if (all.isNotEmpty()) box.addView(button(tr(R.string.delete_an_imported_model)) {
        AlertDialog.Builder(this).setItems(all.map { "${it.name}  ·  ${ModelKind.family(it.kind)}" }.toTypedArray()) { _, i ->
            val m = all[i]
            AlertDialog.Builder(this).setTitle(tr(R.string.delete_from_this_phone, m.name))
                .setPositiveButton(tr(R.string.delete)) { _, _ -> if (Advanced.custom(m.kind) == m.name) Advanced.setCustom(m.kind, ""); m.dir.deleteRecursively(); Logs.add("model deleted"); dlg?.dismiss(); route() }
                .setNegativeButton(tr(R.string.cancel), null).show()
        }.show() }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = px(10) })
    box.setPadding(px(22), px(18), px(22), px(4))
    dlg = AlertDialog.Builder(this).setView(ScrollView(this).apply { addView(box) }).setPositiveButton("OK", null).show()
    dlg.window?.setBackgroundDrawable(round(CARD, 20))
    Logs.add("model settings opened · ${System.currentTimeMillis() - t0} ms")
}
