package dev.kroma.open

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.StatFs
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.concurrent.thread

/**
 * The model files and how they get onto the phone.
 *
 * A model server is any static file host with a manifest.json at its root (the maintainers' tools write it; the published one is in the Kroma-Shared repository):
 *   "packs": what the user can tick -- models (at least one is needed) and options -- each with the "groups"
 *            it is made of, so a size can be broken down on screen
 *   "base" : the groups of the files every pack needs (VAE, content classifier, tokenizer)
 *   "sets" : "base" | "<pack>" | "<pack>@<chip>" -> files { "p" path on the phone, "u" path on the server,
 *            "n" bytes, "h" sha256, "g" group }. "@<chip>" sets are HTP context binaries, compiled per chip:
 *            the same .bin name exists once per chip on the server but at one place on the phone.
 *
 * Files download to "<name>.part", resume with an HTTP Range, and are renamed only after their SHA-256
 * matches, so an interrupted or corrupted download can never be loaded.
 */
class Models(private val ctx: Context) {
    val root: File = ctx.getExternalFilesDir(null)!!
    val bundle get() = File(root, "krea_bundle")
    private val prefs = ctx.getSharedPreferences("kroma", Context.MODE_PRIVATE)

    /** Where the models come from: the repository of this phone's NPU generation ([SERVERS]). There is nothing to type in. */
    val server: String get() = Chip.detect()?.let { SERVERS[it.tag] }.orEmpty()

    class Item(val path: String, val url: String, val size: Long, val sha: String, val group: String)
    /** A named part of a pack ("DiT", "text encoder"...), for showing what a download size consists of. */
    class Group(val id: String, val name: String)
    /** Something the user can tick: a model (at least one is needed to generate) or an option. */
    class Pack(val id: String, val name: String, val isModel: Boolean, val groups: List<Group>)

    /** The packs the user chose to keep on this phone (ids). */
    var chosen: Set<String>
        get() = prefs.getStringSet("packs", emptySet())!!
        set(v) { prefs.edit().putStringSet("packs", v).apply() }
    /** Model packs kept without their built-in image model (group "dit", e.g. Krea 2 Turbo's blocks): for someone who only draws
     *  with fine-tunes brought in, which replace exactly that part; the text encoder, VAE and the rest of the pack stay. */
    var noBase: Set<String>
        get() = prefs.getStringSet("packs_nobase", emptySet())!!
        set(v) { prefs.edit().putStringSet("packs_nobase", v).apply() }
    fun hasBase(id: String) = id !in noBase

    /** the saved manifest, parsed once and kept while the file stays the same (540 KB of JSON: parsed anew on every call it was read
     *  dozens of times for one screen, and the extended mode's gear took seconds to open) */
    @Volatile private var mCache: Triple<Long, Long, JSONObject>? = null
    private fun manifest(): JSONObject? {
        val f = File(root, "manifest.json"); if (!f.exists()) return null
        val stamp = f.lastModified(); val len = f.length()
        mCache?.let { (t, n, j) -> if (t == stamp && n == len) return j }
        val t = System.currentTimeMillis()
        return runCatching { JSONObject(f.readText()) }.getOrNull()?.takeIf { it.has("packs") }?.also { mCache = Triple(stamp, len, it)
            Logs.add("manifest read · ${len / 1024} KB · ${System.currentTimeMillis() - t} ms") }
    }
    private fun groups(a: org.json.JSONArray) = (0 until a.length()).map { a.getJSONObject(it).let { o -> Group(o.getString("g"), tr(o.getString("name"), o.optString("name_ko", o.getString("name")))) } }

    /** Every pack the server offers, from the manifest saved by the last fetch. */
    fun offered(): List<Pack> = (manifest()?.getJSONArray("packs")?.let { a -> (0 until a.length()).map { a.getJSONObject(it).let { o ->
        Pack(o.getString("id"), tr(o.getString("name"), o.optString("name_ko", o.getString("name"))), o.optString("kind") == "model", groups(o.getJSONArray("groups"))) } }.sortedBy { if (it.id == "yue") 1 else 0 } } ?: emptyList())  // the song model after the image models
        // a cut-down v73 (Chip.SMALL73): Anima and the options that run on the CPU, no other model and no upscaler (laid out for 8 MB)
        .filter { p -> Chip.detect()?.animaOnly != true || p.id == "anima" || (!p.isModel && p.id != "upscale") }
    fun baseGroups(): List<Group> = manifest()?.optJSONArray("base")?.let { groups(it) } ?: emptyList()

    private fun set(m: JSONObject, name: String): List<Item> {
        val a = m.getJSONObject("sets").optJSONArray(name) ?: return emptyList()
        return (0 until a.length()).map { a.getJSONObject(it).let { o -> Item(o.getString("p"), o.optString("u", o.getString("p")), o.getLong("n"), o.getString("h"), o.optString("g")) } }
    }

    /** The files one pack needs on this chip, without the shared base (and without its image model when kept without one). */
    fun filesOf(id: String, chip: Chip): List<Item> = allFilesOf(id, chip).filter { hasBase(id) || it.group != "dit" }
    /** all of a pack's files, its image model included, in the weight format chosen for it ([fmt]) */
    fun allFilesOf(id: String, chip: Chip): List<Item> = manifest()?.let { m ->
        val all = set(m, id) + set(m, "$id@${chip.tag}")
        val v = fmt(id).takeIf { it != "f16" }?.let { set(m, "$id~$it") }.orEmpty() + teFmt(id).takeIf { it != "f16" }?.let { set(m, "$id~te8") }.orEmpty()
        if (v.isEmpty()) all else v.map { it.path }.toSet().let { r -> all.filter { it.path !in r } + v }
    } ?: emptyList()

    /** The weight formats a model pack comes in: "f16" (as made, the default) and the sets "<pack>~q8" / "<pack>~q4" of the manifest,
     *  each replacing the image model's block files (and adding format.txt, which the engine reads: none = fp16). */
    fun formats(id: String): List<String> = manifest()?.getJSONObject("sets")?.let { s -> listOf("f16") + listOf("q8", "q4").filter { s.has("$id~$it") } } ?: listOf("f16")
    fun fmt(id: String): String = prefs.getString("fmt_$id", "f16")!!.takeIf { it in formats(id) } ?: "f16"
    fun setFmt(id: String, f: String) { prefs.edit().putString("fmt_$id", f).apply() }
    /** The text encoder's own weight format, chosen apart from the image model's: "f16" or "q8" (the set "<pack>~te8", present for the
     *  packs with a big encoder; it replaces the encoder's files and adds kte/format.txt, which the encoder reads: none = fp16). */
    fun teFormats(id: String): List<String> = manifest()?.getJSONObject("sets")?.let { s -> if (s.has("$id~te8")) listOf("f16", "q8") else listOf("f16") } ?: listOf("f16")
    fun teFmt(id: String): String = prefs.getString("tfmt_$id", "f16")!!.takeIf { it in teFormats(id) } ?: "f16"
    fun setTeFmt(id: String, f: String) { prefs.edit().putString("tfmt_$id", f).apply() }
    /** Download through the Hugging Face mirror (hf-mirror.com) for those who cannot reach huggingface.co: the same files, each still hash-checked. */
    var hfMirror: Boolean get() = prefs.getBoolean("hf_mirror", false); set(v) { prefs.edit().putBoolean("hf_mirror", v).apply() }
    /** One small GET of the model repository's config.json when a pack's download starts: Hugging Face counts requests to that file
     *  as downloads of the repository (the model repositories have no other file it counts), so this is the pack's install count. */
    fun countInstall(id: String, chip: Chip) { runCatching {
        val u = filesOf(id, chip).firstOrNull { it.url.startsWith("https://") && "/resolve/main/" in it.url }?.url ?: return
        val base = u.substring(0, u.indexOf("/resolve/main/") + 14)
        (URL(viaMirror(base + "config.json")).openConnection() as HttpURLConnection).run { connectTimeout = 8000; readTimeout = 8000; responseCode; disconnect() } } }
    private fun viaMirror(u: String) = if (hfMirror) u.replaceFirst("https://huggingface.co/", "https://hf-mirror.com/") else u
    /** Before a download: a pack whose chosen format is not the one on the phone loses the files the formats differ in (its block
     *  files and format.txt), so the new ones come down and no fp16 run reads a format.txt left from Q8. */
    fun applyFormats(ids: Set<String>) {
        val m = manifest() ?: return
        for (id in ids) {
            val have = prefs.getString("fmt_have_$id", "f16")!!; val want = fmt(id)
            if (have == want) continue
            listOf("q8", "q4").flatMap { set(m, "$id~$it") }.map { it.path }.toSet().forEach { File(root, it).delete(); File(root, "$it.part").delete() }
            prefs.edit().putString("fmt_have_$id", want).apply(); Logs.add("weight format $id: $have -> $want")
        }
        for (id in ids) {
            val have = prefs.getString("tfmt_have_$id", "f16")!!; val want = teFmt(id)
            if (have == want) continue
            set(m, "$id~te8").map { it.path }.forEach { File(root, it).delete(); File(root, "$it.part").delete() }
            prefs.edit().putString("tfmt_have_$id", want).apply(); Logs.add("encoder format $id: $have -> $want")
        }
    }
    fun baseFiles(chip: Chip): List<Item> = manifest()?.let { set(it, "base") + set(it, "base@${chip.tag}") } ?: emptyList()

    /** Everything the chosen packs need. */
    fun needed(chip: Chip, ids: Set<String> = chosen): List<Item> = baseFiles(chip) + ids.flatMap { filesOf(it, chip) }

    /** Present with the right size (the hash was checked when it was renamed into place). */
    fun missing(items: List<Item>) = items.filter { File(root, it.path).length() != it.size }

    /** The chosen packs that are completely on the phone (with the base). */
    fun installed(chip: Chip): List<Pack> {
        if (manifest() == null || missing(baseFiles(chip)).isNotEmpty()) return emptyList()
        return offered().filter { it.id in chosen && filesOf(it.id, chip).let { f -> f.isNotEmpty() && missing(f).isEmpty() } }
    }
    fun has(id: String, chip: Chip) = installed(chip).any { it.id == id }

    /** Everything chosen is on the phone, and that includes at least one model. */
    fun ready(chip: Chip) = installed(chip).let { i -> i.any { it.isModel } && i.size == chosen.size }

    fun freeBytes() = StatFs(root.absolutePath).availableBytes

    fun onWifi(): Boolean {
        val cm = ctx.getSystemService(ConnectivityManager::class.java)
        return cm.getNetworkCapabilities(cm.activeNetwork)?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true
    }

    /** Reads the server's manifest and saves it; [offered] and [needed] answer from the saved copy. */
    fun fetchManifest() {
        val text = (URL(viaMirror("$server/manifest.json")).openConnection() as HttpURLConnection).run {
            connectTimeout = 15000; readTimeout = 30000
            if (responseCode != 200) error("manifest.json: HTTP $responseCode")
            inputStream.use { it.readBytes().toString(Charsets.UTF_8) }
        }
        if (!JSONObject(text).has("packs")) error("the model server's manifest is from an older layout")
        File(root, "manifest.json").writeText(text)
    }

    /**
     * Downloads what is missing. [progress] gets (bytes done, bytes total, current file). Throws on a
     * network error after retries, on a hash mismatch (the partial file is removed), or when [stop] turns true.
     */
    fun download(items: List<Item>, stop: () -> Boolean, progress: (Long, Long, String) -> Unit) {
        val todo = missing(items)
        val total = todo.sumOf { it.size }
        val next = java.util.concurrent.atomic.AtomicInteger(0)
        val seen = java.util.concurrent.ConcurrentHashMap<String, Long>()   // bytes per file, its .part counted from the start
        for (it in todo) seen[it.path] = File(root, it.path + ".part").length().coerceAtMost(it.size)
        val failed = java.util.concurrent.atomic.AtomicReference<Throwable?>(null)
        val halt = { stop() || failed.get() != null }
        // several files at a time: one connection left the line half used, and each large file's checksum held up the next
        val workers = List(minOf(DOWNLOADS, maxOf(todo.size, 1))) { thread {
            try {
                while (!halt()) {
                    val it = todo.getOrNull(next.getAndIncrement()) ?: break
                    val dst = File(root, it.path); val part = File(root, it.path + ".part")
                    dst.parentFile?.mkdirs()
                    var tries = 0
                    while (true) {
                        if (halt()) throw InterruptedException()
                        try {
                            fetch(it, part, halt) { n -> seen[it.path] = n; progress(seen.values.sum(), total, it.path) }
                            break
                        } catch (e: InterruptedException) { throw e }
                        catch (e: Exception) {
                            if (++tries >= 8) throw e
                            Logs.add("download retry $tries ${it.path}: ${e.message?.take(120)}")
                            Thread.sleep(2000L * tries)
                        }
                    }
                    val got = sha256(part)
                    if (!got.equals(it.sha, ignoreCase = true)) { part.delete(); error("${it.path}: checksum mismatch, the file was discarded") }
                    dst.delete(); if (!part.renameTo(dst)) error("${it.path}: could not move into place")
                }
            } catch (e: InterruptedException) { }
            catch (e: Throwable) { failed.compareAndSet(null, e) }
        } }
        workers.forEach { it.join() }
        failed.get()?.let { throw it }
        if (stop()) throw InterruptedException()
    }

    private fun fetch(it: Item, part: File, stop: () -> Boolean, onBytes: (Long) -> Unit) {
        var have = part.length()
        if (have > it.size) { part.delete(); have = 0 }
        if (have == it.size) { onBytes(have); return }
        // "u": a path on the server, or a whole address (a file in another repository: each model has its own, the manifest lives in the shared one)
        fun enc(p: String) = p.split('/').joinToString("/") { s -> java.net.URLEncoder.encode(s, "UTF-8").replace("+", "%20") }
        val abs = it.url.startsWith("https://"); val cut = it.url.indexOf("/resolve/main/")
        val c = URL(viaMirror(if (!abs) "$server/${enc(it.url)}" else if (cut < 0) it.url else it.url.substring(0, cut + 14) + enc(it.url.substring(cut + 14)))).openConnection() as HttpURLConnection
        c.connectTimeout = 15000; c.readTimeout = 30000
        if (have > 0) c.setRequestProperty("Range", "bytes=$have-")
        try {
            val code = c.responseCode
            if (code == 200 && have > 0) { part.delete(); have = 0 }        // the server ignored the Range
            else if (code != 200 && code != 206) error("HTTP $code")
            RandomAccessFile(part, "rw").use { out ->
                out.seek(have)
                c.inputStream.use { ins ->
                    val buf = ByteArray(1 shl 20); var last = 0L
                    while (true) {
                        if (stop()) throw InterruptedException()
                        val n = ins.read(buf); if (n < 0) break
                        out.write(buf, 0, n); have += n
                        if (have - last > (1 shl 20)) { onBytes(have); last = have }
                    }
                }
            }
            if (have != it.size) error("got $have of ${it.size} bytes")
            onBytes(have)
        } finally { c.disconnect() }
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { s -> val buf = ByteArray(1 shl 20); while (true) { val n = s.read(buf); if (n < 0) break; md.update(buf, 0, n) } }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val DOWNLOADS = 4
        /** where manifest.json is, per NPU generation (Chip.tag): the engine's files are the same for every chip, so one shared repository
         *  (Kroma-Shared, VAE, filter, tokenizers, translation) whose manifest points at each model's own repository. 8 Gen 1 is not supported. */
        private const val HUB = "https://huggingface.co/Pixfanthav/Kroma-Shared/resolve/main"
        val SERVERS = mapOf("8gen1" to "", "8gen2" to HUB, "8gen3" to HUB, "8elite" to HUB, "8elite_gen5" to HUB)
    }
}
