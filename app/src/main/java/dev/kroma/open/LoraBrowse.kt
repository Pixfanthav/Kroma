package dev.kroma.open

import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * Finding a LoRA on Civitai from inside the app: the site's public listing, filtered to the model on screen, and the file
 * fetched into the cache for [LoraImport]. No account or key is used, so a file its author put behind a login does not
 * download (the site answers 401, and the app says so). Models the site marks as adult are never asked for.
 */
object LoraBrowse {
    /** [repo]: a Hugging Face repository (its file is picked when the hit is chosen, [url] is empty until then) */
    class Hit(val name: String, val creator: String, val downloads: Long, var sizeMb: Int, var file: String, var url: String, val page: String, val repo: String? = null) {
        /** what the version's sample pictures were made with, for a model (null when they say nothing usable) */
        var defaults: Advanced.Defaults? = null
        /** a model's files on the page, each with what its weights are stored as ([ModelSrc]); [fmt] the one taken */
        var srcs: List<ModelSrc.Src> = emptyList(); var fmt = ""; var srcAsked = false
        /** the version's trigger words as Civitai lists them (trainedWords): many files carry none of their own */
        var words: List<String> = emptyList()
    }

    /** the sampler names sites write, as the app's */
    private fun sampler(s: String) = when (s.trim().lowercase().replace(' ', '_')) {
        "er_sde", "er-sde" -> "er_sde"; "euler_a", "euler_ancestral" -> "euler_ancestral"; "euler" -> "euler"; "dpm++_2m", "dpmpp_2m" -> "dpmpp_2m"; else -> null }
    /** the most used sampler and scheduler, and the median CFG and steps, of a version's sample pictures (their generation data) */
    fun metaDefaults(v: JSONObject): Advanced.Defaults? {
        val im = v.optJSONArray("images") ?: return null
        val metas = (0 until im.length()).mapNotNull { im.optJSONObject(it)?.optJSONObject("meta") }
        if (metas.isEmpty()) return null
        fun <T> most(xs: List<T>) = xs.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key
        fun med(xs: List<Double>) = xs.sorted().let { if (it.isEmpty()) null else it[it.size / 2] }
        val smp = most(metas.mapNotNull { m -> m.optString("sampler").takeIf { it.isNotEmpty() }?.let { sampler(it) } })
        val sch = most(metas.mapNotNull { m -> (m.optString("scheduler").ifEmpty { m.optString("Schedule type") }).lowercase().takeIf { it in Advanced.SCHEDULERS } })
        val cfg = med(metas.mapNotNull { it.optDouble("cfgScale").takeIf { c -> !c.isNaN() && c > 0 } })
        val steps = med(metas.mapNotNull { it.optInt("steps").takeIf { s -> s > 0 }?.toDouble() })
        if (smp == null && cfg == null && steps == null) return null
        val b = Advanced.BASE
        return Advanced.Defaults(smp ?: b.sampler, sch ?: b.scheduler, b.shift, (cfg ?: b.cfg.toDouble()).toFloat(), (steps ?: b.steps.toDouble()).toInt())
    }

    /** The site's sort names, in the order of the chips on screen. */
    val SORTS = listOf("Most Downloaded", "Highest Rated", "Newest")
    /** The site's name for each of the app's models. */
    private val BASE = mapOf("anima" to "Anima", "krea" to "Krea 2", "sdxl" to "SDXL 1.0")
    // a LoRA of the ranks these models use is a few hundred MB at most; the site also lists whole merged models as "LoRA"
    const val MAX_MB = 1024
    /** a whole model: Anima ~4 GB, Krea 2 ~25 GB in bf16 (an fp32 file is twice that) */
    const val MODEL_MAX_MB = 60000
    /** files bigger than a LoRA: the checkpoints, when the browser looks for models */
    @Volatile var models = false
    private fun maxMb() = if (models) MODEL_MAX_MB else MAX_MB

    private fun enc(s: String) = URLEncoder.encode(s, "UTF-8").replace("+", "%20")

    fun listUrl(model: String, sort: Int, query: String, cursor: String?) =
        "https://civitai.com/api/v1/models?types=${if (models) "Checkpoint" else "LORA"}&baseModels=${enc(BASE.getValue(model))}&sort=${enc(SORTS[sort])}&limit=20&nsfw=false" +
            (if (query.isBlank()) "" else "&query=${enc(query.trim())}") + (if (cursor == null) "" else "&cursor=${enc(cursor)}")

    /** One page of the listing: the entries that have a .safetensors for this model, and the cursor of the next page. */
    fun parse(json: String, model: String): Pair<List<Hit>, String?> {
        val o = JSONObject(json); val items = o.optJSONArray("items") ?: return emptyList<Hit>() to null
        val base = BASE.getValue(model); val hits = ArrayList<Hit>()
        for (i in 0 until items.length()) {
            val m = items.getJSONObject(i)
            if (m.optBoolean("nsfw")) continue
            val vs = m.optJSONArray("modelVersions") ?: continue
            // the newest version made for this model (an entry often carries versions for other models too)
            val v = (0 until vs.length()).map { vs.getJSONObject(it) }.firstOrNull { it.optString("baseModel") == base } ?: continue
            val fs = v.optJSONArray("files") ?: continue
            val cand = (0 until fs.length()).map { fs.getJSONObject(it) }.filter { (it.optString("name").endsWith(".safetensors", true) || models && it.optString("name").endsWith(".gguf", true)) && it.optString("downloadUrl").startsWith("https://civitai.com/") }
            // a model: every file the version offers (fp16 / fp32 / fp8, pruned / full, GGUF ...), the best one kconv reads taken, not the first listed
            val srcs = if (!models) emptyList() else cand.filter { !ModelSrc.isPart(it.optString("name")) }.map { ModelSrc.Src(it.optString("name"), (it.optDouble("sizeKB", 0.0) / 1024).toInt(), civitaiFmt(it), it.optString("downloadUrl")) }
            val pickSrc = if (models) ModelSrc.best(srcs.filter { it.mb <= maxMb() }) ?: continue else null
            val f = if (pickSrc != null) cand.first { it.optString("downloadUrl") == pickSrc.url } else cand.firstOrNull() ?: continue
            val mb = (f.optDouble("sizeKB", 0.0) / 1024).toInt()
            if (mb > maxMb()) continue
            hits.add(Hit(m.optString("name").trim(), m.optJSONObject("creator")?.optString("username") ?: "", m.optJSONObject("stats")?.optLong("downloadCount") ?: 0,
                         mb, f.optString("name"), f.optString("downloadUrl"), "https://civitai.com/models/${m.optLong("id")}").also { it.words = v.optJSONArray("trainedWords")?.let { w -> (0 until w.length()).map { w.optString(it).trim() }.filter { it.isNotEmpty() } }.orEmpty(); if (models) { it.defaults = metaDefaults(v); it.srcs = srcs; it.fmt = pickSrc!!.fmt } })
        }
        return hits to o.optJSONObject("metadata")?.optString("nextCursor")?.takeIf { it.isNotEmpty() }
    }

    /** what a file is, before its header is read ([probe]): its name first (pages label an nvfp4 file "bf16" or "fp8_scaled"),
     *  then Civitai's metadata.fp */
    private fun civitaiFmt(f: JSONObject): String {
        val name = f.optString("name"); if (name.endsWith(".gguf", true)) return ModelSrc.nameFmt(name) ?: "gguf"
        ModelSrc.nameFmt(name)?.let { return it }
        return when (val fp = f.optJSONObject("metadata")?.optString("fp")?.lowercase() ?: "") { "fp16" -> "f16"; "fp32" -> "f32"; "bf16" -> "bf16"
            "fp8", "fp8_scaled", "fp8_mixed" -> "fp8"; "" -> "f16"; else -> fp }
    }

    /** a model's files told by their headers (a few hundred KB each, read through the site's redirect); a file that will not
     *  answer (behind a login) keeps what its name and page said */
    fun probe(srcs: List<ModelSrc.Src>): List<ModelSrc.Src> {
        val pool = java.util.concurrent.Executors.newFixedThreadPool(4)
        val out = srcs.take(12).map { s -> pool.submit<ModelSrc.Src> {
            ModelSrc.Src(s.path, s.mb, runCatching { ModelSrc.fromUrl(s.path, s.url) }.getOrNull()?.ifEmpty { null } ?: s.fmt, s.url) } }.map { it.get() }
        pool.shutdown(); return out
    }

    fun search(model: String, sort: Int, query: String, cursor: String?): Pair<List<Hit>, String?> {
        val c = URL(listUrl(model, sort, query, cursor)).openConnection() as HttpURLConnection
        c.connectTimeout = 15000; c.readTimeout = 30000
        try {
            if (c.responseCode != 200) throw java.io.IOException("civitai.com answered ${c.responseCode}")
            return parse(c.inputStream.bufferedReader().readText(), model)
        } finally { c.disconnect() }
    }

    /**
     * The user's own Civitai API key (made in their account settings), or null. With it, files whose authors require a login
     * download too. It is sent to civitai.com only: the storage the site redirects to gets a signed address, not the key.
     */
    @Volatile var apiKey: String? = null

    /** The site's answer to a download request: the storage address it redirects to, or the status when it does not. */
    private fun ask(url: String, timeout: Int = 15000): Pair<Int, String?> {
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = timeout; c.readTimeout = timeout; c.instanceFollowRedirects = false; c.requestMethod = "GET"
        apiKey?.takeIf { it.isNotBlank() }?.let { c.setRequestProperty("Authorization", "Bearer ${it.trim()}") }
        try { return c.responseCode to c.getHeaderField("Location") } finally { c.disconnect() }
    }

    /** Whether a file needs a login (true), downloads as it is (false), or the site did not say (null). Asks without fetching it. */
    fun needsLogin(hit: Hit): Boolean? = runCatching { ask(hit.url, 8000) }.getOrNull()?.let { (code, loc) ->
        when { code in 300..399 && loc != null -> false; code == 401 || code == 403 -> true; else -> null }
    }

    /** Fetches the file; the site redirects to its storage. 401 means the author requires a login. */
    fun download(hit: Hit, dst: File, stop: () -> Boolean, progress: (Long, Long) -> Unit) {
        val (code, loc) = if (hit.repo != null) 302 to hit.url else ask(hit.url)
        if (code == 401 || code == 403) throw java.io.IOException(LOGIN)
        if (code !in 300..399 || loc == null) throw java.io.IOException("civitai.com answered $code")
        val c = URL(loc).openConnection() as HttpURLConnection
        c.connectTimeout = 15000; c.readTimeout = 30000; c.instanceFollowRedirects = true
        try {
            if (c.responseCode != 200) throw java.io.IOException("the file host answered ${c.responseCode}")
            val total = c.contentLengthLong; var got = 0L; var shown = 0L
            if (total > maxMb() * 1048576L) throw java.io.IOException("the file is larger than ${maxMb()}MB")
            c.inputStream.use { i -> dst.outputStream().use { o ->
                val buf = ByteArray(1 shl 16)
                while (true) {
                    if (stop()) throw java.io.IOException("stopped")
                    val n = i.read(buf); if (n < 0) break
                    o.write(buf, 0, n); got += n
                    if (got > maxMb() * 1048576L) throw java.io.IOException("the file is larger than ${maxMb()}MB")
                    if (got - shown >= (1 shl 20)) { shown = got; progress(got, total) }
                }
            } }
            if (total > 0 && got != total) throw java.io.IOException("the download was cut short")
        } finally { c.disconnect() }
    }

    const val LOGIN = "login required"

    // ---------------------------------------------------------------- Hugging Face: public, ungated repositories only
    /** the base models whose fine-tunes (models) or adapters (LoRAs) the Hub tags with base_model:<relation>:<id> */
    private val HF_BASE = mapOf("anima" to listOf("circlestone-labs/Anima"), "krea" to listOf("krea/Krea-2-Turbo", "krea/Krea-2-Raw"), "moss" to listOf("OpenMOSS-Team/MOSS-Image-SFT", "OpenMOSS-Team/MOSS-Image-Base"), "klein" to listOf("black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-base-4B"),
        "klein9" to listOf("black-forest-labs/FLUX.2-klein-9B", "black-forest-labs/FLUX.2-klein-base-9B"),
        "sdxl" to listOf("stabilityai/stable-diffusion-xl-base-1.0"))
    /** repositories of forms the phone cannot convert (Apple MLX, TFLite / LiteRT, ONNX, Core ML, OpenVINO, TensorRT, packed 4-bit) */
    private val NOT_HERE = Regex("mlx|mflux|tflite|litert|onnx|coreml|openvino|tensorrt|nunchaku|svdq|(?<![a-z])(awq|gptq|bnb|nf4)(?![a-z])", RegexOption.IGNORE_CASE)
    private val HF_SORT = listOf("downloads", "likes", "lastModified")

    private fun get(url: String): String {
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = 15000; c.readTimeout = 30000
        try { if (c.responseCode != 200) throw java.io.IOException("huggingface.co answered ${c.responseCode}"); return c.inputStream.bufferedReader().readText() }
        finally { c.disconnect() }
    }

    /** repositories built on the model on screen: fine-tunes when [models], else adapters (LoRAs); the file is picked later ([hfFiles]) */
    fun searchHf(model: String, sort: Int, query: String): List<Hit> {
        val rel = if (models) "finetune" else "adapter"; val seen = HashSet<String>(); val hits = ArrayList<Hit>()
        for (base in HF_BASE.getValue(model)) {
            val url = "https://huggingface.co/api/models?filter=${enc("base_model:$rel:$base")}&sort=${HF_SORT[sort]}&direction=-1&limit=30" +
                (if (query.isBlank()) "" else "&search=${enc(query.trim())}")
            val a = org.json.JSONArray(get(url))
            for (i in 0 until a.length()) {
                val m = a.getJSONObject(i); val id = m.optString("id")
                if (id.isEmpty() || !seen.add(id) || m.optBoolean("private") || (m.opt("gated") ?: false) != false || models && NOT_HERE.containsMatchIn(id)) continue
                hits.add(Hit(id.substringAfter('/'), id.substringBefore('/'), m.optLong("downloads"), 0, "", "", "https://huggingface.co/$id", id))
            }
        }
        return when (sort) { 0 -> hits.sortedByDescending { it.downloads }; else -> hits }
    }

    /** a repository's .safetensors (and for a model .gguf) files that fit. For a model each is told by its header (two small ranged
     *  requests, [ModelSrc.fromUrl]), the ones kconv reads first, best first; a checkpoint split in parts is listed once, as not
     *  readable yet. A LoRA's: largest first. */
    fun hfFiles(repo: String): List<ModelSrc.Src> {
        val a = org.json.JSONArray(get("https://huggingface.co/api/models/$repo/tree/main?recursive=true"))
        val all = (0 until a.length()).map { a.getJSONObject(it) }
            .filter { it.optString("type") == "file" && (it.optString("path").endsWith(".safetensors", true) || models && it.optString("path").endsWith(".gguf", true)) }
            .map { it.optString("path") to (it.optLong("size") shr 20).toInt() }
        fun url(path: String) = "https://huggingface.co/$repo/resolve/main/" + path.split('/').joinToString("/") { enc(it) }
        if (!models) return all.filter { it.second in 1..maxMb() }.sortedByDescending { it.second }.map { ModelSrc.Src(it.first, it.second, "", url(it.first)) }
        val part = Regex("-\\d{5}-of-\\d{5}\\.safetensors$")
        val parts = all.filter { part.containsMatchIn(it.first) && !ModelSrc.isPart(it.first) }.groupBy { it.first.replace(part, "") }
            .map { (b, fs) -> ModelSrc.Src("$b (${fs.size})", fs.sumOf { it.second }, "shards") }
        val pool = java.util.concurrent.Executors.newFixedThreadPool(4)
        val one = all.filter { !part.containsMatchIn(it.first) && !ModelSrc.isPart(it.first) && it.second in 1..maxMb() }.take(16).map { (p, mb) ->
            pool.submit<ModelSrc.Src> { ModelSrc.Src(p, mb, runCatching { ModelSrc.fromUrl(p, url(p)) }.getOrNull()?.ifEmpty { null } ?: ModelSrc.nameFmt(p) ?: "", url(p)) } }.map { it.get() }
        pool.shutdown()
        return ModelSrc.offered(one).ifEmpty { one + parts }   // nothing readable: all of them, for the chooser to say why
    }

    /** the hit's file chosen: its download address */
    fun hfChoose(hit: Hit, path: String, mb: Int, fmt: String = "") {
        hit.file = path.substringAfterLast('/'); hit.sizeMb = mb; hit.fmt = fmt
        hit.url = "https://huggingface.co/${hit.repo}/resolve/main/" + path.split('/').joinToString("/") { enc(it) }
    }
}
