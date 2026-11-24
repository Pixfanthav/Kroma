package dev.kroma.open

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.io.File
import java.nio.FloatBuffer

/**
 * The content filter the Krea 2 Community License asks of anyone who deploys the model (section 4.2), aimed at
 * what its Acceptable Use Policy prohibits. It cannot be switched off.
 *
 *  before: a prompt that sexualises minors is refused outright -- a youth term together with a sexual term.
 *  after : every finished picture goes through an image classifier (Falconsai/nsfw_image_detection, one of the
 *          classifiers the license names; ViT-base, 224px, CPU, well under a second).
 *            - flagged and the prompt carries a youth term -> discarded, never shown or saved
 *            - flagged otherwise -> shown blurred and not saved, unless the user confirmed being an adult in settings
 *
 * Adult content between adults is not prohibited by the policy, which is why that last case is a setting
 * and the first two are not. Depicting real people in sexual or deceptive contexts is prohibited too; a
 * classifier cannot tell who a face is, so that rule is stated on the first-run agreement instead.
 */
object Safety {
    /** both checks skipped: only a debug build on a test phone (MainActivity sets it), never a release */
    @Volatile var off = false
    private var sess: OrtSession? = null; private var sessPath = ""
    // Unambiguous words only. "girl", "teen", "schoolgirl" and the like are everyday words for adult
    // characters too and are deliberately not here: the classifier after generation is the second line.
    private val YOUTH = listOf("child", "children", "kid", "kids", "toddler", "infant", "preteen", "pre-teen", "underage",
        "minor", "minors", "loli", "lolicon", "shota", "shotacon", "little girl", "little boy",
        "elementary school", "middle school", "kindergarten", "kindergartener",
        "어린이", "아동", "유아", "초등학생", "초딩", "중학생", "중딩", "미성년", "로리", "쇼타", "유치원생", "여중생", "여고생", "남중생", "남고생")
    private val SEXUAL = listOf("nude", "naked", "nsfw", "sex", "sexual", "sexy", "erotic", "porn", "hentai", "explicit", "topless", "bottomless",
        "nipple", "nipples", "breasts", "genital", "genitals", "vagina", "penis", "pussy", "cock", "cum", "orgasm", "masturbat", "intercourse",
        "blowjob", "fellatio", "lingerie", "underwear", "panties", "bikini", "undress", "undressed", "spread legs", "bdsm", "bondage", "fetish", "lewd", "aroused", "seductive",
        "누드", "나체", "알몸", "벗은", "벗고", "섹스", "성행위", "성적", "야한", "에로", "포르노", "음란", "가슴", "유두", "성기", "속옷", "팬티", "비키니", "란제리", "자위", "삽입")
    // an explicit age below 18 next to "years old" or the Korean age words
    private val AGE = Regex("(?<![0-9])(1[0-7]|[1-9])\\s*(-|\\s)?(years?\\s*old|yo\\b|y/o|살|세)", RegexOption.IGNORE_CASE)

    private fun norm(p: String) = " " + p.lowercase().replace(Regex("[^\\p{L}\\p{N}/]+"), " ").trim() + " "
    private fun has(text: String, words: List<String>) = words.any { w ->
        if (w.any { it.code > 0x2E7F }) text.contains(w) else text.contains(" $w ") || (w.length >= 6 && text.contains(" $w"))
    }

    fun youth(prompt: String) = norm(prompt).let { has(it, YOUTH) } || AGE.containsMatchIn(prompt)

    /* What each refusal rests on, shown under it so the user knows it is a rule and whose rule it is -- per model, since only Krea 2's
     * license asks for a content filter: Anima's (CircleStone Non-Commercial) does not, so what is held back from
     * them is the app's own choice and says so. */
    /** sexual content with a minor: refused or discarded for every model */
    fun minorBasis(model: String) = when (model) {
        "krea" -> tr(R.string.basis_krea_2_community_license_4)
        "anima" -> tr(R.string.basis_the_law_and_the_circlestone)
        else -> tr(R.string.basis_the_law_child_sexual_abuse)
    }
    /** adult content held back until the user confirms being an adult */
    fun adultBasis(model: String) = if (model == "krea") minorBasis(model)
        else tr(R.string.basis_kroma_s_own_age_setting)

    /** Why the prompt is refused (reason, then basis), or null. */
    fun refuse(prompt: String, model: String): String? {
        if (off) return null
        val t = norm(prompt)
        return if (youth(prompt) && has(t, SEXUAL))
            tr(R.string.prompt_refused_nreason_it_combines) + minorBasis(model)
        else null
    }

    /** Probability the picture is NSFW, 0..1. Throws when the classifier is missing: the caller must not show the picture. */
    fun nsfw(root: File, bmp: Bitmap): Float {
        if (off) return 0f
        val f = File(root, "onnx/nsfw.onnx")
        if (!f.exists()) error("content classifier missing (onnx/nsfw.onnx)")
        val S = 224
        val px = IntArray(S * S); Bitmap.createScaledBitmap(bmp, S, S, true).getPixels(px, 0, S, 0, 0, S, S)
        val inp = FloatArray(3 * S * S)
        for (c in 0 until 3) { val sh = 16 - 8 * c; for (p in 0 until S * S) inp[c * S * S + p] = ((px[p] shr sh) and 0xFF) / 127.5f - 1f }   // (x/255 - 0.5) / 0.5
        val env = OrtEnvironment.getEnvironment()
        // the session is opened once and kept: opening it for each picture took most of the check's seconds
        val s = synchronized(this) { sess?.takeIf { sessPath == f.absolutePath } ?: env.createSession(f.absolutePath, OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) })
            .also { sess?.close(); sess = it; sessPath = f.absolutePath } }
        val logits = OnnxTensor.createTensor(env, FloatBuffer.wrap(inp), longArrayOf(1, 3, S.toLong(), S.toLong())).use { t ->
                s.run(mapOf(s.inputNames.first() to t)).use { r -> val b = (r[0] as OnnxTensor).floatBuffer; FloatArray(b.remaining()).also { b.get(it) } } }
        // id2label: 0 normal, 1 nsfw
        val m = maxOf(logits[0], logits[1]); val e0 = Math.exp((logits[0] - m).toDouble()); val e1 = Math.exp((logits[1] - m).toDouble())
        return (e1 / (e0 + e1)).toFloat()
    }

    const val THRESHOLD = 0.5f

    /** A heavy box blur at 1/24 scale: nothing of the picture's content survives, only its colours. */
    fun blur(bmp: Bitmap): Bitmap {
        val w = maxOf(2, bmp.width / 24); val h = maxOf(2, bmp.height / 24)
        return Bitmap.createScaledBitmap(Bitmap.createScaledBitmap(bmp, w, h, true), bmp.width, bmp.height, true)
    }
}
