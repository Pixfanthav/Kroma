package dev.kroma.open

import java.io.ByteArrayOutputStream
import java.util.zip.CRC32

/**
 * Generation settings carried inside the saved PNG as one iTXt chunk (UTF-8, so Korean prompts
 * survive), keyword "kroma", plain JSON text (about 1KB on a ~2MB image).
 */
object PngMeta {
    const val KEY = "kroma"
    private val SIG = byteArrayOf(-119, 80, 78, 71, 13, 10, 26, 10)

    /** [png] with the iTXt chunk inserted right after IHDR (bytes 8..32). */
    fun withText(png: ByteArray, text: String, key: String = KEY): ByteArray {
        val body = ByteArrayOutputStream().apply {
            write(key.toByteArray(Charsets.ISO_8859_1)); write(0)
            write(0); write(0)          // not compressed, method 0
            write(0); write(0)          // no language tag, no translated keyword
            write(text.toByteArray(Charsets.UTF_8))
        }.toByteArray()
        val type = "iTXt".toByteArray(Charsets.ISO_8859_1)
        val crc = CRC32().apply { update(type); update(body) }.value
        val out = ByteArrayOutputStream(png.size + body.size + 12)
        out.write(png, 0, 33)
        out.write(int(body.size)); out.write(type); out.write(body); out.write(int(crc.toInt()))
        out.write(png, 33, png.size - 33)
        return out.toByteArray()
    }


    /** The text stored under [key], or null (not a PNG, or saved before this existed). */
    fun read(png: ByteArray, key: String = KEY): String? {
        if (png.size < 8 || !png.copyOfRange(0, 8).contentEquals(SIG)) return null
        var p = 8
        while (p + 8 <= png.size) {
            val len = ((png[p].toInt() and 0xFF) shl 24) or ((png[p + 1].toInt() and 0xFF) shl 16) or
                      ((png[p + 2].toInt() and 0xFF) shl 8) or (png[p + 3].toInt() and 0xFF)
            val type = String(png, p + 4, 4, Charsets.ISO_8859_1)
            if (len < 0 || p + 12 + len > png.size) return null
            if (type == "iTXt") {
                val s = p + 8; val e = s + len
                val z = (s until e).firstOrNull { png[it].toInt() == 0 } ?: -1
                if (z > 0 && String(png, s, z - s, Charsets.ISO_8859_1) == key) {
                    // skip: 2 flag bytes, language tag\0, translated keyword\0
                    var q = z + 3
                    repeat(2) { while (q < e && png[q].toInt() != 0) q++; q++ }
                    return if (q <= e) String(png, q, e - q, Charsets.UTF_8) else null
                }
            }
            if (type == "IEND") return null
            p += 12 + len
        }
        return null
    }

    private fun int(v: Int) = byteArrayOf((v ushr 24).toByte(), (v ushr 16).toByte(), (v ushr 8).toByte(), v.toByte())
}
