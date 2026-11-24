package dev.kroma.open

/**
 * The models the app runs, by id (the pack ids of the model list, the folders under lora/ and custom/): what the screens and the
 * pictures call them. One table instead of the chains of ifs each screen had, which disagreed: a MOSS-Image or Klein fine-tune was
 * listed as "Anima", a Klein 9B LoRA was announced as "Krea 2 Turbo".
 *
 * [family]: the model on the model button and in lists ("Krea 2", "FLUX.2 Klein 9B"); [full]: the model a picture was made with,
 * for its metadata ("Krea 2 Turbo").
 */
enum class ModelKind(val id: String, val family: String, val full: String) {
    KREA("krea", "Krea 2", "Krea 2 Turbo"),
    ANIMA("anima", "Anima", "Anima Turbo"),
    MOSS("moss", "MOSS-Image", "MOSS-Image SFT + HaDeS Turbo"),
    KLEIN("klein", "FLUX.2 Klein 4B", "FLUX.2 Klein 4B"),
    KLEIN9("klein9", "FLUX.2 Klein 9B", "FLUX.2 Klein 9B"),
    SDXL("sdxl", "SDXL", "SDXL 1.0"),
    YUE("yue", "YuE2", "YuE2");

    companion object {
        fun of(id: String?): ModelKind? = entries.firstOrNull { it.id == id }
        /** [family] of the id; an id not in the table as it is */
        fun family(id: String?): String = of(id)?.family ?: id.orEmpty()
    }
}
