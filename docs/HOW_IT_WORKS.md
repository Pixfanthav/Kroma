# How Kroma works

The technical side of the app: what runs where, what is downloaded, and how the chips were checked.
For what the app does and how to install it, see the [README](../README.md).

## What runs where

Everything runs on the phone. The only network use is downloading model files, and LoRAs or fine-tunes when you ask for one.

The image models' transformer blocks, their text encoders and their VAEs run on the phone's NPU through Kroma's own engine, [Kengine](https://github.com/Pixfanthav/Kengine),
not through a vendor runtime. One set of weight files serves every supported chip. The weights are read from storage as
the blocks run; the blocks that fit in memory are kept there between steps.

| Model | What it does |
|---|---|
| Krea 2 Turbo | pictures in 8 steps |
| Anima Turbo | pictures in 10 steps |
| SDXL 1.0 | pictures, and SDXL fine-tunes including v-prediction ones |
| FLUX.2 [klein] 4B and 9B | pictures in 4 steps, and editing with up to three reference pictures |
| MOSS-Image | pictures in 8 or 4 steps (HaDeS Turbo LoRAs on MOSS-Image-SFT) |
| YuE2 | songs, with or without lyrics |

Prompts in Korean, Japanese, Chinese or Russian are translated on the phone (PLaMo 3 Translate 0.8B), and for Anima and SDXL
their short items are matched to Danbooru tags.

## Weight formats

The built-in models are fp16, as their makers published them. Krea 2, Anima, FLUX.2 [klein] and MOSS-Image can be downloaded
in Q8 or Q4 instead, about half or a quarter of the size; with those the text encoder comes in Q8. SDXL is fp16 only.

A fine-tune in `.safetensors` is converted on the phone, to fp16, Q8 or Q4. While it converts, it needs about twice the
file's size in free storage.

## LoRA

A LoRA is not merged into a copy of the model by default: its matrices are computed next to the weights as the blocks run,
so nothing is written and LoKr and DoRA files apply exactly. Merging into a kept copy is a setting, faster per picture at the
cost of storage.

## Downloads

Each model has its own Hugging Face repository, with its license and notice:
[Krea-2-Turbo-Kroma](https://huggingface.co/Pixfanthav/Krea-2-Turbo-Kroma),
[Anima-Turbo-Kroma](https://huggingface.co/Pixfanthav/Anima-Turbo-Kroma),
[SDXL-1.0-Kroma](https://huggingface.co/Pixfanthav/SDXL-1.0-Kroma),
[FLUX.2-klein-4B-Kroma](https://huggingface.co/Pixfanthav/FLUX.2-klein-4B-Kroma),
[FLUX.2-klein-9B-Kroma](https://huggingface.co/Pixfanthav/FLUX.2-klein-9B-Kroma),
[MOSS-Image-Kroma](https://huggingface.co/Pixfanthav/MOSS-Image-Kroma),
[YuE2-Kroma](https://huggingface.co/Pixfanthav/YuE2-Kroma), and the files every model shares in
[Kroma-Shared](https://huggingface.co/Pixfanthav/Kroma-Shared), whose `manifest.json` the app reads.

Downloads resume, and every file is checked against its SHA-256 before it is used. A model can be downloaded without its
built-in weights, only the text encoder and VAE, for using fine-tunes alone.

## Chips

Snapdragon 8 Gen 2, 8 Gen 3, 8 Elite and 8 Elite Gen 5.

Snapdragon 8s Gen 3, 8s Gen 4 and 7+ Gen 3 run Anima only, as a trial, not yet checked on a phone.

## Content filter

The model licenses require filtering prohibited content.

- Prompts that combine a minor with sexual content are refused.
- A finished picture goes through `Falconsai/nsfw_image_detection`. A flagged picture is blurred and not saved; flagged with
  a word for a minor in the prompt, it is discarded.
- Adults can choose in Settings to see and keep flagged pictures. Then a picture whose prompt has no word for a minor is not
  checked, since it would be shown either way; the rule about minors does not change.

It is a small classifier and a word list: it will sometimes be wrong in both directions.
