# Building Kroma from source

For developers. To just use the app, install the APK instead.

1. Android SDK 35 and JDK 17.
2. The native libraries are not in this repository. Put them in `app/src/main/jniLibs/arm64-v8a/`:
   - the Kroma engine, which runs the models on the NPU: `libkengine.so`, `libkengine_skel_v73.so`, `libkengine_skel_v75.so`,
     `libkengine_skel_v79.so`, `libkengine_skel_v81.so`, `libkblock.so` and `libkconv.so`, built from [Kengine](https://github.com/Pixfanthav/Kengine), which is the `engine` submodule (`git clone --recursive`).
   - prompt translation: `libllamacomp.so`, llama.cpp's `llama-completion` built with the NDK (MIT License).
   - YuE2 songs: `libyuesynth.so` (yue2.cpp's `yue-synth` with Kengine's `yue2/` patch), `libggml.so`, `libggml-base.so` and `libggml-cpu.so`.
   The libraries can be taken out of the APK (`lib/arm64-v8a/`).
3. `./gradlew assembleGithubDebug`. `assembleStoreDebug` builds the app-store edition instead: it has no setting for
   pictures the content filter flagged and no "close other apps".

The model files are downloaded by the app from the Hugging Face repositories listed in the README. `tools/serve_models.py`
serves a local folder with a `manifest.json` for testing the downloader, and `tools/export_nsfw.py` exports the content
classifier.
