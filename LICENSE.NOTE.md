# License notes

The application source code in this repository is licensed under the GNU General Public License, version 3
(see `LICENSE`).

**Additional permission under GNU GPL version 3 section 7.** The engine calls the system interfaces of the phone's
DSP (FastRPC, QuRT, HAP), which are part of the phone's software and proprietary to Qualcomm Technologies, Inc. If you
modify this program, or any covered work, by linking or combining it with those interfaces, you are granted permission
to convey the resulting work without their source code. They are not part of this repository.

**Public sources of the engine's hardware details.** The engine's NPU code is Kroma's own, written against these public,
openly licensed projects:

- The HMX matrix instructions of its fp16 routines are the instruction forms that llama.cpp's Hexagon backend
  (`ggml/src/ggml-hexagon`, MIT License) uses.
- The DMA descriptor format follows llama.cpp's `ggml-hexagon/htp/dma-queue.h` (MIT), and the DSP power request fields
  and values follow `ggml-hexagon/htp/main.c` (MIT).
- The FastRPC interface (remote handles, arguments, session control) follows the public quic/fastrpc project (now qualcomm/fastrpc, BSD-3-Clause).

llama.cpp: Copyright (c) 2023-2026 The ggml authors, MIT License. quic/fastrpc: Copyright (c) Qualcomm Innovation
Center, Inc., BSD-3-Clause.

The model files the app downloads are not covered by the GPL. Each is under its own license; see
`app/src/main/assets/NOTICE.txt`.

Snapdragon is a trademark of Qualcomm Incorporated. Kroma is an independent project, not affiliated with or endorsed by Qualcomm.
