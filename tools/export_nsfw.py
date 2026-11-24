# export_nsfw.py -- Falconsai/nsfw_image_detection (ViT-base, Apache-2.0) to ONNX for the app's content filter.
# Input "pixel_values" [1,3,224,224] = (rgb/255 - 0.5) / 0.5, output "logits" [1,2] (0 normal, 1 nsfw).
# Prints the logits of a grey image and of noise from both PyTorch and ONNX Runtime to show they agree.
import os, sys, numpy as np, torch, onnxruntime as ort
from transformers import AutoModelForImageClassification
out = sys.argv[1] if len(sys.argv) > 1 else "nsfw"; os.makedirs(out, exist_ok=True)
m = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection").eval()
print("labels", m.config.id2label)
class W(torch.nn.Module):
    def __init__(s): super().__init__(); s.m = m
    def forward(s, x): return s.m(pixel_values=x).logits
x = torch.zeros(1, 3, 224, 224)
torch.onnx.export(W(), (x,), f"{out}/nsfw.onnx", input_names=["pixel_values"], output_names=["logits"], opset_version=17, dynamo=False)
s = ort.InferenceSession(f"{out}/nsfw.onnx", providers=["CPUExecutionProvider"])
for name, t in (("grey", x), ("noise", torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0)).clamp(-1, 1))):
    with torch.no_grad(): a = m(pixel_values=t).logits.numpy()
    b = s.run(None, {"pixel_values": t.numpy()})[0]
    print(name, "torch", a.round(4), "onnx", b.round(4), "max diff %.2e" % np.abs(a - b).max())
print("size %.0f MB" % (os.path.getsize(f"{out}/nsfw.onnx") / 1e6))
