# serve_models.py <manifest dir> [port] -- a local model server for testing the app's downloader.
# Serves manifest.json and, through sources.json, every listed file straight from where it lies on this PC
# (no copy), with HTTP Range so resuming works. Binds to localhost: reach it from the phone with
#   adb reverse tcp:8000 tcp:8000      and "http://127.0.0.1:8000" as the app's model server.
import sys, os, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

root = sys.argv[1]; port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
src = json.load(open(f"{root}/sources.json"))

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        u = unquote(self.path.lstrip("/").split("?")[0])
        p = f"{root}/manifest.json" if u == "manifest.json" else src.get(u)
        if not p or not os.path.exists(p): self.send_error(404); return
        size = os.path.getsize(p); start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0] or 0)
            if start >= size: self.send_error(416); return
            self.send_response(206); self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
        else: self.send_response(200)
        self.send_header("Content-Length", str(size - start)); self.send_header("Accept-Ranges", "bytes"); self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            while True:
                b = f.read(1 << 20)
                if not b: break
                try: self.wfile.write(b)
                except (BrokenPipeError, ConnectionResetError): return
    def log_message(self, fmt, *a): print(self.address_string(), fmt % a, flush=True)

print(f"serving {len(src)} files on 127.0.0.1:{port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
