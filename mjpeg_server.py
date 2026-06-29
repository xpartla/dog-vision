"""Minimal MJPEG-over-HTTP display server (stdlib only).

OpenCV's Qt/Tk GUI windows do not render reliably under WSLg (frames only repaint
on a window-geometry change). Streaming frames to a browser sidesteps the Linux GUI
stack entirely: WSL2 forwards localhost to the VM, so the stream is viewable from a
Windows browser with zero extra setup.

Usage:
    server = MJPEGServer(port=8090)
    server.start()
    ...
    server.update(frame_bgr)   # call per frame
    ...
    server.stop()
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_PAGE = b"""<!doctype html><html><head><title>Dog Vision</title>
<style>html,body{margin:0;height:100%;background:#111;display:flex;
align-items:center;justify-content:center}img{max-width:100%;max-height:100vh}</style>
</head><body><img src="/stream"></body></html>"""


class MJPEGServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8090, quality: int = 80):
        self.host = host
        self.port = port
        self.quality = int(quality)
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._stop = False
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence per-request logging
                pass

            def do_GET(self):  # noqa: N802 (stdlib naming)
                if self.path.startswith("/stream"):
                    self._stream()
                elif self.path in ("/", "/index.html"):
                    self._send_bytes("text/html", _PAGE)
                else:
                    self.send_error(404)

            def _send_bytes(self, ctype: str, body: bytes):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _stream(self):
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=FRAME",
                )
                self.end_headers()
                try:
                    while not server._stop:
                        with server._cond:
                            server._cond.wait(timeout=1.0)
                            jpeg = server._jpeg
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--FRAME\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # browser tab closed/refreshed

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="mjpeg", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def update(self, frame_bgr) -> None:
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._cond.notify_all()

    def stop(self) -> None:
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()
