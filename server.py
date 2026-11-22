#!/usr/bin/env python3
"""SignBridge backend. Pure Python standard library: no pip install needed.

- Serves the frontend from ./public
- Keeps the Gemini API key on the server (never sent to the browser)
- Auto-discovers a working Gemini model and retries with fallbacks
- Endpoints:
    GET  /api/health      -> status (key present, active model)
    POST /api/key         -> save API key to .env   {key}
    POST /api/translate   -> sign frames -> English  {frames:[b64 jpeg,...]}
    POST /api/polish      -> recognised words -> sentence {words:[...]}
"""
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(ROOT, "public")
ENV_PATH = os.path.join(ROOT, ".env")
API = "https://generativelanguage.googleapis.com/v1beta"
PORT = int(os.environ.get("PORT", "8000"))

_lock = threading.Lock()
_state = {"key": "", "models": [], "model": ""}


def load_env():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key and os.path.exists(ENV_PATH):
        with open(ENV_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    _state["key"] = key


def save_key(key):
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write(f"GEMINI_API_KEY={key}\n")
    with _lock:
        _state.update(key=key, models=[], model="")


class GeminiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def gemini_call(path, body=None, timeout=60):
    if not _state["key"]:
        raise GeminiError(401, "No Gemini API key set")
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "x-goog-api-key": _state["key"]},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode()).get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        raise GeminiError(e.code, msg)
    except urllib.error.URLError as e:
        raise GeminiError(0, f"Network error: {e.reason}")
    except (TimeoutError, OSError) as e:  # read timeouts are not URLError
        raise GeminiError(0, f"Timed out / network error: {e}")


def discover_models():
    """Ask Google which models this key can use; rank newest Flash first."""
    data = gemini_call("/models?pageSize=200")
    found = []
    for m in data.get("models", []):
        name = m.get("name", "").replace("models/", "")
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        if "flash" not in name or any(
            x in name for x in ("lite", "image", "tts", "live", "audio", "embedding", "thinking", "robotics", "computer")
        ):
            continue
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        ver = float(v.group(1)) if v else 0.0
        stable = 0 if re.search(r"(preview|exp|\d{2}-\d{2})", name) else 1
        found.append((ver, stable, name))
    found.sort(reverse=True)
    return [n for _, _, n in found]


def ensure_models(force=False):
    with _lock:
        if _state["models"] and not force:
            return list(_state["models"])
    models = discover_models()
    # Safety net if discovery returns nothing
    models = models or ["gemini-flash-latest"]
    with _lock:
        _state["models"] = models
        _state["model"] = models[0]
    return models


def generate(parts, temperature=0.2, max_tokens=300):
    """Try discovered models in order until one answers."""
    last = None
    for attempt in range(2):
        models = ensure_models(force=(attempt == 1))
        # Try the last model that worked first, then the rest
        active = _state["model"]
        if active in models:
            models = [active] + [m for m in models if m != active]
        for name in models[:8]:
            try:
                out = gemini_call(
                    f"/models/{name}:generateContent",
                    {
                        "contents": [{"parts": parts}],
                        # Gemini 3 "thinking" tokens count against this limit, so leave headroom
                        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens + 1024},
                    },
                    timeout=20,
                )
                cands = out.get("candidates") or []
                text = ""
                if cands:
                    text = " ".join(
                        p.get("text", "")
                        for p in cands[0].get("content", {}).get("parts", [])
                        if p.get("text") and not p.get("thought")
                    ).strip()
                with _lock:
                    _state["model"] = name
                return text, name
            except GeminiError as e:
                last = e
                if e.status in (400, 401, 403):  # key / request problem: other models will fail too
                    if "API key" in e.message or e.status in (401, 403):
                        raise
                continue
    raise last or GeminiError(500, "Gemini unavailable")


SIGN_PROMPT = (
    "You are an American Sign Language (ASL) interpreter. The images are sequential frames "
    "(oldest first) of one person signing. Identify fingerspelled letters or signs and give the "
    "most likely English meaning as a short natural sentence or word. "
    "Reply with ONLY the English text. If no sign is clearly visible, reply exactly: ..."
)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=PUBLIC, **k)

    def end_headers(self):
        # Never let the browser run a stale app.js / index.html after an update
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _json(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n > 25_000_000:
            raise ValueError("Body too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path.startswith("/api/health"):
            return self._json(200, {"ok": True, "hasKey": bool(_state["key"]), "model": _state["model"]})
        if self.path == "/api/models":
            try:
                return self._json(200, {"models": ensure_models(force=True), "active": _state["model"]})
            except GeminiError as e:
                return self._json(200, {"models": [], "error": e.message})
        return super().do_GET()

    def do_POST(self):
        try:
            data = self._body()
        except Exception as e:
            return self._json(400, {"error": str(e)})
        try:
            if self.path == "/api/key":
                key = str(data.get("key", "")).strip()
                if len(key) < 10:
                    return self._json(400, {"error": "Key looks too short"})
                old = _state["key"]
                with _lock:
                    _state.update(key=key, models=[], model="")
                try:
                    models = ensure_models(force=True)  # validates the key
                except GeminiError:
                    with _lock:
                        _state.update(key=old, models=[], model="")
                    raise
                save_key(key)
                with _lock:
                    _state.update(models=models, model=models[0])
                return self._json(200, {"ok": True, "model": models[0], "models": models[:5]})

            if self.path == "/api/translate":
                frames = [f for f in data.get("frames", []) if isinstance(f, str)][:10]
                if not frames:
                    return self._json(400, {"error": "No frames"})
                parts = [{"text": SIGN_PROMPT}] + [
                    {"inline_data": {"mime_type": "image/jpeg", "data": f}} for f in frames
                ]
                text, model = generate(parts, temperature=0.1, max_tokens=120)
                clear = bool(text) and text.strip() != "..."
                return self._json(200, {"text": text if clear else "", "model": model, "clear": clear})

            if self.path == "/api/polish":
                words = [str(w) for w in data.get("words", [])][:30]
                if not words:
                    return self._json(400, {"error": "No words"})
                prompt = (
                    "These words were recognised one by one from sign language, in order: "
                    + ", ".join(words)
                    + ". Turn them into one short, natural English sentence. Keep the meaning, add only "
                    "the small words needed. Reply with ONLY the sentence."
                )
                text, model = generate([{"text": prompt}], temperature=0.3, max_tokens=80)
                return self._json(200, {"text": text, "model": model})

            if self.path == "/api/spellfix":
                text = str(data.get("text", ""))[:500].strip()
                if not text:
                    return self._json(400, {"error": "No text"})
                prompt = (
                    "This text was fingerspelled letter by letter in ASL and read by a camera, so some letters "
                    "may be wrong, missing or doubled (commonly confused: M/N/S/T/A/E, U/V/R, G/H, K/P): "
                    + json.dumps(text)
                    + ". Fix only obvious recognition mistakes and add normal capitalisation. Keep names and "
                    "unknown words as spelled. Do not add new words. Reply with ONLY the corrected text."
                )
                fixed, model = generate([{"text": prompt}], temperature=0.1, max_tokens=80)
                return self._json(200, {"text": fixed or text, "model": model})

            return self._json(404, {"error": "Unknown endpoint"})
        except GeminiError as e:
            return self._json(502, {"error": e.message, "status": e.status})
        except Exception as e:  # keep the server alive
            return self._json(500, {"error": str(e)})


if __name__ == "__main__":
    load_env()
    print(f"SignBridge running -> http://localhost:{PORT}")
    print("Gemini key:", "found" if _state["key"] else "not set (add it in the app, or put GEMINI_API_KEY in .env)")
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
