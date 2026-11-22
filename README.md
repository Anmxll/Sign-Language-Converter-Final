# 🤟 SignBridge — Live Sign Language → English

SignBridge turns sign language into English text and speech, live from a webcam.
It tracks your hands in the browser, recognises signs and **ASL fingerspelling (A–Z)** that you teach it,
and can use **Google Gemini** to turn recognised words into natural sentences or read signs directly from camera frames.

- **No installs.** The backend uses only the Python standard library, and the frontend is plain HTML/CSS/JS.
- **Works offline.** Your own trained signs and letters are recognised on your device. Gemini is optional.
- **Private by design.** Hand tracking runs in the browser. The Gemini API key stays on the server.

---

## Contents
- [Quick start](#quick-start)
- [Modes](#modes)
- [Teaching signs and letters](#teaching-signs-and-letters)
- [Fingerspelling (Spell A–Z)](#fingerspelling-spell-az)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Backend API](#backend-api)
- [Saving and backing up your data](#saving-and-backing-up-your-data)
- [Troubleshooting](#troubleshooting)
- [Demo script](#demo-script)
- [Limitations and future work](#limitations-and-future-work)

---

## Quick start

**Requirements:** Python 3.8+, Google Chrome (or Edge), a webcam. You need internet the first time the page loads, to download the hand-tracking model.

```bash
cd signbridge
python server.py
```

Open **http://localhost:8000** in Chrome, press **▶ Start camera**, and allow camera access.

> Open the page through `http://localhost:8000`, not by double-clicking `index.html`. Browsers only allow camera access on `localhost` or `https`.

### Gemini API key (optional)
Gemini powers the *Hybrid* sentence polishing, *Gemini vision* mode and the fingerspelling spell-fix. Get a free key at https://aistudio.google.com/apikey, then either:

- paste it into the **Gemini** card in the app and press **Save**, or
- create a file `signbridge/.env` containing:
  ```
  GEMINI_API_KEY=your_key_here
  ```

`.env` is listed in `.gitignore`, so it is never committed. To use a different port, set `PORT`, e.g. `PORT=9000 python server.py`.

---

## Modes

Switch modes with the buttons under the video. The current mode is always shown in the top-left corner of the video.

| Mode | What it reads | Needs Gemini? | Best for |
|---|---|---|---|
| **Hybrid** | Your trained **word** signs. Gemini turns the words into a sentence. | Yes (falls back to raw words) | Natural sentences in a demo |
| **My signs** | Your trained **word** signs, shown as-is | No, fully on-device | Reliable offline demo |
| **Spell A–Z** | Your trained **ASL letters**, typed letter by letter | Optional (spell-fix) | Names, places, any word |
| **Gemini vision** | Camera frames sent to Gemini | Yes | Trying signs you haven't trained |

> **Hybrid and My signs only read words. Spell A–Z only reads letters.** If you recorded letters but only your words are recognised, switch to **Spell A–Z**.

Output options: **Speak** (reads text aloud), **Copy**, **Save .txt** and **Clear**. Every finished sentence is added to the history list with a timestamp.

---

## Teaching signs and letters

SignBridge recognises the signs **you** record, so it adapts to your hands, camera and lighting.

### Word signs ("Teach it your signs" card)
1. Start the camera.
2. Pick a preset word (Hello, Yes, Thank you…) or type your own.
3. Make the sign, press **● Record sign**, and hold still. It captures 25 frames.
4. Record each sign **2–3 times**, moving your hand slightly between recordings.

Use the **Strictness** slider to control how close a match has to be. Lower means stricter (fewer false matches); higher means looser.

### Letters ("Fingerspelling alphabet" card)
1. Start the camera.
2. Click a letter tile, or type the letter. The status box shows how to make that hand shape.
3. Make the shape, press **● Record letter**, and hold still.
4. Tiles turn **green** at 25 samples and **bright green** at 50. Two rounds per letter gives the best accuracy.

**J** needs no recording: it's the I shape plus a hook. For **Z**, record your pointing hand.

---

## Fingerspelling (Spell A–Z)

| Action | How |
|---|---|
| Type a letter | Hold the letter shape for about **0.7 s** (the bar under the video fills up) |
| Repeat a letter (the LL in HELLO) | Relax your hand briefly, then sign the letter again |
| **J** | Sign **I**, then hook your pinky down. A just-typed I becomes J. |
| **Z** | Make the pointing hand, then **draw a Z in the air** |
| Space | Lower your hand for **1 s**, or press **Space** |
| Finish the sentence | Lower your hand for **2.6 s**, or press **Done ✓** |
| Fix a mistake | **⌫ Delete** |
| Autocomplete | Tap a suggested word. Suggestions come from a built-in word list plus your history. |

With **Gemini spelling fix** on, finished text is sent to Gemini to correct misread letters, e.g. `helo my nane is anu` → `Hello my name is Anu`. Names and unknown words are kept as spelled.

Letters are recognised from **one hand**, and left hands are mirrored, so left- and right-handed signers can share the same recordings.

---

## How it works

```
                        ┌──────────────── Browser (public/) ────────────────┐
 Webcam ──► MediaPipe Hands ──► 21 3-D landmarks per hand
                        │                  │
                        │     normalise: wrist-relative, scale-invariant
                        │                  │
                        │      ┌───────────┼─────────────────────┐
                        │      ▼           ▼                     ▼
                        │  Word KNN     Letter KNN          Frame capture
                        │  (2 hands)    (1 hand, mirrored)  (JPEG frames)
                        │      │        + J/Z motion             │
                        │      ▼           ▼                     │
                        │    words      letters → words          │
                        └──────┼───────────┼─────────────────────┼──┘
                               ▼           ▼                     ▼
                         ┌──────────── server.py (Python) ────────────┐
                         │ /api/polish   /api/spellfix   /api/translate│
                         │   picks a working Gemini model, retries,    │
                         │   keeps the API key off the browser         │
                         └──────────────────┬──────────────────────────┘
                                            ▼
                                English text ──► screen + speech
```

### Recognition pipeline
1. **Hand tracking.** [MediaPipe Hands](https://developers.google.com/mediapipe) finds up to two hands and returns 21 (x, y, z) landmarks for each.
2. **Features.** Each landmark is made relative to the wrist and divided by the hand's size, so position in the frame and distance from the camera don't matter.
   - Words: both hands (sorted left→right) plus a two-hands flag, 127 values.
   - Letters: one hand, mirrored if it's a left hand, 63 values.
3. **Classification.** k-nearest neighbours (k = 5) compares against your recorded samples, with majority voting and a distance threshold (the Strictness slider).
4. **Debouncing.** A result must stay stable for 0.7 s before it's accepted, which prevents flicker between similar shapes.
5. **Motion letters.** For J and Z, the app tracks the pinky or index fingertip over the last 0.9 s. It only counts movement made *after* the I or Z shape was recognised, so moving your hand into the shape isn't mistaken for a J.
6. **Sentence building.** Lowering your hand ends a sentence. In Hybrid mode Gemini polishes the word list into a sentence; in Spell mode the optional spell-fix runs.

### Backend reliability
`server.py` asks Google which Gemini models your key can use, ranks Flash models newest-first, and **tries up to 8 of them** in turn. Each gets 20 s before the next is tried, so a busy or retired model doesn't stall the app. The last model that worked is tried first next time. Static files are sent with `Cache-Control: no-store`, so the browser always runs the latest code.

---

## Project structure

```
signbridge/
├── server.py          # Backend: static file server + Gemini proxy (standard library only)
├── .env               # GEMINI_API_KEY=... (not committed)
├── .gitignore
├── README.md
└── public/
    ├── index.html     # Layout: video, controls, translation, trainers
    ├── style.css      # Styling
    └── app.js         # Camera, MediaPipe, features, KNN, modes, trainers
```

---

## Backend API

All endpoints are served by `server.py` on `http://localhost:8000`.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | — | `{ ok, hasKey, model }` |
| GET | `/api/models` | — | `{ models: [...], active }` or `{ error }` |
| POST | `/api/key` | `{ key }` | Checks the key against Google, saves it to `.env`: `{ ok, model, models }` |
| POST | `/api/translate` | `{ frames: [base64 JPEG, …] }` (max 10) | `{ text, model, clear }` |
| POST | `/api/polish` | `{ words: [...] }` | `{ text, model }`: words → natural sentence |
| POST | `/api/spellfix` | `{ text }` | `{ text, model }`: fixes misread fingerspelling |

When Gemini fails, endpoints return HTTP 502 with `{ error, status }`. The frontend shows the message and keeps working on-device.

---

## Saving and backing up your data

Recorded signs and letters are stored in the **browser** (`localStorage`), not in the project folder. Clearing browser data, or switching browser or computer, loses them.

- **Export** saves everything to `signbridge-signs.json` (`{ "signs": [...], "letters": [...] }`).
- **Import** restores it, and also accepts older files that contain only word signs.

Export before a demo, and keep the file with the project.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Buttons do nothing / new features missing | Press **Ctrl+Shift+R** to force-reload, and make sure you restarted `python server.py` after updating |
| "Camera blocked" | Click the lock icon in the address bar → allow Camera → reload |
| Camera doesn't start | Open via `http://localhost:8000`, not the file directly. Close other apps using the webcam. |
| "MediaPipe failed to load" | Check your internet connection or ad-blocker (the model loads from cdn.jsdelivr.net) |
| Letters not recognised, only words | Switch to **Spell A–Z** mode |
| Wrong letter (M/N/S/T/A/E look alike) | Record those letters again, lower the Strictness, use good front lighting |
| Gemini: "high demand" / timeouts | Google's servers are busy. The backend tries other models automatically; use **My signs** or **Spell A–Z** meanwhile. |
| Gemini: bad key / no quota | Make a new key at aistudio.google.com/apikey and save it in the app |
| Port 8000 already in use | Another server is running. Stop it, or run `PORT=8001 python server.py`. |

**Tips for accuracy:** good even lighting, a plain background, your whole hand in view, and samples recorded in the same place you'll demo.

---

## Demo script

1. **Status chips.** Show Camera, Hands and Gemini turning green.
2. **Teach words.** Record Hello, Yes, No, Thank you and Help (about 10 s each).
3. **Hybrid mode.** Sign a few words, lower your hands, and Gemini makes a sentence. Turn on **Speak**.
4. **Spell A–Z.** Fingerspell your name, and show J/Z motion and word suggestions.
5. **Offline fallback.** Turn off Wi-Fi after the page has loaded. **My signs** and **Spell A–Z** keep working.
6. **Robustness.** Mention the model fallback: if one Gemini model is busy, the next one is used.

---

## Limitations and future work

**Current limits**
- Recognises **static hand shapes** that you train, plus the two motion letters J and Z. It doesn't understand full ASL vocabulary, grammar, facial expressions or most motion-based signs.
- The classifier is per-user. Someone else's hands may need their own recordings.
- Similar handshapes (M/N/S/T/A/E, U/V/R, K/P, G/Q) can be confused by a 2-D webcam.
- Gemini vision reads still frames, which isn't enough for signs where movement matters.

**Future work**
- Sequence models (LSTM / Transformer) on landmark time series to recognise moving signs.
- Training on public datasets (e.g. WLASL, ASL Fingerspelling) so it works without per-user recording.
- MediaPipe Holistic (face + body) for location- and expression-dependent signs.
- An accuracy and confusion-matrix page for evaluation.
- Two-way conversation: speech → text/sign for the hearing participant.
- Indian Sign Language (ISL) support.

---

*Built as a capstone project. Hand tracking by Google MediaPipe; language features by Google Gemini.*

