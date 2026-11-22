# SignBridge — Live Sign Language → English

## Run (2 steps, no installs)
```
python server.py
```
Open http://localhost:8000 in Chrome → **Start camera**.

Gemini key (optional): paste it in the app, or create `.env` with `GEMINI_API_KEY=your_key`.
Get a free key: https://aistudio.google.com/apikey

## How it works
```
Webcam → MediaPipe Hands (21 landmarks, in browser)
          ├─ My signs / Hybrid: on-device KNN classifier (trained by you) → words
          │        └─ Hybrid: Gemini text turns the words into a natural sentence
          └─ Gemini vision: captured frames → backend → Gemini → English
Backend (server.py): hides the API key, auto-picks a working Gemini model,
                     retries other models, returns clear errors.
```

## Fingerspelling mode (Spell A–Z)
1. Start the camera, scroll to **Fingerspelling alphabet**, click a letter (or type it), make the shape, press **Record letter** and hold steady. Do two rounds for better accuracy.
2. Switch to **Spell A–Z**. Hold a letter ~0.7 s to type it. Lower your hand 1 s = space, 2.6 s = finish the sentence.
3. Motion letters: **J** = sign I, then hook the pinky. **Z** = pointing hand, then draw a Z in the air.
4. Word suggestions appear while spelling; tap one to accept it. Gemini (optional) fixes misread letters (`/api/spellfix`).
5. Letters are one-handed and mirrored, so left- or right-handed signers both work. Export/Import saves signs and letters together.

## If Gemini does not work
- **Wrong/old model name:** the backend now lists models your key can use and picks the newest Flash automatically.
- **Bad key / no quota / no internet:** use **My signs** mode. It runs fully offline and is the most reliable for a live demo.
- Still-frame vision is weak for real ASL (motion matters). That is why the trainer exists.

## Demo script (viva)
1. Show the status chips (camera, hands, Gemini model).
2. Teach 4–5 signs (Hello, Yes, No, Thank you, Help). ~10 seconds each.
3. Hold signs in order. Words appear. Lower hands → Gemini turns them into a sentence.
4. Turn off Wi-Fi → My signs still works (offline fallback).

## Limits (say this honestly)
- Recognises static hand shapes you train, not full ASL grammar or motion.
- Future work: sequence model (LSTM/Transformer) on landmark time series, WLASL dataset, two-hand + face context.

## Files
- `server.py` backend (Python standard library only)
- `public/index.html`, `style.css`, `app.js` frontend
