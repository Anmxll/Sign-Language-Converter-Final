/* SignBridge frontend
 * Pipeline: webcam -> MediaPipe Hands (21 landmarks) -> either
 *   (a) on-device KNN classifier trained by the user  -> words -> Gemini polishes into a sentence
 *   (b) frames sent to Gemini vision via our backend
 * Everything degrades gracefully: if Gemini fails, local mode still works offline.
 */
const $ = id => document.getElementById(id);
const v = $("v"), c = $("c"), ctx = c.getContext("2d");
const cap = document.createElement("canvas");

const S = {
  mode: "hybrid", running: false, hands: null,
  samples: JSON.parse(localStorage.getItem("sb_samples") || "[]"),
  recording: null,            // {label, left, buf}
  lastFeat: null,
  // gemini-vision state
  frames: [], lastCap: 0, busy: false,
  // local recogniser state
  cand: null, candSince: 0, committed: null, words: [], lastHand: 0, handsSeen: false,
  history: [], sentence: "", aiOk: false,
  // fingerspelling state
  letters: JSON.parse(localStorage.getItem("sb_letters") || "[]"),
  spell: { word: "", words: [] }, trail: [], lastShape: null, lastTyped: 0, lostSince: 0,
};
const HOLD_MS = 700, GAP_MS = 1800, REC_N = 25;
const SPACE_MS = 1000, END_MS = 2600;           // spell mode: hand down 1s = space, 2.6s = finish
const PRESETS = ["Hello", "Yes", "No", "Thank you", "Please", "Help", "I", "You", "Love", "Water", "Food", "Sorry", "Good", "Friend"];

/* ---------- ASL fingerspelling alphabet ---------- */
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const HINTS = {
  A: "Fist, thumb resting on the side", B: "Flat hand, fingers up together, thumb across palm",
  C: "Curve the hand into a C", D: "Index up, other fingertips touch the thumb",
  E: "Fingertips bent down onto the thumb", F: "Thumb + index touch, other 3 fingers up",
  G: "Index + thumb point sideways, parallel", H: "Index + middle point sideways together",
  I: "Pinky up, rest in a fist", J: "Make I, then hook the pinky down (motion — auto from I)",
  K: "Index + middle up in a V, thumb between them", L: "Thumb + index make an L",
  M: "Thumb tucked under 3 fingers", N: "Thumb tucked under 2 fingers",
  O: "All fingertips touch the thumb (O shape)", P: "Like K, but pointing down",
  Q: "Like G, but pointing down", R: "Index + middle crossed",
  S: "Fist, thumb across the front of the fingers", T: "Thumb tucked between index + middle",
  U: "Index + middle up, together", V: "Index + middle up, apart",
  W: "Index, middle, ring up, apart", X: "Index finger hooked",
  Y: "Thumb + pinky out", Z: "Record the pointing hand, then draw a Z in the air (motion)",
};
const TRAINABLE = LETTERS.filter(l => l !== "J");   // J = I-shape + hook motion
// Small dictionary for live word suggestions while spelling
const WORDLIST = ("hello hi bye yes no please thank thanks you sorry help good bad morning night afternoon evening " +
  "name my your me i we they he she it is am are was what where when why who how which this that here there " +
  "water food eat drink hungry thirsty sleep tired happy sad angry love like want need have go come stop wait " +
  "home school college work friend family mother father mom dad brother sister baby doctor hospital teacher student " +
  "class book phone help emergency pain sick medicine bathroom toilet today tomorrow yesterday now later time day week " +
  "again more less fine okay ok well nice meet see look learn sign language deaf hearing understand know think " +
  "can cannot will not do does did make take give get call open close start finish left right up down big small " +
  "hot cold new old fast slow money buy pay car bus train road street city india delhi mumbai chennai bangalore " +
  "kolkata hyderabad english hindi tamil project capstone computer camera hand bridge welcome excuse").split(" ");

/* ---------- UI helpers ---------- */
function chip(id, state, text) { const e = $(id); e.className = "chip " + state; if (text) e.lastChild.textContent = text; }
function status(t) { $("status").textContent = t; }
const MODE_NAMES = { hybrid: "Hybrid (words)", local: "My signs (words)", spell: "Spell A–Z (letters)", gemini: "Gemini vision" };
// Always show which mode is active, so it's clear whether words or letters are being read
function hud(t) { $("hud").textContent = MODE_NAMES[S.mode] + " · " + t; }

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}

/* ---------- Gemini status ---------- */
async function checkAI() {
  try {
    const h = await api("/api/health");
    if (!h.hasKey) { chip("chipAI", "warn", "Gemini: no key"); $("aiInfo").textContent = "Paste your API key and press Save. Local mode works without it."; S.aiOk = false; return; }
    const m = await api("/api/models");
    if (m.models && m.models.length) {
      S.aiOk = true; chip("chipAI", "ok", "Gemini: " + m.active);
      $("aiInfo").textContent = "Using " + m.active + ". Other available: " + m.models.slice(1, 4).join(", ");
    } else { S.aiOk = false; chip("chipAI", "bad", "Gemini: error"); $("aiInfo").textContent = m.error || "Could not list models. Check the key."; }
  } catch (e) { S.aiOk = false; chip("chipAI", "bad", "Backend offline"); $("aiInfo").textContent = "Start the server: python server.py"; }
}
$("saveKey").onclick = async () => {
  const key = $("key").value.trim(); if (!key) return;
  $("aiInfo").textContent = "Checking key…";
  try { const r = await api("/api/key", { key }); $("key").value = ""; S.aiOk = true; chip("chipAI", "ok", "Gemini: " + r.model); $("aiInfo").textContent = "Key saved. Using " + r.model; }
  catch (e) { S.aiOk = false; chip("chipAI", "bad", "Gemini: bad key"); $("aiInfo").textContent = "Error: " + e.message; }
};

/* ---------- Camera + MediaPipe ---------- */
// One button toggles the camera: ▶ Start camera  <->  ■ Stop camera
$("start").onclick = () => (S.running ? stopCamera() : startCamera());

async function startCamera() {
  const btn = $("start");
  btn.disabled = true; btn.textContent = "Starting…";
  try {
    if (!navigator.mediaDevices) throw new Error("Open http://localhost:8000 (not the file directly)");
    status("Requesting camera…");
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    v.srcObject = stream; await v.play();
    chip("chipCam", "ok");
    if (typeof Hands === "undefined") throw new Error("MediaPipe failed to load (internet / ad-blocker?)");
    c.width = v.videoWidth; c.height = v.videoHeight;
    cap.width = 512; cap.height = Math.round(512 * v.videoHeight / v.videoWidth);
    if (!S.hands) {   // load the hand model once; reuse it after Stop/Start
      S.hands = new Hands({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
      S.hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.5 });
      S.hands.onResults(onResults);
      status("Loading hand model (first time takes a few seconds)…");
    } else status("Camera on");
    S.running = true; S.lastHand = Date.now();
    btn.textContent = "■ Stop camera"; btn.classList.add("stop"); btn.disabled = false;
    hud("Show your hand");
    $("abcStatus").textContent = "Camera on. Click a letter below, then press “● Record letter”.";
    const gen = S.camGen = (S.camGen || 0) + 1;   // stops an old loop if the camera is restarted
    const loop = async () => {
      if (!S.running || gen !== S.camGen) return;
      try { if (v.readyState >= 2) await S.hands.send({ image: v }); } catch (e) { }
      requestAnimationFrame(loop);
    };
    loop();
  } catch (e) {
    stopCamera(true);
    chip("chipCam", "bad"); status("Error: " + (e.name || "") + " " + e.message);
    if (e.name === "NotAllowedError") status("Camera blocked. Click the lock icon in the address bar, allow Camera, reload.");
  }
}

function stopCamera(silent) {
  S.running = false;
  if (v.srcObject) { v.srcObject.getTracks().forEach(t => t.stop()); v.srcObject = null; }   // turns the webcam light off
  ctx.clearRect(0, 0, c.width, c.height);
  if (S.recording) {   // cancel a recording in progress
    S.recording = null; $("rec").disabled = false; $("recLetter").disabled = false; renderAbc();
  }
  S.cand = null; S.committed = null; S.trail = []; S.frames = []; S.handsSeen = false;
  $("holdfill").style.width = "0";
  const btn = $("start"); btn.textContent = "▶ Start camera"; btn.classList.remove("stop"); btn.disabled = false;
  chip("chipCam", ""); chip("chipHands", "", "Hands");
  hud("Press Start to begin");
  if (!silent) { status("Camera stopped"); $("abcStatus").textContent = "Camera off. Press “▶ Start camera” at the top to record letters."; }
}

const CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
function draw(hs) {
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.lineWidth = 3; ctx.strokeStyle = "#35e0b0"; ctx.fillStyle = "#6c8cff";
  hs.forEach(lm => {
    CONN.forEach(([a, b]) => { ctx.beginPath(); ctx.moveTo(lm[a].x * c.width, lm[a].y * c.height); ctx.lineTo(lm[b].x * c.width, lm[b].y * c.height); ctx.stroke(); });
    lm.forEach(p => { ctx.beginPath(); ctx.arc(p.x * c.width, p.y * c.height, 4, 0, 7); ctx.fill(); });
  });
}

/* ---------- Features + KNN ---------- */
function handFeat(lm) {
  const w = lm[0]; let m = 1e-6;
  lm.forEach(p => { m = Math.max(m, Math.hypot(p.x - w.x, p.y - w.y, (p.z - w.z))); });
  const f = []; lm.forEach(p => f.push((p.x - w.x) / m, (p.y - w.y) / m, (p.z - w.z) / m));
  return f;
}
function features(hs) {
  const sorted = [...hs].sort((a, b) => a[0].x - b[0].x).slice(0, 2);
  const f = handFeat(sorted[0]);
  const g = sorted[1] ? handFeat(sorted[1]) : new Array(63).fill(0);
  return f.concat(g, [sorted.length === 2 ? 1 : 0]);
}
// Letters use one hand only. Mirror left hands so either hand matches the same samples.
function letterFeat(lm, handed) {
  const flip = handed === "Left" ? -1 : 1;
  return handFeat(lm.map(p => ({ x: p.x * flip, y: p.y, z: p.z })));
}
function dist(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }
function classify(f, set = S.samples) {
  if (!set.length) return null;
  const k = 5;
  const near = set.map(s => ({ l: s.label, d: dist(f, s.f) })).sort((a, b) => a.d - b.d).slice(0, k);
  const votes = {}; near.forEach(n => votes[n.l] = (votes[n.l] || 0) + 1);
  const [label, n] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
  const ds = near.filter(x => x.l === label).map(x => x.d);
  const mean = ds.reduce((a, b) => a + b, 0) / ds.length;
  const thr = parseFloat($("strict").value) / 4;   // slider 2..12 -> 0.5..3
  const need = Math.min(3, Math.ceil(Math.min(k, set.filter(s => s.label === label).length) * 0.6));
  return n >= need && mean < thr ? { label, mean } : null;
}

/* ---------- Main per-frame handler ---------- */
function onResults(r) {
  if (!S.running) return;   // ignore a late result after Stop camera
  const hs = r.multiHandLandmarks || [];
  draw(hs);
  const now = Date.now();
  chip("chipHands", hs.length ? "ok" : "warn", hs.length ? `Hands: ${hs.length}` : "Hands: none");
  if (!hs.length) { onNoHands(now); return; }
  S.lastHand = now; S.handsSeen = true;
  const f = features(hs); S.lastFeat = f;
  const handed = (r.multiHandedness && r.multiHandedness[0] && r.multiHandedness[0].label) || "Right";
  const lf = letterFeat(hs[0], handed);

  if (S.recording) { recordTick(S.recording.set === "letters" ? lf : f); return; }
  if (S.mode === "spell") { onSpell(hs[0], lf, now); return; }

  // Gemini vision frame capture
  if ((S.mode === "gemini") && now - S.lastCap > 350 && S.frames.length < 8) { S.lastCap = now; grab(); }
  if (S.mode === "gemini") { hud("Signing… " + S.frames.length + " frames"); $("holdfill").style.width = "0"; return; }

  // Local recognition
  const res = classify(f);
  if (!res) { S.cand = null; $("holdfill").style.width = "0"; hud(S.samples.length ? "Hand seen (no match)" : "Train a sign first ↓"); return; }
  if (res.label !== S.cand) { S.cand = res.label; S.candSince = now; }
  const held = now - S.candSince;
  $("holdfill").style.width = Math.min(100, held / HOLD_MS * 100) + "%";
  hud(res.label + (S.committed === res.label ? " ✓" : ""));
  if (held >= HOLD_MS && S.committed !== res.label) { S.committed = res.label; addWord(res.label); }
}

function onNoHands(now) {
  $("holdfill").style.width = "0"; S.cand = null;
  if (!S.running) return;
  const gone = now - S.lastHand;
  if (S.mode === "spell" && !S.recording) {
    S.trail = [];
    if (gone > 400) S.committed = null;
    hud(S.letters.length ? "Show a letter" : "Record the alphabet first ↓");
    if (S.spell.word && gone > SPACE_MS) endWord();
    if (S.spell.words.length && gone > END_MS) finishSpell();
    return;
  }
  if (S.mode === "gemini") {
    hud("Show your hand");
    if (S.handsSeen && gone > 1200) { S.handsSeen = false; translateFrames(); }
    return;
  }
  if (gone > 600) S.committed = null;           // allows repeating the same word
  hud(S.recording ? "Show your hand to record" : "Show your hand");
  if (S.recording && S.recording.set === "letters" && gone > 800)
    $("abcStatus").textContent = `⏳ Waiting for your hand to record ${S.recording.label}: keep it fully inside the camera view (${S.recording.n}/${REC_N})`;
  if (S.words.length && gone > GAP_MS) finishSentence();
}

/* ---------- Local words -> sentence ---------- */
function addWord(w) {
  S.words.push(w); renderWords();
  $("live").className = "live"; $("live").textContent = S.words.join(" ");
}
function renderWords() { $("words").innerHTML = S.words.map(w => `<span>${w}</span>`).join(""); }

async function finishSentence() {
  const words = S.words; S.words = []; renderWords();
  let text = words.join(" ");
  if (S.mode === "hybrid" && S.aiOk && words.length > 1) {
    status("Gemini is polishing…");
    try { const r = await api("/api/polish", { words }); if (r.text) text = r.text; status("Done"); }
    catch (e) { status("Gemini unavailable, showing raw words (" + e.message.slice(0, 60) + ")"); }
  } else status("Done");
  commit(text);
}
function commit(text) {
  if (!text) return;
  S.history.unshift({ t: new Date().toLocaleTimeString(), text });
  $("live").className = "live"; $("live").textContent = text;
  $("history").innerHTML = S.history.map(h => `<div>${h.t} &nbsp;<b>${h.text}</b></div>`).join("");
  if ($("speak").checked) speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

/* ---------- Fingerspelling ---------- */
// How far a fingertip travelled recently, in palm-lengths (bounding box of its path)
// Only counts movement made AFTER the shape was recognised (so moving into the shape is ignored).
function extent(key, from) {
  const pts = S.trail.filter(p => p.t >= from);
  if (pts.length < 6) return { x: 0, y: 0 };
  const xs = pts.map(p => p[key + "x"]), ys = pts.map(p => p[key + "y"]);
  return { x: Math.max(...xs) - Math.min(...xs), y: Math.max(...ys) - Math.min(...ys) };
}
const baseShape = L => (L === "J" ? "I" : L);   // J is signed with the I hand shape
function onSpell(lm, lf, now) {
  const palm = Math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y) || 0.1;
  S.trail.push({ t: now, ix: lm[8].x / palm, iy: lm[8].y / palm, px: lm[20].x / palm, py: lm[20].y / palm });
  while (S.trail.length && now - S.trail[0].t > 900) S.trail.shift();

  if (!S.letters.length) { hud("Record the alphabet first ↓"); return; }
  const res = classify(lf, S.letters);
  if (res) { S.lastShape = { label: res.label, t: now }; S.lostSince = 0; }
  const recent = S.lastShape && now - S.lastShape.t < 600 ? S.lastShape.label : null;
  if (recent !== (S.streak && S.streak.label)) S.streak = recent ? { label: recent, since: now } : null;
  const from = S.streak ? S.streak.since + 150 : now;

  // Motion letters: J = I-shape + pinky hook, Z = pointing hand drawing a Z
  const pk = extent("p", from), ix = extent("i", from);
  if (recent === "I" && pk.x + pk.y > 0.9) { typeLetter("J", now); return; }
  if (recent === "Z" && ix.x > 0.9 && ix.y > 0.6) { typeLetter("Z", now); return; }

  if (!res) {
    if (!S.lostSince) S.lostSince = now;
    if (now - S.lostSince > 300) S.committed = null;   // relax the hand briefly to repeat a letter (e.g. LL)
    S.cand = null; $("holdfill").style.width = "0"; hud("Hand seen (no letter match)"); return;
  }
  if (res.label === "Z") { $("holdfill").style.width = "0"; hud("Z — now draw a Z in the air"); return; }
  if (res.label !== S.cand) { S.cand = res.label; S.candSince = now; }
  const held = now - S.candSince;
  $("holdfill").style.width = Math.min(100, held / HOLD_MS * 100) + "%";
  const done = S.committed && baseShape(S.committed) === res.label;
  hud("Letter " + res.label + (done ? " ✓" : ""));
  if (held >= HOLD_MS && !done) typeLetter(res.label, now);
}
function typeLetter(L, now) {
  if (S.committed === L) return;
  if (S.streak) S.streak.since = now;            // a new motion is needed for the next J/Z
  // J starts as an I: replace the I we just typed
  if (L === "J" && S.spell.word.endsWith("I") && now - S.lastTyped < 2500) S.spell.word = S.spell.word.slice(0, -1);
  S.spell.word += L; S.committed = L; S.lastTyped = now; S.trail = [];
  hud("Letter " + L + " ✓"); renderSpell();
}
function endWord(w) {
  const word = (w || S.spell.word).trim();
  if (word) S.spell.words.push(word.toUpperCase());
  S.spell.word = ""; renderSpell();
}
async function finishSpell() {
  if (S.spell.word) endWord();
  const words = S.spell.words; S.spell.words = []; renderSpell();
  if (!words.length) return;
  let text = words.join(" ").toLowerCase();
  if (S.aiOk && $("spellfix").checked) {
    status("Gemini is fixing spelling…");
    try { const r = await api("/api/spellfix", { text }); if (r.text) text = r.text; status("Done"); }
    catch (e) { status("Gemini unavailable, showing spelled text (" + e.message.slice(0, 60) + ")"); }
  } else status("Done");
  commit(text.charAt(0).toUpperCase() + text.slice(1));
}
function suggestions(prefix) {
  if (!prefix) return [];
  const p = prefix.toLowerCase();
  const mine = S.history.flatMap(h => h.text.toLowerCase().split(/\W+/));
  return [...new Set([...mine, ...WORDLIST])].filter(w => w.length > p.length && w.startsWith(p)).slice(0, 5);
}
function renderSpell() {
  const { word, words } = S.spell;
  $("live").className = (word || words.length) ? "live" : "live empty";
  $("live").textContent = (word || words.length) ? [...words, word + "▌"].join(" ") : "Spell a word…";
  $("suggest").innerHTML = suggestions(word).map(w => `<button>${w}</button>`).join("");
  $("suggest").querySelectorAll("button").forEach(b => b.onclick = () => endWord(b.textContent));
}
$("spSpace").onclick = () => endWord();
$("spBack").onclick = () => {
  if (S.spell.word) S.spell.word = S.spell.word.slice(0, -1);
  else if (S.spell.words.length) S.spell.word = S.spell.words.pop();
  S.committed = null; renderSpell();
};
$("spDone").onclick = () => finishSpell();

/* ---------- Gemini vision ---------- */
function grab() { cap.getContext("2d").drawImage(v, 0, 0, cap.width, cap.height); S.frames.push(cap.toDataURL("image/jpeg", 0.7).split(",")[1]); }
async function translateFrames() {
  if (S.busy) return;
  if (!S.frames.length && S.running) grab();
  if (!S.frames.length) { status("Start the camera first"); return; }
  const frames = S.frames; S.frames = []; S.busy = true; status("Gemini is reading " + frames.length + " frames…");
  try {
    const r = await api("/api/translate", { frames });
    if (r.clear) { commit(r.text); status("Done (" + r.model + ")"); } else status("Gemini saw no clear sign. Sign slower, closer, with good light.");
  } catch (e) {
    status("Gemini failed: " + e.message.slice(0, 120) + " — switch to “My signs” mode (works offline).");
    chip("chipAI", "bad", "Gemini: failed");
  }
  S.busy = false;
}
$("now").onclick = () => {
  if (S.mode === "gemini") translateFrames();
  else if (S.mode === "spell") finishSpell();
  else if (S.words.length) finishSentence();
  else status("Nothing to translate yet. Hold a trained sign.");
};

/* ---------- Mode switch ---------- */
function setMode(mode) {
  document.querySelectorAll("#mode button").forEach(x => x.classList.toggle("on", x.dataset.mode === mode));
  S.mode = mode; S.frames = []; S.words = []; renderWords(); status("Mode: " + MODE_NAMES[mode]);
  S.spell = { word: "", words: [] }; S.committed = null; S.cand = null; S.trail = [];
  const sp = mode === "spell";
  $("spellbar").hidden = !sp; $("suggest").innerHTML = "";
  $("useSpell").textContent = sp ? "✓ Spell A–Z mode is on" : "Use my letters → switch to Spell A–Z mode";
  $("useSpell").disabled = sp;
  if (sp) {
    renderSpell();
    if (!S.letters.length) status("Spell A–Z: record some letters first (Fingerspelling alphabet, below).");
  } else { $("live").className = "live empty"; $("live").textContent = "Waiting for signs…"; }
  hud(S.running ? "Show your hand" : "Press Start to begin");
}
document.querySelectorAll("#mode button").forEach(b => b.onclick = () => setMode(b.dataset.mode));
$("useSpell").onclick = () => { setMode("spell"); window.scrollTo({ top: 0, behavior: "smooth" }); };

/* ---------- Trainer ---------- */
function persist() { localStorage.setItem("sb_samples", JSON.stringify(S.samples)); renderSamples(); }
function renderSamples() {
  const counts = {}; S.samples.forEach(s => counts[s.label] = (counts[s.label] || 0) + 1);
  $("samples").innerHTML = Object.keys(counts).length
    ? Object.entries(counts).map(([l, n]) => `<span>${l} <em>${n} samples</em><button data-l="${l}" title="Delete">×</button></span>`).join("")
    : '<div class="hint">No signs yet. Pick a word, start the camera, then press Record.</div>';
  $("samples").querySelectorAll("button").forEach(b => b.onclick = () => { S.samples = S.samples.filter(s => s.label !== b.dataset.l); persist(); });
}
$("presets").innerHTML = PRESETS.map(p => `<button>${p}</button>`).join("");
$("presets").querySelectorAll("button").forEach(b => b.onclick = () => { $("label").value = b.textContent; $("label").focus(); });

$("rec").onclick = () => {
  const label = $("label").value.trim();
  if (!label) { status("Type or pick a word first."); return; }
  if (!S.running) { status("Start the camera first."); return; }
  if (S.recording) return;
  S.recording = { label, n: 0, set: "signs", start: Date.now() + 800 }; $("rec").disabled = true; status(`Recording “${label}”… hold the sign steady`);
};
function recordTick(f) {
  const R = S.recording;
  if (Date.now() < R.start) { hud(`Get ready: ${R.label}`); return; }
  (R.set === "letters" ? S.letters : S.samples).push({ label: R.label, f: f.map(x => Math.round(x * 1000) / 1000) });
  R.n++; hud(`Recording ${R.label} ${R.n}/${REC_N}`);
  $("holdfill").style.width = (R.n / REC_N * 100) + "%";
  if (R.set === "letters") {
    $("abcStatus").textContent = `● Recording ${R.label}: ${R.n}/${REC_N}, hold it steady`;
    const em = document.querySelector(`.tile[data-l="${R.label}"] em`); if (em) em.textContent = `${R.n}/${REC_N}…`;
  }
  if (R.n >= REC_N) {
    S.recording = null; $("rec").disabled = false; $("recLetter").disabled = false;
    if (R.set === "letters") { persistLetters(); abcMsg(`✓ Saved letter ${R.label}. ` + (nextLetter() ? `Not recorded yet: ${nextLetter()}.` : "Alphabet complete!") +
      (S.mode !== "spell" ? " To read letters, switch to “Spell A–Z” mode (button below). Hybrid / My signs only read your words." : "")); }
    else { status(`Saved “${R.label}”. Record it again slightly differently for better accuracy.`); persist(); }
  }
}

/* ---------- Alphabet trainer ---------- */
function letterCounts() { const c = {}; S.letters.forEach(s => c[s.label] = (c[s.label] || 0) + 1); return c; }
function nextLetter() {
  const c = letterCounts();
  return TRAINABLE.find(l => (c[l] || 0) < REC_N) || TRAINABLE.find(l => (c[l] || 0) < REC_N * 2) || null;
}
function persistLetters() { localStorage.setItem("sb_letters", JSON.stringify(S.letters)); renderAbc(); }
// Messages shown both under the video and inside the alphabet card (where the user is looking)
function abcMsg(t) { $("abcStatus").textContent = t; status(t); }
function recordLetter(L) {
  if (!S.running) { abcMsg("⚠ Camera is off. Press “▶ Start camera” at the top, wait for the hand model to load, then try again."); return; }
  if (S.recording) { abcMsg(`Still recording ${S.recording.label}, finish that first.`); return; }
  if (L === "J") { abcMsg("J needs no recording: sign I, then hook your pinky down. Record I instead."); return; }
  S.recording = { label: L, n: 0, set: "letters", start: Date.now() + 800 };
  $("recLetter").disabled = true;
  const t = document.querySelector(`.tile[data-l="${L}"]`); if (t) t.classList.add("rec");
  abcMsg(`Get ready: letter ${L}. ${HINTS[L]}. Show your hand to the camera…`);
}
function renderAbc() {
  const c = letterCounts();
  $("abcGrid").innerHTML = LETTERS.map(L => {
    const n = L === "J" ? (c.I || 0) : (c[L] || 0);
    const cls = n >= REC_N * 2 ? "good" : n >= REC_N ? "ready" : "";
    return `<button class="tile ${cls}" data-l="${L}" title="${HINTS[L]}"><b>${L}</b><small>${HINTS[L]}</small><em>${L === "J" ? "auto (uses I)" : n + " samples"}</em></button>`;
  }).join("");
  // Clicking a tile just selects the letter (like the word presets)
  $("abcGrid").querySelectorAll(".tile").forEach(t => t.onclick = () => {
    $("letter").value = t.dataset.l;
    $("abcStatus").textContent = `Selected ${t.dataset.l}: ${HINTS[t.dataset.l]}. Make the shape, then press Record letter.`;
  });
  const done = TRAINABLE.filter(l => (c[l] || 0) >= REC_N).length;
  $("abcProg").textContent = `${done}/${TRAINABLE.length} letters recorded`;
}
$("recLetter").onclick = () => {
  const L = $("letter").value.trim().toUpperCase();
  if (!/^[A-Z]$/.test(L)) { abcMsg("Type or click a single letter A–Z first."); return; }
  recordLetter(L);
};
$("letter").onkeydown = e => { if (e.key === "Enter") $("recLetter").click(); };
$("abcWipe").onclick = () => { if (confirm("Delete all recorded letters?")) { S.letters = []; persistLetters(); } };
$("strict").oninput = () => { $("strictVal").textContent = $("strict").value; };
$("strictVal").textContent = $("strict").value;
$("wipe").onclick = () => { if (confirm("Delete all trained signs?")) { S.samples = []; persist(); } };
$("export").onclick = () => download("signbridge-signs.json", JSON.stringify({ signs: S.samples, letters: S.letters }));
$("import").onclick = () => $("file").click();
$("file").onchange = async e => {
  try {
    const a = JSON.parse(await e.target.files[0].text());
    if (Array.isArray(a)) { S.samples = a; persist(); status("Imported " + a.length + " samples"); }   // old format
    else if (a && (a.signs || a.letters)) {
      if (Array.isArray(a.signs)) { S.samples = a.signs; persist(); }
      if (Array.isArray(a.letters)) { S.letters = a.letters; persistLetters(); }
      status(`Imported ${S.samples.length} sign samples, ${S.letters.length} letter samples`);
    } else status("Bad file");
  } catch (x) { status("Bad file"); }
  e.target.value = "";
};

/* ---------- Output actions ---------- */
function download(name, text) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text])); a.download = name; a.click(); }
$("copy").onclick = () => navigator.clipboard.writeText(S.history.map(h => h.text).reverse().join("\n")).then(() => status("Copied"));
$("save").onclick = () => download("transcript.txt", S.history.map(h => `${h.t}  ${h.text}`).reverse().join("\n"));
$("clear").onclick = () => {
  S.history = []; S.words = []; renderWords(); $("history").innerHTML = "";
  S.spell = { word: "", words: [] }; $("suggest").innerHTML = "";
  $("live").className = "live empty"; $("live").textContent = S.mode === "spell" ? "Spell a word…" : "Waiting for signs…";
};

/* ---------- init ---------- */
$("live").className = "live empty";
renderSamples(); renderAbc(); checkAI();
