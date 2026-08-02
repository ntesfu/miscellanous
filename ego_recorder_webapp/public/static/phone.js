// Phone (sender) role.
//  * Captures the camera with getUserMedia.
//  * Sends live video to any viewer over WebRTC (one PeerConnection per viewer).
//  * Records locally with MediaRecorder and streams the encoded chunks to the
//    server (/upload) so the file lands on the Mac in real time.

import { Signaling, fmtClock } from "./common.js";

const ICE = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };

const el = (id) => document.getElementById(id);
const localVideo = el("local");
const dot = el("dot");
const stateText = el("stateText");
const recPill = el("recPill");
const recTime = el("recTime");
const camLabel = el("camLabel");
const resLabel = el("resLabel");
const toast = el("toast");

const RES = {
  "720p": { width: 1280, height: 720 },
  "1080p": { width: 1920, height: 1080 },
  "4k": { width: 3840, height: 2160 },
};

const state = {
  facing: "environment",
  deviceId: null,      // when set, a specific physical lens is selected
  res: "1080p",
  fps: 30,
  bitrate: 8_000_000,
  audio: true,
  recording: false,
  torch: false,
};

let lenses = [];       // [{ deviceId, label, group, zoom, order }]
let stream = null;
let sig = null;
const peers = new Map(); // viewerId -> { pc, vSender, aSender }
let wakeLock = null;

// Recording state. The active MediaRecorder owns its own session id / seq /
// upload chain (captured in startRecording) so a second recording can never
// touch the first one's uploads. `recorder` stays non-null until onstop fully
// finalizes, which blocks a stop->start from clobbering the finished file.
let recorder = null;
let recStart = 0;
let recTimer = null;
let swapping = false;   // true while a camera/lens/quality swap is acquiring a stream

function showToast(msg, ms = 2200) {
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toast.classList.remove("show"), ms);
}

function setConn(s) {
  dot.className = "dot " + (s === "connected" ? "ok" : s === "reconnecting" ? "warn" : "");
  stateText.textContent =
    s === "connected" ? "Linked to server" :
    s === "reconnecting" ? "Reconnecting…" : "Connecting…";
}

// --------------------------------------------------------------------------- //
// Camera
// --------------------------------------------------------------------------- //
async function getStream() {
  const r = RES[state.res] || RES["1080p"];
  const video = {
    width: { ideal: r.width },
    height: { ideal: r.height },
    frameRate: { ideal: state.fps },
  };
  // Prefer an explicitly chosen lens; fall back to front/back facing mode.
  if (state.deviceId) video.deviceId = { exact: state.deviceId };
  else video.facingMode = { ideal: state.facing };
  const constraints = {
    audio: state.audio ? { echoCancellation: true, noiseSuppression: true } : false,
    video,
  };
  return navigator.mediaDevices.getUserMedia(constraints);
}

async function applyStream(newStream) {
  const old = stream;
  stream = newStream;
  localVideo.srcObject = stream;
  try { await localVideo.play(); } catch {}

  const vTrack = stream.getVideoTracks()[0];
  const aTrack = stream.getAudioTracks()[0] || null;

  // Swap tracks into every existing peer connection.
  for (const p of peers.values()) {
    if (p.vSender && vTrack) await p.vSender.replaceTrack(vTrack).catch(() => {});
    if (p.aSender && aTrack) await p.aSender.replaceTrack(aTrack).catch(() => {});
  }

  if (old && old !== stream) old.getTracks().forEach((t) => t.stop());
  updateLabels(vTrack);
  broadcastStatus();
}

function updateLabels(vTrack) {
  const s = (vTrack && vTrack.getSettings && vTrack.getSettings()) || {};
  if (s.width && s.height) resLabel.textContent = `${s.width}×${s.height} @ ${Math.round(s.frameRate || state.fps)}`;
  const active = lenses.find((l) => l.deviceId === state.deviceId);
  const base = state.facing === "environment" ? "Rear camera" : "Front camera";
  camLabel.textContent = active && active.group === "back" ? `Rear camera · ${active.zoom}` : base;
}

// --------------------------------------------------------------------------- //
// Lens selection (Ultra-Wide / Wide / Tele / Front)
// --------------------------------------------------------------------------- //
function classifyLens(d) {
  const l = (d.label || "").toLowerCase();
  if (l.includes("front")) return { deviceId: d.deviceId, group: "front", zoom: "Front", order: 100, virtual: false };
  const virtual = l.includes("dual") || l.includes("triple");
  let zoom = "1×", order = 1;
  if (l.includes("ultra")) { zoom = "0.5×"; order = 0; }
  else if (l.includes("tele")) { zoom = "2×"; order = 2; }
  return { deviceId: d.deviceId, group: "back", zoom, order, virtual };
}

async function enumerateLenses() {
  let devices = [];
  try { devices = await navigator.mediaDevices.enumerateDevices(); } catch { return; }
  let list = devices.filter((d) => d.kind === "videoinput" && d.deviceId).map(classifyLens);
  // If real ultra-wide / tele lenses are exposed, hide the virtual "Dual/Triple" combos.
  const hasPhysical = list.some((x) => x.group === "back" && x.zoom !== "1×");
  if (hasPhysical) list = list.filter((x) => !x.virtual);
  // De-duplicate by group+zoom.
  const seen = new Set();
  list = list.filter((x) => { const k = x.group + x.zoom; if (seen.has(k)) return false; seen.add(k); return true; });
  list.sort((a, b) => a.order - b.order);
  lenses = list;
  buildLensUI();
  updateLabels(stream && stream.getVideoTracks()[0]);
  broadcastStatus();
}

function isActiveLens(lens) {
  if (state.deviceId) return lens.deviceId === state.deviceId;
  if (state.facing === "user") return lens.group === "front";
  return lens.group === "back" && lens.zoom === "1×";
}

function buildLensUI() {
  const bar = el("lensbar");
  if (!bar) return;
  bar.innerHTML = "";
  if (lenses.length < 2) return;
  for (const lens of lenses) {
    const b = document.createElement("button");
    b.className = "lens" + (isActiveLens(lens) ? " active" : "");
    b.textContent = lens.zoom;
    b.addEventListener("click", () => selectLens(lens.deviceId));
    bar.appendChild(b);
  }
}

async function selectLens(deviceId) {
  if (state.recording) return showToast("Stop recording to change lens");
  if (swapping) return;
  const lens = lenses.find((l) => l.deviceId === deviceId);
  if (!lens) return;
  state.deviceId = deviceId;
  state.facing = lens.group === "front" ? "user" : "environment";
  try { await swapStream(); }
  catch (e) { showToast("Lens switch failed: " + e.message); }
}

async function initCamera() {
  const s = await getStream();
  await applyStream(s);
  requestWakeLock();
}

// --------------------------------------------------------------------------- //
// WebRTC — phone is always the offerer
// --------------------------------------------------------------------------- //
function makePeer(viewerId) {
  const pc = new RTCPeerConnection(ICE);
  const entry = { pc, vSender: null, aSender: null };
  peers.set(viewerId, entry);

  for (const track of stream.getTracks()) {
    const sender = pc.addTrack(track, stream);
    if (track.kind === "video") entry.vSender = sender;
    else entry.aSender = sender;
  }
  // Prefer a high bitrate on the LAN for the live preview.
  applyBitrate(entry);

  pc.onicecandidate = (e) => {
    if (e.candidate) sig.send(viewerId, "ice", e.candidate.toJSON());
  };
  pc.onconnectionstatechange = () => {
    if (["failed", "closed", "disconnected"].includes(pc.connectionState)) {
      // Viewer will re-offer on reconnect; nothing to do here.
    }
  };
  return entry;
}

function applyBitrate(entry) {
  if (!entry.vSender || !entry.vSender.getParameters) return;
  try {
    const params = entry.vSender.getParameters();
    if (!params.encodings || !params.encodings.length) params.encodings = [{}];
    params.encodings[0].maxBitrate = state.bitrate;
    entry.vSender.setParameters(params).catch(() => {});
  } catch {}
}

async function offerTo(viewerId) {
  let entry = peers.get(viewerId);
  if (!entry) entry = makePeer(viewerId);
  const offer = await entry.pc.createOffer();
  await entry.pc.setLocalDescription(offer);
  sig.send(viewerId, "offer", { sdp: offer.sdp, type: offer.type });
}

function dropPeer(viewerId) {
  const entry = peers.get(viewerId);
  if (entry) { try { entry.pc.close(); } catch {} peers.delete(viewerId); }
}

// --------------------------------------------------------------------------- //
// Recording — encode locally, stream chunks to the server
// --------------------------------------------------------------------------- //
function pickMime() {
  const prefs = [
    "video/mp4;codecs=h264,aac",
    "video/mp4",
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
  ];
  for (const m of prefs) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Upload one chunk with bounded retries so a transient Wi-Fi blip doesn't
// silently punch a hole in the recording. Ordering is preserved because the
// caller awaits this fully (including retries) before sending the next seq.
async function uploadChunk(session, seq, blob) {
  const url = `/upload?session=${encodeURIComponent(session)}&seq=${seq}`;
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: blob,
      });
      if (res.ok) return true;
      if (res.status === 404) return false; // session already finished/gone
    } catch { /* network error — retry */ }
    await sleep(300 * (attempt + 1));
  }
  return false;
}

async function startRecording(label) {
  // Synchronous re-entrancy guard: `recorder` is set/held until onstop fully
  // finalizes, and state.recording flips before any await, so a double-tap or a
  // rapid stop->start can't start a second recorder or clobber the previous file.
  if (state.recording || recorder || swapping || !stream) return;
  state.recording = true;

  const mime = pickMime();
  const ext = mime.startsWith("video/mp4") ? ".mp4" : ".webm";
  const session = "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  let seq = 0;
  let chain = Promise.resolve();
  let hadUploadError = false;

  const queueUpload = (blob) => {
    const s = seq++;
    chain = chain.then(async () => {
      const ok = await uploadChunk(session, s, blob);
      if (!ok) hadUploadError = true;
    });
  };

  try {
    const res = await fetch("/upload/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session, ext, meta: { mime, ...state, label: label || null } }),
    });
    if (!res.ok) throw new Error("start " + res.status);
  } catch (e) {
    state.recording = false;
    showToast("Couldn't start recording");
    return;
  }

  const opts = { videoBitsPerSecond: state.bitrate };
  if (mime) opts.mimeType = mime;
  let mr;
  try {
    mr = new MediaRecorder(stream, opts);
  } catch (e) {
    state.recording = false;
    // Close the just-opened server session so it isn't left dangling.
    fetch("/upload/finish", { method: "POST", headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ session }) }).catch(() => {});
    showToast("Recorder unsupported: " + e.message);
    return;
  }

  recorder = mr;
  mr.ondataavailable = (ev) => { if (ev.data && ev.data.size > 0) queueUpload(ev.data); };
  mr.onstop = async () => {
    await chain; // drain all pending / retrying chunk uploads first
    await fetch("/upload/finish", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session }),
    }).catch(() => {});
    if (recorder === mr) recorder = null; // now a new recording may start
    if (hadUploadError) showToast("Saved, but some chunks failed — file may be incomplete");
  };

  mr.start(1000); // fire ondataavailable roughly once per second
  recStart = Date.now();
  recPill.classList.add("on");
  el("btnRec").classList.add("recording");
  el("btnRec").querySelector(".label").textContent = "Stop";
  recTimer = setInterval(() => (recTime.textContent = fmtClock(Date.now() - recStart)), 250);
  showToast("Recording started");
  broadcastStatus();
  requestWakeLock();
}

function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  clearInterval(recTimer);
  recPill.classList.remove("on");
  recTime.textContent = "00:00";
  el("btnRec").classList.remove("recording");
  el("btnRec").querySelector(".label").textContent = "Record";
  // Leave `recorder` set; onstop clears it once the file is finalized. If it's
  // somehow already inactive (or stop throws), clear it now so a future
  // recording isn't blocked by the re-entrancy guard.
  try {
    if (recorder && recorder.state !== "inactive") recorder.stop();
    else recorder = null;
  } catch { recorder = null; }
  showToast("Recording saved to Mac");
  broadcastStatus();
}

// --------------------------------------------------------------------------- //
// Camera controls (only safe to change while not recording)
// --------------------------------------------------------------------------- //
async function swapStream() {
  // Serialize stream re-acquisition; startRecording refuses to start mid-swap so
  // it can never build a recorder on tracks we're about to stop.
  swapping = true;
  try { await applyStream(await getStream()); buildLensUI(); }
  finally { swapping = false; }
}

async function switchCamera() {
  if (state.recording) return showToast("Stop recording to switch camera");
  if (swapping) return;
  state.facing = state.facing === "environment" ? "user" : "environment";
  state.deviceId = null; // fall back to the facing-mode default lens
  try { await swapStream(); } catch (e) { showToast("Switch failed: " + e.message); }
}

async function setQuality(res, fps, bitrate) {
  if (res) state.res = res;
  if (fps) state.fps = fps;
  if (bitrate) state.bitrate = bitrate;
  for (const p of peers.values()) applyBitrate(p);
  if (state.recording || swapping) return; // don't disturb an active recording
  try { await swapStream(); } catch (e) { showToast("Quality change failed"); }
}

async function toggleTorch(on) {
  const track = stream && stream.getVideoTracks()[0];
  if (!track) return;
  const caps = track.getCapabilities ? track.getCapabilities() : {};
  if (!("torch" in caps)) return showToast("Torch not available on this camera");
  try {
    await track.applyConstraints({ advanced: [{ torch: on }] });
    const st = track.getSettings ? track.getSettings() : {};
    state.torch = "torch" in st ? !!st.torch : on; // reflect what actually happened
    broadcastStatus();
  } catch {
    showToast("Torch not supported on this camera");
  }
}

async function snapshot() {
  const track = stream && stream.getVideoTracks()[0];
  if (!track) return;
  const s = track.getSettings();
  const c = document.createElement("canvas");
  c.width = s.width || localVideo.videoWidth;
  c.height = s.height || localVideo.videoHeight;
  c.getContext("2d").drawImage(localVideo, 0, 0, c.width, c.height);
  const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
  if (!blob) return;
  await fetch("/snapshot", { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob });
  showToast("Photo saved to Mac");
}

// --------------------------------------------------------------------------- //
// Status + Wake Lock
// --------------------------------------------------------------------------- //
function broadcastStatus() {
  if (!sig) return; // signaling not up yet (e.g. during initial camera setup)
  const vTrack = stream && stream.getVideoTracks()[0];
  const s = (vTrack && vTrack.getSettings && vTrack.getSettings()) || {};
  sig.send(null, "status", {
    recording: state.recording,
    facing: state.facing,
    torch: state.torch,
    audio: state.audio,
    res: state.res,
    width: s.width,
    height: s.height,
    fps: Math.round(s.frameRate || state.fps),
    elapsedMs: state.recording ? Date.now() - recStart : 0,
    viewers: peers.size,
    deviceId: state.deviceId,
    lenses: lenses.map((l) => ({ deviceId: l.deviceId, zoom: l.zoom, group: l.group })),
  });
}

async function requestWakeLock() {
  try {
    if ("wakeLock" in navigator && !wakeLock) {
      wakeLock = await navigator.wakeLock.request("screen");
      wakeLock.addEventListener("release", () => (wakeLock = null));
    }
  } catch {}
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") requestWakeLock();
});

// --------------------------------------------------------------------------- //
// Wire up signaling
// --------------------------------------------------------------------------- //
function handleCmd(msg) {
  const p = msg.payload || {};
  switch (p.action) {
    case "start_record": startRecording(p.label); break;
    case "stop_record": stopRecording(); break;
    case "switch_camera": switchCamera(); break;
    case "set_torch": toggleTorch(!!p.on); break;
    case "set_lens": selectLens(p.deviceId); break;
    case "snapshot": snapshot(); break;
    case "set_quality": setQuality(p.res, p.fps, p.bitrate); break;
    case "request_status": broadcastStatus(); break;
  }
}

async function main() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    stateText.textContent = "Camera API unavailable (needs HTTPS)";
    return;
  }
  try {
    await initCamera();
    await enumerateLenses(); // labels are available now that permission is granted
  } catch (e) {
    stateText.textContent = "Camera blocked: " + e.message;
    showToast("Allow camera access, then reload");
    return;
  }

  sig = new Signaling("phone");
  sig.onStatus(setConn);
  sig
    .on("welcome", (m) => { (m.peers || []).forEach((pr) => pr.role === "viewer" && offerTo(pr.id)); broadcastStatus(); })
    .on("peer-joined", (m) => { if (m.role === "viewer") { offerTo(m.id); broadcastStatus(); } })
    .on("peer-left", (m) => dropPeer(m.id))
    .on("answer", async (m) => {
      const entry = peers.get(m.from);
      if (entry) await entry.pc.setRemoteDescription(m.payload).catch((e) => console.warn(e));
    })
    .on("ice", async (m) => {
      const entry = peers.get(m.from);
      if (entry && m.payload) await entry.pc.addIceCandidate(m.payload).catch(() => {});
    })
    .on("cmd", handleCmd);
  sig.connect();

  // Local phone-side buttons (optional convenience)
  el("btnRec").addEventListener("click", () => (state.recording ? stopRecording() : startRecording()));
  el("btnSwitch").addEventListener("click", switchCamera);
  el("btnTorch").addEventListener("click", () => toggleTorch(!state.torch));

  setInterval(broadcastStatus, 3000);
}

main();
