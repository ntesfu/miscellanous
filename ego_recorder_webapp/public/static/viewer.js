// Viewer (receiver) role — runs on the Mac.
//  * Receives the phone's live video over WebRTC and displays it.
//  * Sends control commands (start/stop record, switch camera, torch, snapshot,
//    quality) to the phone.
//  * Lists and plays recordings saved on this machine.

import { Signaling, fmtBytes, fmtClock } from "./common.js";

const ICE = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };
const el = (id) => document.getElementById(id);

const remote = el("remote");
const dot = el("dot");
const stateText = el("stateText");
const recDot = el("recDot");
const recTime = el("recTime");
const metaRes = el("metaRes");
const metaCam = el("metaCam");
const metaRtt = el("metaRtt");
const metaViewers = el("metaViewers");
const toast = el("toast");
const stage = el("stage");
const recList = el("recList");
const noSignal = el("noSignal");

const pcs = new Map(); // phoneId -> pc
let sig = null;
let mirror = false;
let recording = false;
let recStart = 0;
let recTimer = null;
let statsTimer = null;

function showToast(msg, ms = 2400) {
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toast.classList.remove("show"), ms);
}

function setConn(s) {
  dot.className = "dot " + (s === "connected" ? "ok" : s === "reconnecting" ? "warn" : "");
  stateText.textContent =
    s === "connected" ? "Server linked" :
    s === "reconnecting" ? "Reconnecting…" : "Connecting…";
}

// --------------------------------------------------------------------------- //
// WebRTC — viewer answers the phone's offer
// --------------------------------------------------------------------------- //
async function onOffer(msg) {
  const phoneId = msg.from;
  let pc = pcs.get(phoneId);
  if (pc) { try { pc.close(); } catch {} }
  pc = new RTCPeerConnection(ICE);
  pcs.set(phoneId, pc);

  pc.ontrack = (ev) => {
    remote.srcObject = ev.streams[0];
    remote.play().catch(() => {});
    noSignal.classList.add("hide");
    startStats(pc);
  };
  pc.onicecandidate = (e) => { if (e.candidate) sig.send(phoneId, "ice", e.candidate.toJSON()); };
  pc.onconnectionstatechange = () => {
    if (["failed", "disconnected", "closed"].includes(pc.connectionState)) {
      if (remote.srcObject) return; // keep last frame; phone will re-offer
    }
  };

  await pc.setRemoteDescription(msg.payload);
  const answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);
  sig.send(phoneId, "answer", { sdp: answer.sdp, type: answer.type });
  cmd("request_status");
}

async function onIce(msg) {
  const pc = pcs.get(msg.from);
  if (pc && msg.payload) await pc.addIceCandidate(msg.payload).catch(() => {});
}

function onPeerLeft(msg) {
  // Hub-generated peer-left carries `id` (not `from`, which only routed
  // offer/answer/ice/status messages have).
  const pc = pcs.get(msg.id);
  if (pc) { try { pc.close(); } catch {} pcs.delete(msg.id); }
  if (pcs.size === 0) {
    noSignal.classList.remove("hide");
    stopStats();
    setRecordingUI(false);
    metaViewers.textContent = "—";
  }
}

// --------------------------------------------------------------------------- //
// Live stats (round-trip time + resolution)
// --------------------------------------------------------------------------- //
function startStats(pc) {
  stopStats();
  statsTimer = setInterval(async () => {
    try {
      const stats = await pc.getStats();
      let rtt = null, w, h, fps;
      stats.forEach((r) => {
        if (r.type === "candidate-pair" && r.state === "succeeded" && r.currentRoundTripTime != null)
          rtt = r.currentRoundTripTime;
        if (r.type === "inbound-rtp" && r.kind === "video") {
          w = r.frameWidth; h = r.frameHeight; fps = r.framesPerSecond;
        }
      });
      if (rtt != null) metaRtt.textContent = Math.round(rtt * 1000) + " ms";
      if (w && h) metaRes.textContent = `${w}×${h}` + (fps ? ` @ ${Math.round(fps)}` : "");
    } catch {}
  }, 1000);
}
function stopStats() { clearInterval(statsTimer); statsTimer = null; }

// --------------------------------------------------------------------------- //
// Commands to the phone
// --------------------------------------------------------------------------- //
function cmd(action, extra = {}) { sig.send(null, "cmd", { action, ...extra }); }

// The Name + Stage dropdowns decide the saved filename: {Name}-{Stage}-{N}.
// The server allocates the running number N when the recording is finalized.
function currentLabel() {
  return { person: el("person").value, mode: el("mode").value };
}
function updateNamePreview() {
  const p = el("namePreview");
  if (p) p.textContent = `${el("person").value}-${el("mode").value}-…`;
}

let lensSig = "";
function onStatus(msg) {
  const p = msg.payload || {};
  if (p.width && p.height) metaRes.textContent = `${p.width}×${p.height}` + (p.fps ? ` @ ${p.fps}` : "");
  metaCam.textContent = p.facing === "user" ? "Front" : "Rear";
  metaViewers.textContent = String(p.viewers ?? "—");
  el("btnTorch").classList.toggle("active", !!p.torch);
  if (p.recording && !recording) { recStart = Date.now() - (p.elapsedMs || 0); setRecordingUI(true); }
  if (!p.recording && recording) setRecordingUI(false);
  buildLensPills(p.lenses || [], p.deviceId, p.facing);
}

function buildLensPills(list, activeId, facing) {
  const row = el("lensRow"), pills = el("lensPills");
  const sig2 = JSON.stringify(list) + "|" + activeId + "|" + facing;
  if (sig2 === lensSig) return; // no change
  lensSig = sig2;
  if (list.length < 2) { row.style.display = "none"; return; }
  row.style.display = "flex";
  pills.innerHTML = "";
  for (const lens of list) {
    const active = activeId ? lens.deviceId === activeId
      : (facing === "user" ? lens.group === "front" : lens.group === "back" && lens.zoom === "1×");
    const b = document.createElement("button");
    b.className = "lens" + (active ? " active" : "");
    b.textContent = lens.zoom;
    b.addEventListener("click", () => cmd("set_lens", { deviceId: lens.deviceId }));
    pills.appendChild(b);
  }
}

function setRecordingUI(on) {
  recording = on;
  el("btnRec").classList.toggle("recording", on);
  el("btnRec").querySelector(".label").textContent = on ? "Stop" : "Record";
  recDot.classList.toggle("on", on);
  stage.classList.toggle("armed", on);
  // Lock the label while recording — it's captured at start; changing it
  // mid-take would be misleading since the filename is already decided.
  el("person").disabled = on;
  el("mode").disabled = on;
  clearInterval(recTimer);
  if (on) {
    recTimer = setInterval(() => (recTime.textContent = fmtClock(Date.now() - recStart)), 250);
  } else {
    recTime.textContent = "00:00";
  }
}

// --------------------------------------------------------------------------- //
// Recordings library
// --------------------------------------------------------------------------- //
async function refreshRecordings() {
  try {
    const r = await fetch("/api/recordings");
    const { recordings } = await r.json();
    recList.innerHTML = "";
    if (!recordings.length) {
      recList.innerHTML = `<li class="empty">No recordings yet.</li>`;
      return;
    }
    for (const rec of recordings) {
      const li = document.createElement("li");
      const isVid = /\.(mp4|webm)$/i.test(rec.name);
      li.innerHTML = `
        <div class="rmeta">
          <span class="rname" title="${rec.name}">${isVid ? "🎬" : "📸"} ${rec.name}</span>
          <span class="rsub">${fmtBytes(rec.size)}</span>
        </div>
        <div class="ractions">
          ${isVid
            ? `<button class="mini play" data-url="${rec.url}">Play</button>`
            : `<a class="mini" href="${rec.url}" target="_blank" rel="noopener">Open</a>`}
          <button class="mini discard" data-name="${rec.name}" style="color:#e5484d">Discard</button>
        </div>`;
      recList.appendChild(li);
    }
    recList.querySelectorAll(".play").forEach((b) =>
      b.addEventListener("click", () => openPlayer(b.dataset.url)));
    recList.querySelectorAll(".discard").forEach((b) =>
      b.addEventListener("click", () => discardRecording(b.dataset.name)));
  } catch (e) {
    console.warn(e);
  }
}

// Move a recording into the recordings/discarded/ folder (kept off the library
// list, not deleted). Confirmed first since it disappears from view.
async function discardRecording(name) {
  if (!confirm(`Discard "${name}"?\nIt will be moved to the discarded folder.`)) return;
  try {
    const r = await fetch("/api/discard", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    showToast("Discarded: " + name);
    refreshRecordings();
  } catch (e) {
    showToast("Couldn't discard " + name);
    console.warn(e);
  }
}

function openPlayer(url) {
  const modal = el("player");
  el("playerVideo").src = url;
  modal.classList.add("show");
  el("playerVideo").play().catch(() => {});
}
function closePlayer() {
  el("player").classList.remove("show");
  el("playerVideo").pause();
  el("playerVideo").removeAttribute("src");
  el("playerVideo").load();
}

// --------------------------------------------------------------------------- //
// UI wiring
// --------------------------------------------------------------------------- //
function toggleFullscreen() {
  if (!document.fullscreenElement) stage.requestFullscreen?.().catch(() => {});
  else document.exitFullscreen?.();
}
function toggleMirror() {
  mirror = !mirror;
  remote.classList.toggle("mirror", mirror);
  el("btnMirror").classList.toggle("active", mirror);
}

function wireUI() {
  el("btnRec").addEventListener("click", () => {
    if (recording) return cmd("stop_record");
    cmd("start_record", { label: currentLabel() });
  });
  el("person").addEventListener("change", updateNamePreview);
  el("mode").addEventListener("change", updateNamePreview);
  updateNamePreview();
  el("btnSwitch").addEventListener("click", () => cmd("switch_camera"));
  el("btnTorch").addEventListener("click", () => cmd("set_torch", { on: !el("btnTorch").classList.contains("active") }));
  el("btnSnap").addEventListener("click", () => cmd("snapshot"));
  el("btnMute").addEventListener("click", () => {
    // Preview starts muted (required for autoplay); a user gesture may unmute.
    remote.muted = !remote.muted;
    const b = el("btnMute");
    b.classList.toggle("active", !remote.muted);
    b.querySelector(".ico").textContent = remote.muted ? "🔇" : "🔊";
    if (!remote.muted) remote.play().catch(() => {});
  });
  el("btnMirror").addEventListener("click", toggleMirror);
  el("btnFs").addEventListener("click", toggleFullscreen);
  el("btnRefresh").addEventListener("click", refreshRecordings);
  el("quality").addEventListener("change", (e) => {
    const map = {
      "720p": { res: "720p", fps: 30, bitrate: 5_000_000 },
      "1080p": { res: "1080p", fps: 30, bitrate: 8_000_000 },
      "1080p60": { res: "1080p", fps: 60, bitrate: 12_000_000 },
      "4k": { res: "4k", fps: 30, bitrate: 25_000_000 },
    };
    cmd("set_quality", map[e.target.value] || map["1080p"]);
    showToast("Requested " + e.target.value);
  });
  el("closePlayer").addEventListener("click", closePlayer);
  el("player").addEventListener("click", (e) => { if (e.target === el("player")) closePlayer(); });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "r" || e.key === "R") { el("btnRec").click(); }
    if (e.key === "s" || e.key === "S") { el("btnSnap").click(); }
    if (e.key === "f" || e.key === "F") { toggleFullscreen(); }
    if (e.key === "c" || e.key === "C") { el("btnSwitch").click(); }
    if (e.key === "Escape" && el("player").classList.contains("show")) closePlayer();
  });
}

function main() {
  wireUI();
  sig = new Signaling("viewer");
  sig.onStatus(setConn);
  sig
    .on("offer", onOffer)
    .on("ice", onIce)
    .on("peer-left", onPeerLeft)
    .on("status", onStatus)
    .on("recording_started", () => { showToast("Recording started"); })
    .on("recording_saved", (m) => { showToast("Saved: " + m.payload.name); refreshRecordings(); })
    .on("snapshot_saved", (m) => { showToast("Photo saved"); refreshRecordings(); });
  sig.connect();
  refreshRecordings();
  setInterval(() => { if (pcs.size) cmd("request_status"); }, 5000);
}

main();
