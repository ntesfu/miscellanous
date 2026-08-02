// Shared signaling client used by both the phone (sender) and viewer (receiver).
//
// Downstream messages arrive over Server-Sent Events (GET /events).
// Upstream messages are POSTed to /signal. Every message is a small JSON
// envelope: { from, role, to, type, payload }.

export function uid(prefix = "c") {
  return prefix + "_" + Math.random().toString(36).slice(2, 10);
}

export class Signaling {
  constructor(role) {
    this.role = role;
    this.id = uid(role);
    this.handlers = new Map();       // type -> Set(callback)
    this.es = null;
    this._onStatus = null;
  }

  onStatus(cb) { this._onStatus = cb; }
  _status(s) { if (this._onStatus) this._onStatus(s); }

  on(type, cb) {
    if (!this.handlers.has(type)) this.handlers.set(type, new Set());
    this.handlers.get(type).add(cb);
    return this;
  }

  _emit(type, msg) {
    const set = this.handlers.get(type);
    if (set) for (const cb of set) { try { cb(msg); } catch (e) { console.error(e); } }
  }

  connect() {
    const url = `/events?role=${encodeURIComponent(this.role)}&id=${encodeURIComponent(this.id)}`;
    this.es = new EventSource(url);
    this.es.onopen = () => this._status("connected");
    this.es.onerror = () => this._status("reconnecting"); // EventSource auto-retries
    this.es.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg && msg.type) this._emit(msg.type, msg);
    };
  }

  // Send an addressed message. `to` = target client id, or null to broadcast
  // to every client of the other role.
  async send(to, type, payload) {
    const body = JSON.stringify({ from: this.id, role: this.role, to, type, payload });
    try {
      await fetch("/signal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
    } catch (e) {
      console.warn("signal send failed", e);
    }
  }
}

export function fmtBytes(n) {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}

export function fmtClock(ms) {
  const t = Math.floor(ms / 1000);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  const pad = (x) => String(x).padStart(2, "0");
  return (h ? pad(h) + ":" : "") + pad(m) + ":" + pad(s);
}
