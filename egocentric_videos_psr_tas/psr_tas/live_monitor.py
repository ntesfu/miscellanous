#!/usr/bin/env python3
r"""Local live dashboard for the IndustReal v4 reproduction pipeline.

Renders the full v4 architecture diagram (frozen encoders -> 2176-d fusion ->
DiffAct + Type heads -> decode -> outputs) with a live-training overlay: the
component currently training glows and shows its epoch/loss, data particles flow
along the active edges, frozen encoders show a snowflake, done stages get a check.
Plus a live loss curve and a metrics-vs-target panel. Reads the training log +
nvidia-smi. Stdlib only.

    python3 live_monitor.py [--port 8077]   ->  http://localhost:8077
Stop:  pkill -f 'live_monitor\.py'   (pattern quoted so it can't match your shell)
"""
import argparse, glob, json, os, re, subprocess, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT   = os.path.dirname(os.path.abspath(__file__))
GLOG   = os.path.join(ROOT, "logs", "extract_giant.log")
ILOG   = os.path.join(ROOT, "logs", "extract_iv2.log")
DLOG   = os.path.join(ROOT, "logs", "diffact_train.log")
GFEAT  = os.path.join(ROOT, "data_v2", "features")
IFEAT  = os.path.join(ROOT, "fusion", "data", "features_iv2")
FFEAT  = os.path.join(ROOT, "fusion", "data", "features")
DSFEAT = os.path.join(ROOT, "extern", "DiffAct", "datasets", "IndustReal-Fusion", "features")
DCFG   = os.path.join(ROOT, "extern", "DiffAct", "configs", "IndustReal-Fusion-S1.json")
RESULT = os.path.join(ROOT, "extern", "DiffAct", "result", "IndustReal-Fusion-S1")
EVALJS = os.path.join(ROOT, "eval_out", "results.json")
TLOG   = os.path.join(ROOT, "logs", "type_train.log")
TCFG   = os.path.join(ROOT, "extern", "DiffAct", "configs", "IndustReal-Type-Fusion-S1.json")
TRESULT= os.path.join(ROOT, "extern", "DiffAct", "result", "IndustReal-Type-Fusion-S1")
TOTAL  = 84
TARGET = {"Acc": 74.9, "Edit": 79.5, "F1@10": 83.1, "F1@25": 79.1, "F1@50": 70.1}

LPAT = re.compile(r'Epoch\s+(\d+)\s*-\s*Running Loss\s+([\d.eE+-]+)')
MPAT = re.compile(r'Epoch\s+(\d+)\s*-\s*decoder-agg-Test-([A-Za-z0-9@]+)\s+([\d.eE+-]+)')

def count(d, drop_starts=False):
    fs = glob.glob(os.path.join(d, "*.npy"))
    if drop_starts:
        fs = [f for f in fs if not f.endswith("_starts.npy")]
    return len(fs)

def pids(substr):
    try:
        return [int(x) for x in subprocess.run(["pgrep","-f",substr],capture_output=True,text=True,timeout=5).stdout.split()]
    except Exception:
        return []

def gpu():
    info = dict(util=0, used=0, free=0, total=0, giant_mem=0, iv2_mem=0, diffact_mem=0)
    try:
        q = subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.free,memory.total","--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        u, used, free, total = [int(float(x)) for x in q.split(",")]
        info.update(util=u, used=used, free=free, total=total)
    except Exception:
        pass
    apps = {}
    try:
        a = subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        for ln in a.splitlines():
            if "," in ln:
                p, mem = ln.split(","); apps[int(p.strip())] = int(mem.strip())
    except Exception:
        pass
    info["giant_mem"]   = sum(apps.get(p,0) for p in pids("scripts/01_extract_v2.py"))
    info["iv2_mem"]     = sum(apps.get(p,0) for p in pids("fusion/scripts/extract_iv2.py"))
    info["diffact_mem"] = sum(apps.get(p,0) for p in pids("IndustReal-Fusion-S1"))
    info["type_mem"]    = sum(apps.get(p,0) for p in pids("IndustReal-Type-Fusion-S1"))
    return info

def diffact():
    lossd={}; metrics={}
    if os.path.exists(DLOG):
        with open(DLOG, errors="ignore") as f:
            for ln in f:
                m = LPAT.search(ln)
                if m: lossd[int(m.group(1))]={"ep":int(m.group(1)),"loss":float(m.group(2))}
                m2 = MPAT.search(ln)
                if m2: metrics.setdefault(int(m2.group(1)),{})[m2.group(2)]=float(m2.group(3))
    loss=[lossd[e] for e in sorted(lossd)]
    total_ep=1200
    try: total_ep=int(json.load(open(DCFG)).get("num_epochs",1200))
    except Exception: pass
    running=bool(pids("IndustReal-Fusion-S1"))
    cur_ep=loss[-1]["ep"] if loss else -1
    best={}
    for e in metrics:
        for k,v in metrics[e].items():
            if k not in best or v>best[k]: best[k]=v
    return dict(loss=loss, metrics=[dict(ep=e, **metrics[e]) for e in sorted(metrics)],
                best=best, running=running, cur_ep=cur_ep, total_ep=total_ep,
                started=os.path.exists(DLOG) and (len(loss)>0 or running))

def typehead():
    lossd={}; metrics={}
    if os.path.exists(TLOG):
        with open(TLOG, errors="ignore") as f:
            for ln in f:
                m = LPAT.search(ln)
                if m: lossd[int(m.group(1))]={"ep":int(m.group(1)),"loss":float(m.group(2))}
                m2 = MPAT.search(ln)
                if m2: metrics.setdefault(int(m2.group(1)),{})[m2.group(2)]=float(m2.group(3))
    loss=[lossd[e] for e in sorted(lossd)]
    total_ep=401
    try: total_ep=int(json.load(open(TCFG)).get("num_epochs",401))
    except Exception: pass
    running=bool(pids("IndustReal-Type-Fusion-S1"))
    cur_ep=loss[-1]["ep"] if loss else -1
    return dict(loss=loss, metrics=[dict(ep=e, **metrics[e]) for e in sorted(metrics)],
                running=running, cur_ep=cur_ep, total_ep=total_ep,
                started=os.path.exists(TLOG) and (len(loss)>0 or running))

def eval_result():
    if os.path.exists(EVALJS):
        try:
            r=json.load(open(EVALJS)).get("results",{}).get("v4_fusion_diffact",{})
            return r.get("best") or r.get("metrics") or r
        except Exception: return None
    return None

def snapshot():
    gc=count(GFEAT,drop_starts=True); ic=count(IFEAT); fc=count(FFEAT); pc=count(DSFEAT)
    da=diffact(); th=typehead(); ev=eval_result()
    ex_done = gc>=TOTAL and ic>=TOTAL
    train_state = ("active" if da["running"] else
                   ("done" if (da["cur_ep"]>=da["total_ep"]-2 or os.path.exists(os.path.join(RESULT,"test_results_decoder-agg_epoch1000.npy"))) else
                    ("stopped" if da["loss"] else "pending")))
    stages=[
        dict(key="extract", label="Extract", state="done" if ex_done else "active", detail=f"giant {gc}/84 · IV2 {ic}/84"),
        dict(key="fuse", label="Fuse", state="done" if fc>=TOTAL else ("pending" if not ex_done else "active"), detail=f"{fc}/84 · 2176-d"),
        dict(key="prep", label="Prepare", state="done" if pc>=TOTAL else "pending", detail=f"{pc}/84"),
        dict(key="train", label="Train DiffAct", state=train_state, detail=(f"epoch {max(0,da['cur_ep'])}/{da['total_ep']}" if da["started"] else "1200 ep")),
        dict(key="eval", label="Eval", state=("done" if ev else "pending"), detail="F1@50 → 70.1"),
    ]
    return dict(snap=time.strftime("%H:%M:%S"), total=TOTAL, giant_count=gc, iv2_count=ic, fused_count=fc,
                giant_running=bool(pids("scripts/01_extract_v2.py")), iv2_running=bool(pids("fusion/scripts/extract_iv2.py")),
                diffact=da, typehead=th, stages=stages, target=TARGET, eval=ev, gpu=gpu())

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _send(self, body, ctype):
        b=body.encode(); self.send_response(200)
        self.send_header("Content-Type",ctype); self.send_header("Content-Length",str(len(b)))
        self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/data"): self._send(json.dumps(snapshot()),"application/json")
        else: self._send(PAGE,"text/html; charset=utf-8")

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>IndustReal v4 — Live Architecture</title>
<style>
  :root{--bg:#eef1f6;--surface:#fff;--surface-2:#f7f9fc;--ink:#171c28;--muted:#616b7d;--faint:#8b95a7;
    --hairline:#e2e7f0;--grid:#eceff5;--giant:#3b6fd6;--iv2:#d9822b;--acc:#5b63e6;--good:#12916a;--warn:#c67a12;--free:#c3ccdb;
    --c-frozen:#3b6fd6;--c-trained:#d9822b;--c-decode:#8256d0;--c-output:#3f9e6a;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,"SF Mono",SFMono-Regular,"JetBrains Mono",Menlo,Consolas,monospace;}
  @media(prefers-color-scheme:dark){:root{--bg:#0b0f16;--surface:#141b25;--surface-2:#1a222e;--ink:#e7ebf3;
    --muted:#98a2b5;--faint:#6a7688;--hairline:#243040;--grid:#1c2531;--giant:#6b9bf0;--iv2:#eaa04a;--acc:#8b93f2;
    --good:#35c592;--warn:#e0a13e;--free:#33404f;--c-frozen:#6b9bf0;--c-trained:#eaa04a;--c-decode:#a98ae8;--c-output:#57c088;}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);-webkit-font-smoothing:antialiased;line-height:1.45;height:100vh;overflow:hidden}
  .app{display:flex;flex-direction:column;height:100vh;max-width:1480px;margin:0 auto;padding:12px 18px 8px;gap:9px}
  .main{flex:1;display:flex;gap:14px;min-height:0}
  .col-left{flex:1.08;min-height:0;display:flex;align-items:center;justify-content:center}
  .col-right{flex:1;min-width:0;display:flex;flex-direction:column;gap:11px;overflow-y:auto}
  .col-right .card{padding:12px 14px}
  .htitle{display:flex;flex-direction:column;gap:0}
  .foot{font-size:11px;color:var(--faint);padding-top:2px;display:flex;flex-wrap:wrap;gap:4px 14px}.foot code{font-family:var(--mono);font-size:10.5px;color:var(--muted)}
  @media(max-width:900px){body{height:auto;overflow:auto}.app{height:auto}.main{flex-direction:column}.col-left{min-height:74vh}}
  .num{font-family:var(--mono);font-variant-numeric:tabular-nums}
  .eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);font-weight:600;margin-bottom:1px}
  header{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 16px}
  h1{font-size:22px;font-weight:650;letter-spacing:-.01em;margin:0}
  .sub{color:var(--muted);font-size:14px}
  .status{margin-left:auto;display:inline-flex;align-items:center;gap:7px;font-size:12.5px;font-weight:700;font-family:var(--mono)}
  .dot{width:8px;height:8px;border-radius:50%;animation:pulse 1.7s infinite}
  @keyframes pulse{0%{box-shadow:0 0 0 0 currentColor}70%{box-shadow:0 0 0 7px transparent}100%{box-shadow:0 0 0 0 transparent}}
  @media(prefers-reduced-motion:reduce){.dot,.eq rect,.glow,.edge.live{animation:none!important}}
  .card{background:var(--surface);border:1px solid var(--hairline);border-radius:14px;padding:16px}
  h2{font-size:15px;font-weight:600;margin:0}
  .swatch{width:9px;height:9px;border-radius:2.5px;display:inline-block}

  /* architecture */
  .archcard{padding:6px;height:100%;width:100%}
  .arch{width:100%;height:100%;display:block}
  .arch text{font-family:var(--sans)}
  .grpbox{fill:none;stroke-dasharray:6 6;stroke-width:1.5;rx:14;opacity:.8}
  .grplab{font-size:11.5px;font-weight:700;letter-spacing:.03em}
  .edge{fill:none;stroke:var(--hairline);stroke-width:2.2}
  .edge.done{stroke:var(--good);opacity:.85}
  .edge.live{stroke:var(--acc);stroke-dasharray:5 8;animation:march .7s linear infinite}
  .edge.future{stroke:var(--faint);stroke-dasharray:3 6;opacity:.4}
  @keyframes march{to{stroke-dashoffset:-13}}
  .nbox{stroke-width:2}
  .grp-neutral .nbox{fill:var(--surface-2);stroke:var(--hairline)}
  .grp-frozen  .nbox{fill:color-mix(in srgb,var(--c-frozen) 12%,var(--surface));stroke:var(--c-frozen)}
  .grp-trained .nbox{fill:color-mix(in srgb,var(--c-trained) 13%,var(--surface));stroke:var(--c-trained)}
  .grp-decode  .nbox{fill:color-mix(in srgb,var(--c-decode) 12%,var(--surface));stroke:var(--c-decode)}
  .grp-output  .nbox{fill:color-mix(in srgb,var(--c-output) 13%,var(--surface));stroke:var(--c-output)}
  .node.future{opacity:.45}
  .node.active .nbox{stroke:var(--acc);stroke-width:2.8}
  .node.active .glow{animation:glow 1.6s ease-in-out infinite}
  @keyframes glow{0%,100%{opacity:.12}50%{opacity:.5}}
  .ntitle{font-size:13px;font-weight:650;fill:var(--ink)}
  .nsub{font-size:10px;fill:var(--muted);font-family:var(--mono)}
  .badge{font-size:12px;font-weight:700}
  .btag{font-size:9px;font-weight:800;letter-spacing:.06em}
  .eq rect{animation:eq 1s ease-in-out infinite}
  @keyframes eq{0%,100%{transform:scaleY(.3)}50%{transform:scaleY(1)}}
  .particle{filter:drop-shadow(0 0 2px currentColor)}
  .legend text{font-size:11px;fill:var(--muted)}

  .row2{display:grid;grid-template-columns:1.5fr 1fr;gap:16px;margin-bottom:16px}
  @media(max-width:860px){.row2{grid-template-columns:1fr}}
  .charthead{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:4px}
  svg.plot{display:block;width:100%;height:auto}
  .track{height:8px;border-radius:5px;background:var(--surface-2);border:1px solid var(--hairline);overflow:hidden;margin-top:8px}
  .fill{height:100%;background:var(--acc);border-radius:5px 0 0 5px;transition:width .6s ease}
  .mrow{margin:12px 0}.mrow .top{display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:5px}
  .mrow .top .k{color:var(--muted);font-weight:600}.mrow .top .v{font-family:var(--mono)}
  .mtrack{position:relative;height:14px;border-radius:5px;background:var(--surface-2);border:1px solid var(--hairline)}
  .mfill{height:100%;border-radius:5px 0 0 5px;background:var(--acc);transition:width .6s ease}
  .mrow.head .mfill{background-color:var(--good)}
  .mtick{position:absolute;top:-3px;width:2px;height:20px;background:var(--ink);opacity:.6}
  .mtick:after{content:attr(data-t);position:absolute;top:-14px;left:50%;transform:translateX(-50%);font-size:9px;font-family:var(--mono);color:var(--muted)}
  .hint{font-size:11.5px;color:var(--faint);margin-top:10px}
  .enc{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;color:var(--muted)}.enc b{color:var(--ink);font-family:var(--mono)}
  .membar{display:flex;height:24px;border-radius:8px;overflow:hidden;border:1px solid var(--hairline);background:var(--surface-2);margin-top:10px}
  .memseg{display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:11px;font-weight:600;color:#fff;min-width:0;overflow:hidden;white-space:nowrap;transition:flex-grow .5s}
  .memseg.free{color:var(--muted)}
  .memkey{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:12px;color:var(--muted)}.memkey span{display:inline-flex;align-items:center;gap:6px}.memkey .num{color:var(--ink)}
  footer{margin-top:22px;font-size:12.5px;color:var(--faint);border-top:1px solid var(--hairline);padding-top:14px;display:flex;flex-wrap:wrap;gap:6px 18px}footer code{font-family:var(--mono);color:var(--muted);font-size:12px}
  #err{display:none;background:var(--warn);color:#fff;padding:8px 12px;border-radius:8px;font-size:13px;margin-bottom:14px}

  /* ===== ambient motion & glass — aesthetic layer ===== */
  .aurora{position:fixed;inset:-25%;z-index:0;pointer-events:none;filter:blur(70px) saturate(1.15);opacity:.5;
    background:
      radial-gradient(36% 40% at 20% 20%, color-mix(in srgb,var(--giant) 60%,transparent), transparent 62%),
      radial-gradient(34% 38% at 82% 24%, color-mix(in srgb,var(--iv2) 55%,transparent), transparent 62%),
      radial-gradient(40% 44% at 48% 86%, color-mix(in srgb,var(--acc) 60%,transparent), transparent 64%);
    background-repeat:no-repeat;animation:drift 24s ease-in-out infinite alternate}
  @media(prefers-color-scheme:dark){.aurora{opacity:.4}}
  @keyframes drift{
    0%{transform:translate3d(-3%,-2%,0) scale(1.05) rotate(0deg)}
    50%{transform:translate3d(4%,3%,0) scale(1.14) rotate(5deg)}
    100%{transform:translate3d(-2%,4%,0) scale(1.06) rotate(-4deg)}}
  .app{position:relative;z-index:1}
  .card{background:color-mix(in srgb,var(--surface) 84%,transparent);backdrop-filter:blur(12px) saturate(1.25);
    -webkit-backdrop-filter:blur(12px) saturate(1.25);position:relative;
    transition:transform .55s cubic-bezier(.2,.8,.2,1),border-color .55s,box-shadow .55s}
  .card:hover{transform:translateY(-3px);border-color:color-mix(in srgb,var(--acc) 45%,var(--hairline));
    box-shadow:0 14px 36px -20px color-mix(in srgb,var(--acc) 75%,transparent)}
  h1{background:linear-gradient(96deg,var(--ink) 20%,var(--acc) 44%,var(--giant) 60%,var(--ink) 82%);
    background-size:260% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;
    -webkit-text-fill-color:transparent;animation:huemove 9s ease-in-out infinite}
  @keyframes huemove{0%,100%{background-position:0 0}50%{background-position:100% 0}}
  .fill,.mfill{background-image:linear-gradient(100deg,transparent 28%,color-mix(in srgb,#fff 55%,transparent) 50%,transparent 72%);
    background-size:220% 100%;animation:shimmer 2.3s linear infinite}
  @keyframes shimmer{0%{background-position:130% 0}100%{background-position:-130% 0}}
  .edge.future{animation:marchslow 3.2s linear infinite}
  @keyframes marchslow{to{stroke-dashoffset:-9}}
  .grpbox{animation:marchslow 7s linear infinite reverse}
  .node.active{animation:floaty 3.4s ease-in-out infinite}
  @keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-2px)}}
  .pflow.amb circle{opacity:.5}
  @media(prefers-reduced-motion:reduce){
    .aurora,.fill,.mfill,h1,.node.active,.edge.future,.grpbox,.pflow *{animation:none!important}
    h1{color:var(--ink);-webkit-text-fill-color:var(--ink)}}
</style></head><body>
<div class="aurora" aria-hidden="true"></div>
<div class="app">
  <header>
    <div class="htitle"><span class="eyebrow">IndustReal v4 · live training · RTX 4090 · localhost</span>
      <h1>v4 — Architecture, Training Live</h1></div>
    <span class="status" id="status"></span>
  </header>
  <div id="err">Lost connection to <code>live_monitor.py</code> — is it still running?</div>
  <div class="main">
    <div class="col-left card archcard"><svg class="arch" id="arch" viewBox="0 0 940 1000" role="img" aria-label="v4 architecture, training live"></svg></div>
    <div class="col-right">
      <div class="card">
        <div class="charthead"><h2>Active head · training</h2><span class="sub num" id="epinfo"></span></div>
        <div class="track"><div class="fill" id="epfill"></div></div>
      </div>
      <div class="card">
        <div class="charthead"><h2>Test metrics vs target</h2></div>
        <div id="metrics"></div><div class="hint" id="mhint"></div>
      </div>
      <div class="card">
        <div class="charthead"><h2>Training loss</h2></div>
        <svg class="plot" id="loss" viewBox="0 0 640 230" role="img" aria-label="Training loss curve"></svg>
      </div>
      <div class="card">
        <div class="charthead"><h2>GPU memory</h2><span class="sub num" id="gpuutil"></span></div>
        <div class="membar" id="membar"></div><div class="memkey" id="memkey"></div>
      </div>
      <div class="foot" id="foot"></div>
    </div>
  </div>
</div>
<script>
const NS="http://www.w3.org/2000/svg", $=id=>document.getElementById(id);
const el=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
const gb=m=>(m/1024).toFixed(1);
const REDUCE=!!(window.matchMedia&&window.matchMedia('(prefers-reduced-motion:reduce)').matches);
const ORDER=["Acc","Edit","F1@10","F1@25","F1@50"];

/* ---------- v4 architecture (built once, then updated) ---------- */
let archBuilt=false, lossDrawn=false;
const N={
  video:{x:350,y:16,w:240,h:52,grp:'neutral',t:'Egocentric RGB video',s:'16-frame clips · stride 2'},
  giant:{x:100,y:150,w:300,h:64,grp:'frozen',t:'VideoMAEv2-giant',s:'SSv2 · motion · 1408-d'},
  iv2:{x:540,y:150,w:300,h:64,grp:'frozen',t:'InternVideo2-B14',s:'K710 · appearance · 768-d'},
  fusion:{x:320,y:292,w:300,h:56,grp:'neutral',t:'Feature fusion',s:'L2 + concat · 2176-d'},
  diffact:{x:100,y:410,w:300,h:98,grp:'trained',t:'DiffAct head',s:'diffusion seg · 11 cls',big:1},
  types:{x:540,y:410,w:300,h:98,grp:'trained',t:'Type head',s:'correct / incorrect / remove'},
  viterbi:{x:100,y:566,w:300,h:60,grp:'decode',t:'Viterbi decode',s:'procedure-aware smoothing'},
  segclose:{x:540,y:558,w:300,h:76,grp:'decode',t:'Segment-close pooling',s:'TYPE logits · last ~1–2 s'},
  stepseg:{x:120,y:674,w:260,h:54,grp:'neutral',t:'Step segments',s:'t0 → t1 = part'},
  merge:{x:260,y:766,w:420,h:52,grp:'neutral',t:'Merge: segment × verdict',s:'one verdict per step'},
  timeline:{x:66,y:872,w:252,h:58,grp:'output',t:'Step timeline',s:'t0 → t1 → part'},
  outcome:{x:344,y:872,w:252,h:58,grp:'output',t:'Per-step outcome',s:'✓ / ✗ / − removed'},
  verdict:{x:622,y:872,w:252,h:58,grp:'output',t:'Procedure verdict',s:'PASS / deviations'},
};
const GBOX=[
  {x:82,y:124,w:776,h:116,c:'var(--c-frozen)',lab:'Dual frozen encoders · extracted once, shared'},
  {x:82,y:384,w:776,h:150,c:'var(--c-trained)',lab:'Trained heads · read the same 2176-d features'},
];
const EDGES=[['video','giant'],['video','iv2'],['giant','fusion'],['iv2','fusion'],
  ['fusion','diffact'],['fusion','types'],['diffact','viterbi'],['types','segclose'],
  ['viterbi','stepseg'],['stepseg','merge'],['segclose','merge'],
  ['merge','timeline'],['merge','outcome'],['merge','verdict']];
const eid=(a,b)=>'e_'+a+'_'+b;
function epath(a,b){
  const A=N[a],B=N[b]; const fx=A.x+A.w/2, fy=A.y+A.h, tx=B.x+B.w/2, ty=B.y, mid=(fy+ty)/2;
  return `M${fx} ${fy} C${fx} ${mid} ${tx} ${mid} ${tx} ${ty}`;
}
function buildArch(){
  const svg=$("arch"); svg.innerHTML="";
  GBOX.forEach(g=>{ svg.appendChild(el('rect',{x:g.x,y:g.y,width:g.w,height:g.h,rx:14,class:'grpbox',stroke:g.c}));
    const t=el('text',{x:g.x+14,y:g.y+18,class:'grplab',fill:g.c}); t.textContent=g.lab; svg.appendChild(t); });
  EDGES.forEach(([a,b])=> svg.appendChild(el('path',{id:eid(a,b),d:epath(a,b),class:'edge'})));
  for(const k in N){
    const n=N[k], g=el('g',{id:'n_'+k,class:'node grp-'+n.grp});
    g.appendChild(el('rect',{x:n.x-3,y:n.y-3,width:n.w+6,height:n.h+6,rx:15,fill:'var(--acc)',class:'glow',opacity:0}));
    g.appendChild(el('rect',{x:n.x,y:n.y,width:n.w,height:n.h,rx:12,class:'nbox'}));
    const t=el('text',{x:n.x+14,y:n.y+(n.big?28:25),class:'ntitle'}); t.textContent=n.t; g.appendChild(t);
    const s=el('text',{x:n.x+14,y:n.y+(n.big?44:42),class:'nsub'}); s.textContent=n.s; g.appendChild(s);
    const b=el('text',{id:'badge_'+k,x:n.x+n.w-11,y:n.y+17,'text-anchor':'end',class:'badge'}); g.appendChild(b);
    svg.appendChild(g);
  }
  // DiffAct internals: epoch/loss text, progress bar, equalizer
  const d=N.diffact;
  const ep=el('text',{id:'arch_ep',x:d.x+14,y:d.y+70,class:'nsub',fill:'var(--acc)','font-weight':700}); svg.appendChild(ep);
  const eqg=el('g',{class:'eq',id:'arch_eq'}); const bx=d.x+d.w-56, by=d.y+d.h-14;
  for(let i=0;i<6;i++){const r=el('rect',{x:bx+i*8,y:by-16,width:5,height:16,rx:1.5,fill:'var(--acc)',opacity:.8});
    r.style.transformBox='fill-box'; r.style.transformOrigin='bottom'; r.style.animationDelay=(i*0.13)+'s'; eqg.appendChild(r);}
  svg.appendChild(eqg);
  svg.appendChild(el('rect',{x:d.x+12,y:d.y+d.h-9,width:d.w-24,height:4,rx:2,fill:'var(--hairline)'}));
  svg.appendChild(el('rect',{id:'arch_prog',x:d.x+12,y:d.y+d.h-9,width:0,height:4,rx:2,fill:'var(--acc)'}));
  // Type head internals (mirror DiffAct): epoch/loss text, progress bar, equalizer
  const ty=N.types;
  svg.appendChild(el('text',{id:'type_ep',x:ty.x+14,y:ty.y+70,class:'nsub',fill:'var(--acc)','font-weight':700}));
  const teqg=el('g',{class:'eq',id:'type_eq'}); const tbx=ty.x+ty.w-56, tby=ty.y+ty.h-14;
  for(let i=0;i<6;i++){const r=el('rect',{x:tbx+i*8,y:tby-16,width:5,height:16,rx:1.5,fill:'var(--acc)',opacity:.8});
    r.style.transformBox='fill-box'; r.style.transformOrigin='bottom'; r.style.animationDelay=(i*0.13)+'s'; teqg.appendChild(r);}
  svg.appendChild(teqg);
  svg.appendChild(el('rect',{x:ty.x+12,y:ty.y+ty.h-9,width:ty.w-24,height:4,rx:2,fill:'var(--hairline)'}));
  svg.appendChild(el('rect',{id:'type_prog',x:ty.x+12,y:ty.y+ty.h-9,width:0,height:4,rx:2,fill:'var(--acc)'}));
  // legend
  const ly=968, items=[['frozen','var(--c-frozen)'],['trained','var(--c-trained)'],['decode','var(--c-decode)'],['output','var(--c-output)']];
  const lg=el('g',{class:'legend'}); let lx=100;
  items.forEach(([lab,c])=>{ lg.appendChild(el('rect',{x:lx,y:ly-10,width:14,height:14,rx:3,fill:'color-mix(in srgb,'+c+' 16%,var(--surface))',stroke:c,'stroke-width':1.5}));
    const t=el('text',{x:lx+20,y:ly+1}); t.textContent=lab; lg.appendChild(t); lx+=130; });
  svg.appendChild(lg);
  archBuilt=true;
}
const EDGECOLOR={video:'var(--muted)',giant:'var(--giant)',iv2:'var(--iv2)',fusion:'var(--acc)',
  diffact:'var(--c-decode)',types:'var(--c-decode)',viterbi:'var(--c-output)',segclose:'var(--c-output)',
  stepseg:'var(--c-output)',merge:'var(--c-output)'};
// idempotent particle stream on edge a->b; re-uses the existing <g> when unchanged so flow never resets
function flow(a,b,color,n,dur,mode){
  const svg=$("arch"), id=eid(a,b), sig=mode+'|'+color+'|'+n+'|'+dur;
  let g=svg.querySelector('.pflow[data-e="'+id+'"]');
  if(g&&g.dataset.sig===sig){ g.dataset.keep='1'; return; }
  if(g) g.remove();
  g=el('g',{class:'pflow '+mode,'data-e':id}); g.dataset.sig=sig; g.dataset.keep='1';
  const r=mode==='amb'?2.2:3.6;
  for(let i=0;i<n;i++){
    const c=el('circle',{r:r,class:'particle',fill:color,color:color});
    const am=el('animateMotion',{dur:dur+'s',repeatCount:'indefinite',begin:(-i*dur/n)+'s'});
    am.appendChild(el('mpath',{href:'#'+id})); c.appendChild(am); g.appendChild(c);
  }
  svg.appendChild(g);
}
function stateOf(d,key){const s=d.stages.find(x=>x.key===key);return s?s.state:'pending';}
function updateArch(d){
  if(!archBuilt) buildArch();
  const ex=stateOf(d,'extract'),fu=stateOf(d,'fuse'),tr=stateOf(d,'train'),ev=stateOf(d,'eval');
  const encDone=ex==='done', fuseDone=fu==='done', trDone=tr==='done', evDone=ev==='done';
  const set=(k,cls)=>{const g=$("n_"+k); if(g) g.setAttribute('class','node grp-'+N[k].grp+' '+cls);};
  const bdg=(k,txt,col)=>{const b=$("badge_"+k); if(b){b.textContent=txt; if(col)b.setAttribute('fill',col);}};
  set('video','done'); bdg('video','▶','var(--muted)');
  set('giant',encDone?'done':'active'); set('iv2',encDone?'done':'active');
  bdg('giant',encDone?'❄':'…','var(--c-frozen)'); bdg('iv2',encDone?'❄':'…','var(--c-frozen)');
  set('fusion',fuseDone?'done':(encDone?'active':'future')); bdg('fusion',fuseDone?'✓':'','var(--good)');
  set('diffact',tr==='active'?'active':(trDone?'done':'future')); bdg('diffact',tr==='active'?'':(trDone?'✓':''),'var(--good)');
  const th=d.typehead||{running:false,started:false,cur_ep:-1,total_ep:401,loss:[]};
  const tyState=th.running?'active':((th.started&&th.cur_ep>=th.total_ep-2)?'done':(th.started?'active':'future'));
  set('types',tyState); bdg('types',tyState==='active'?'':(tyState==='done'?'✓':'NEW'),tyState==='done'?'var(--good)':'var(--warn)');
  set('viterbi',evDone?'done':(trDone?'active':'future')); bdg('viterbi',evDone?'✓':'','var(--good)');
  set('segclose','future'); bdg('segclose','NEW','var(--warn)');
  set('stepseg',evDone?'done':'future'); bdg('stepseg',evDone?'✓':'','var(--good)');
  set('merge',evDone?'done':'future'); bdg('merge','','');
  ['timeline','outcome','verdict'].forEach(k=>{set(k,evDone?'done':'future'); bdg(k,evDone?'✓':'','var(--good)');});
  const setE=(a,b,cls)=>{const p=$(eid(a,b)); if(p) p.setAttribute('class','edge '+cls);};
  setE('video','giant',encDone?'done':'live'); setE('video','iv2',encDone?'done':'live');
  setE('giant','fusion',fuseDone?'done':'future'); setE('iv2','fusion',fuseDone?'done':'future');
  setE('fusion','diffact',tr==='active'?'live':(trDone?'done':'future'));
  setE('fusion','types',tyState==='active'?'live':(tyState==='done'?'done':'future')); setE('diffact','viterbi',evDone?'done':(trDone?'live':'future'));
  setE('types','segclose','future'); setE('viterbi','stepseg',evDone?'done':'future');
  setE('stepseg','merge',evDone?'done':'future'); setE('segclose','merge','future');
  ['timeline','outcome','verdict'].forEach(k=>setE('merge',k,evDone?'done':'future'));
  // data particles: always drift along completed edges, brighter/faster into the active head
  const asvg=$("arch");
  [...asvg.querySelectorAll('.pflow')].forEach(g=>g.dataset.keep='');
  if(!REDUCE){
    EDGES.forEach(([a,b])=>{ const p=$(eid(a,b)); if(!p) return; const c=p.getAttribute('class')||'';
      if(c.includes('live')) flow(a,b,'var(--acc)',3,1.9,'hot');
      else if(c.includes('done')) flow(a,b,EDGECOLOR[a]||'var(--acc)',2,3.6,'amb'); });
    if(tr==='active') flow('fusion','diffact','var(--acc)',4,1.4,'hot');
    if(tyState==='active') flow('fusion','types','var(--acc)',4,1.4,'hot');
  }
  [...asvg.querySelectorAll('.pflow')].forEach(g=>{ if(!g.dataset.keep) g.remove(); });
  const da=d.diffact, prog=N.diffact.w-24;
  $("arch_ep").textContent = da.started?`ep ${Math.max(0,da.cur_ep)}/${da.total_ep} · loss ${da.loss.length?da.loss[da.loss.length-1].loss.toFixed(3):'—'}`:'';
  $("arch_prog").setAttribute('width', prog*Math.max(0,da.cur_ep)/da.total_ep);
  $("arch_eq").style.display = tr==='active'?'':'none';
  $("type_ep").textContent = th.started?`ep ${Math.max(0,th.cur_ep)}/${th.total_ep} · loss ${th.loss.length?th.loss[th.loss.length-1].loss.toFixed(3):'—'}`:'';
  $("type_prog").setAttribute('width', (N.types.w-24)*Math.max(0,th.cur_ep)/th.total_ep);
  $("type_eq").style.display = tyState==='active'?'':'none';
}

/* ---------- status / loss / metrics / encoders / gpu ---------- */
function statusPill(d){
  const tr=stateOf(d,'train'),ev=stateOf(d,'eval'),th=d.typehead||{}; let txt,col,pulse=true;
  if(ev==='done' && !th.running){txt="COMPLETE";col="var(--good)";pulse=false;}
  else if(th.running){txt="TRAINING TYPE HEAD · ep "+Math.max(0,th.cur_ep)+"/"+th.total_ep;col="var(--acc)";}
  else if(tr==='active'){txt="TRAINING · ep "+Math.max(0,d.diffact.cur_ep)+"/"+d.diffact.total_ep;col="var(--acc)";}
  else if(tr==='done'){txt="TRAINED · awaiting eval";col="var(--good)";pulse=false;}
  else if(d.giant_running||d.iv2_running){txt="EXTRACTING";col="var(--good)";}
  else if(tr==='stopped'){txt="TRAIN STOPPED";col="var(--warn)";pulse=false;}
  else{txt="IDLE";col="var(--warn)";pulse=false;}
  $("status").style.color=col; $("status").innerHTML=`<span class="dot" style="background:${col};${pulse?'':'animation:none'}"></span>${txt}`;
}
function lossCurve(d){
  const da=(d.typehead && (d.typehead.running||d.typehead.loss.length)) ? d.typehead : d.diffact, loss=da.loss, svg=$("loss"); svg.innerHTML="";
  $("epinfo").textContent=da.started?`epoch ${Math.max(0,da.cur_ep)} / ${da.total_ep}`:"not started";
  $("epfill").style.width=(100*Math.max(0,da.cur_ep)/da.total_ep)+"%";
  const W=640,H=280,m={l:46,r:14,t:14,b:28},add=e=>{svg.appendChild(e);return e;};
  if(!loss.length){const t=add(el('text',{x:W/2,y:H/2,'text-anchor':'middle',fill:'var(--faint)','font-size':13}));t.textContent='waiting for first epoch…';return;}
  const xmax=da.total_ep,ys=loss.map(p=>p.loss),ymax=Math.max(...ys)*1.08||1,ymin=Math.min(...ys,0);
  const px=x=>m.l+x/xmax*(W-m.l-m.r),py=v=>H-m.b-(v-ymin)/(ymax-ymin||1)*(H-m.t-m.b);
  for(let g=0;g<=4;g++){const v=ymin+(ymax-ymin)*g/4;
    add(el('line',{x1:m.l,y1:py(v),x2:W-m.r,y2:py(v),stroke:'var(--grid)','stroke-width':1}));
    const t=add(el('text',{x:m.l-7,y:py(v)+4,'text-anchor':'end',fill:'var(--faint)','font-size':10.5,'font-family':'var(--mono)'}));t.textContent=v.toFixed(2);}
  [0,300,600,900,1200].filter(x=>x<=xmax).forEach(x=>{const t=add(el('text',{x:px(x),y:H-m.b+18,'text-anchor':'middle',fill:'var(--faint)','font-size':10.5,'font-family':'var(--mono)'}));t.textContent=x;});
  da.metrics.forEach(mm=>add(el('line',{x1:px(mm.ep),y1:m.t,x2:px(mm.ep),y2:H-m.b,stroke:'var(--good)','stroke-width':1,'stroke-dasharray':'2 3',opacity:.5})));
  const dl=loss.map((p,i)=>(i?'L':'M')+px(p.ep)+' '+py(p.loss)).join(' ');
  const grad=el('linearGradient',{id:'lossgrad',x1:'0',y1:'0',x2:'1',y2:'0'});
  grad.appendChild(el('stop',{offset:'0','stop-color':'var(--giant)'}));
  grad.appendChild(el('stop',{offset:'1','stop-color':'var(--acc)'}));
  add(el('defs',{})).appendChild(grad);
  add(el('path',{d:dl+` L${px(loss[loss.length-1].ep)} ${py(ymin)} L${px(loss[0].ep)} ${py(ymin)} Z`,fill:'var(--acc)','fill-opacity':.10}));
  const line=add(el('path',{d:dl,fill:'none',stroke:'url(#lossgrad)','stroke-width':2.4,'stroke-linejoin':'round','stroke-linecap':'round'}));
  if(!lossDrawn&&!REDUCE){ const L=line.getTotalLength(); line.style.strokeDasharray=L; line.style.strokeDashoffset=L;
    requestAnimationFrame(()=>{ line.style.transition='stroke-dashoffset 1.2s ease'; line.style.strokeDashoffset=0; }); }
  lossDrawn=true;
  const last=loss[loss.length-1], lxp=px(last.ep), lyp=py(last.loss);
  if(da.running&&!REDUCE){ const ring=add(el('circle',{cx:lxp,cy:lyp,r:4,fill:'none',stroke:'var(--acc)','stroke-width':2}));
    ring.appendChild(el('animate',{attributeName:'r',values:'4;11;4',dur:'1.9s',repeatCount:'indefinite'}));
    ring.appendChild(el('animate',{attributeName:'opacity',values:'.85;0;.85',dur:'1.9s',repeatCount:'indefinite'})); }
  add(el('circle',{cx:lxp,cy:lyp,r:3.6,fill:'var(--acc)',stroke:'var(--surface)','stroke-width':1.5}));
}
function metricsPanel(d){
  const da=d.diffact, latest=da.metrics.length?da.metrics[da.metrics.length-1]:null;
  $("metrics").innerHTML=ORDER.map(k=>{const tgt=d.target[k],val=latest?latest[k]:null,head=(k==="F1@50");
    const w=val!=null?Math.min(100,val):0,tp=Math.min(100,tgt);
    return `<div class="mrow${head?' head':''}"><div class="top"><span class="k">${k}${head?' ★':''}</span>
      <span class="v">${val!=null?val.toFixed(1):'—'} <span style="color:var(--faint)">/ ${tgt}</span></span></div>
      <div class="mtrack"><div class="mfill" style="width:${w}%"></div><div class="mtick" data-t="${tgt}" style="left:${tp}%"></div></div></div>`;}).join('');
  const bF=da.best&&da.best["F1@50"]!=null?da.best["F1@50"]:null;
  $("mhint").textContent=latest?`Latest eval @ epoch ${latest.ep} · best F1@50 ${bF!=null?bF.toFixed(1):'—'} / 70.1 · evals every 200 ep`
                               :`First eval at epoch 0, then every 200 — bars fill in as evals arrive.`;
}
function encoders(d){
  $("encsub").textContent=(d.giant_count>=84&&d.iv2_count>=84)?"extraction complete · weights frozen":"extracting";
  $("enc").innerHTML=`<span><span class="swatch" style="background:var(--giant)"></span>giant-SSv2 <b>${d.giant_count}/84</b> · 1408-d</span>`+
    `<span><span class="swatch" style="background:var(--iv2)"></span>IV2-B14 <b>${d.iv2_count}/84</b> · 768-d</span>`+
    `<span><span class="swatch" style="background:var(--acc)"></span>fused <b>${d.fused_count}/84</b> · 2176-d</span>`;
}
function mem(d){
  const g=d.gpu,sys=Math.max(0,g.total-g.giant_mem-g.iv2_mem-g.diffact_mem-g.free);
  $("gpuutil").textContent=g.util+'% util · '+gb(g.used)+' / '+gb(g.total)+' GB';
  const segs=[{lab:'giant',mib:g.giant_mem,c:'var(--giant)'},{lab:'IV2',mib:g.iv2_mem,c:'var(--iv2)'},{lab:'Type',mib:g.type_mem||0,c:'var(--c-trained)'},
    {lab:'DiffAct',mib:g.diffact_mem,c:'var(--acc)'},{lab:'system',mib:sys,c:'var(--faint)'},{lab:'free',mib:g.free,c:'var(--free)',free:1}].filter(s=>s.mib>0||s.free);
  $("membar").innerHTML=segs.map(s=>`<div class="memseg${s.free?' free':''}" style="flex:${Math.max(0.001,s.mib)};background:${s.c}">${s.mib/g.total>0.1?gb(s.mib)+'G':''}</div>`).join('');
  $("memkey").innerHTML=segs.map(s=>`<span><span class="swatch" style="background:${s.c}"></span>${s.lab} <span class="num">${gb(s.mib)} GB</span></span>`).join('');
}
async function poll(){
  try{
    const d=await (await fetch('/data',{cache:'no-store'})).json();
    statusPill(d);updateArch(d);lossCurve(d);metricsPanel(d);mem(d);
    $("err").style.display='none';
    $("foot").innerHTML=`<span>● live · updated <span class="num">${d.snap}</span> · every 4 s</span><span>reads <code>diffact_train.log</code> · <code>nvidia-smi</code></span>`;
  }catch(e){$("err").style.display='';}
}
poll(); setInterval(poll,4000);
</script></body></html>"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"live_monitor on http://localhost:{a.port}  ROOT={ROOT}", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
