#!/usr/bin/env python3
"""Generate a standalone, self-contained HTML strobe editor for a run (2026-07-09).

For a given run it computes the median background, cuts each fish snapshot into a
transparent sprite (segment_fish), and builds the trajectory, then embeds them all
as data URIs into one HTML file with a canvas editor. Open the HTML in a browser to
move / show-hide / delete / add / resize each fish, tune the trace (opacity, width,
colour, under/over, border), then export a PNG + a settings JSON.

Usage:  python3 make_strobe_editor.py [run ...]     (default: the 6 report runs)
Output: strobe_editor/<run>_editor.html
"""
import os, sys, base64, json
import cv2, numpy as np
import matplotlib
from build_strobe_plots import CONFIG, segment_fish, _HERE, RUNS_DIR, CACHE_DIR
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))   # repository root
from paths import ANALYSIS  # noqa: E402

OUT_W = 1280
OUT_DIR = str(ANALYSIS / "strobe_editor")
os.makedirs(OUT_DIR, exist_ok=True)

REPORT_RUNS = ["hope_didnt_work_2_highres", "insane_run_8_middle_bg_highres", "insane_run_2_highres",
               "finetune_run_7_highres", "finetune_run_8_highres", "figure_eight_6_clean_highres"]


def plasma_lut():
    cm = matplotlib.colormaps["plasma"]
    return [[int(255 * c) for c in cm(i / 255.0)[:3]] for i in range(256)]


def png_datauri(img_bgr_or_bgra):
    ok, buf = cv2.imencode(".png", img_bgr_or_bgra)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def median_bg(cap, total, w, h, bg_frame_idx=None):
    if bg_frame_idx is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_idx)
        ret, fr = cap.read()
        if ret:
            return cv2.resize(fr, (w, h)).astype(np.float32)
    idx = np.linspace(0, total - 1, min(total, 60)).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, fr = cap.read()
        if ret:
            frames.append(cv2.resize(fr, (w, h)))
    return np.median(np.stack(frames), 0).astype(np.float32)


def collect(run_name, cfg):
    base = cfg.get("base_run", run_name)
    c = np.load(os.path.join(CACHE_DIR, f"{base}.npz"), allow_pickle=True)
    fish_px, tgt_px, frame_time, total = c["fish_px"], c["tgt_px"], c["frame_time"], int(c["total"])
    folder = [f for f in os.listdir(RUNS_DIR) if base in f][0]
    vid = os.path.join(RUNS_DIR, folder, "raw", "video.avi")
    cap = cv2.VideoCapture(vid)
    Wv, Hv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = OUT_W
    h = int(round(OUT_W * Hv / Wv)); h -= h % 2
    sc = w / Wv
    r_fish = int(120 * sc * (Wv / OUT_W))  # keep body radius ~ physical; sc*Wv/OUT_W == 1
    r_fish = int(120)

    min_idx = total - 1
    if cfg.get("stop_at_min"):
        fw, tw = c["fish_world"], c["tgt_world"]
        v = ~np.isnan(fw[:, 0]) & ~np.isnan(tw[:, 0])
        if v.sum():
            tm = np.median(tw[v], 0)
            dd = np.hypot(fw[:, 0] - tm[0], fw[:, 1] - tm[1])
            vi = np.where(v)[0]
            min_idx = int(vi[np.argmin(dd[v])])

    bg = median_bg(cap, total, w, h, cfg.get("bg_frame_idx"))

    t0 = frame_time[0]
    tspan = max(frame_time[min_idx] - t0, 1e-6)
    trail = []
    for fi in range(min_idx + 1):
        fp = fish_px[fi]
        if not np.isnan(fp[0]):
            trail.append([round(float(fp[0] * sc), 1), round(float(fp[1] * sc), 1),
                          round(float((frame_time[fi] - t0) / tspan), 4)])

    sprites = []
    last = (-999, -999)
    md = cfg["min_dist_px"] * sc
    for fi in range(min_idx + 1):
        fp = fish_px[fi]
        if np.isnan(fp[0]):
            continue
        cx, cy = int(fp[0] * sc), int(fp[1] * sc)
        if last[0] == -999 or np.hypot(cx - last[0], cy - last[1]) > md:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, fr = cap.read()
            if not ret:
                continue
            frame = cv2.resize(fr, (w, h)).astype(np.float32)
            mask = segment_fish(frame, cx, cy, r_fish, 1.0)
            ys, xs = np.where(mask > 0.04)
            if len(xs) == 0:
                last = (cx, cy); continue
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            crop = frame[y0:y1, x0:x1].astype(np.uint8)
            alpha = np.clip(mask[y0:y1, x0:x1] * 255, 0, 255).astype(np.uint8)
            bgra = np.dstack([crop, alpha])
            sprites.append({"uri": png_datauri(bgra), "x": int(x0), "y": int(y0),
                            "w": int(x1 - x0), "hh": int(y1 - y0),
                            "t": round(float((frame_time[fi] - t0) / tspan), 3)})
            last = (cx, cy)
    cap.release()
    return bg.astype(np.uint8), sprites, trail, w, h


def build_html(run, bg_uri, sprites, trail, w, h, lut):
    data = {"run": run, "w": w, "h": h, "bg": bg_uri, "sprites": sprites, "trail": trail, "lut": lut}
    payload = json.dumps(data)
    return _HTML_TEMPLATE.replace("/*__DATA__*/", payload)


_HTML_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Strobe editor</title><style>
body{margin:0;font-family:system-ui,sans-serif;background:#1e1e24;color:#e8e8ea;display:flex}
#stage{flex:1;overflow:auto;padding:12px}
#panel{width:290px;background:#26262e;padding:14px;box-sizing:border-box;height:100vh;overflow:auto}
canvas{background:#000;max-width:100%;box-shadow:0 0 12px #000;cursor:move}
h2{font-size:15px;margin:4px 0 10px}h3{font-size:12px;text-transform:uppercase;color:#9a9aa5;margin:16px 0 6px}
label{display:block;font-size:12px;margin:7px 0 2px;color:#c8c8d0}
input[type=range]{width:100%}
.row{display:flex;gap:6px;align-items:center}
button{background:#3a3a46;color:#e8e8ea;border:1px solid #4a4a58;border-radius:5px;padding:6px 9px;font-size:12px;cursor:pointer}
button:hover{background:#464654}button.on{background:#2d6cdf;border-color:#2d6cdf}
#fishlist{max-height:230px;overflow:auto;border:1px solid #3a3a46;border-radius:5px;padding:4px}
.fitem{display:flex;align-items:center;gap:6px;font-size:12px;padding:3px 2px}
.fitem.sel{background:#333344;border-radius:4px}
.sw{font-size:11px;color:#9a9aa5}
</style></head><body>
<div id="stage"><canvas id="cv"></canvas></div>
<div id="panel">
<h2 id="title">Strobe editor</h2>
<h3>Trajectory</h3>
<label>Opacity <span class="sw" id="oL"></span></label><input type="range" id="op" min="0" max="100" value="90">
<label>Width <span class="sw" id="wL"></span></label><input type="range" id="tw" min="1" max="10" value="4">
<div class="row" style="margin-top:8px">
<button id="ord" class="on">trace under fish</button>
<button id="bd">border</button></div>
<div class="row" style="margin-top:8px">
<button id="cmode" class="on">plasma</button>
<input type="color" id="scol" value="#ffcc00"></div>
<h3>Fish snapshots <span class="sw" id="nfish"></span></h3>
<div class="row"><button id="addmode">＋ add (click canvas)</button><button id="dupe">duplicate</button></div>
<label>Selected scale <span class="sw" id="scL"></span></label><input type="range" id="fsc" min="40" max="180" value="100">
<label>Selected opacity <span class="sw" id="foL"></span></label><input type="range" id="fop" min="0" max="100" value="100">
<button id="del" style="margin-top:6px;background:#7a2d2d;border-color:#7a2d2d">delete selected</button>
<div id="fishlist" style="margin-top:8px"></div>
<h3>Export</h3>
<div class="row"><button id="png">⬇ PNG</button><button id="cfg">⬇ settings</button></div>
<label style="margin-top:8px">Load settings</label><input type="file" id="load" accept=".json">
</div>
<script>
const D = /*__DATA__*/;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
cv.width=D.w;cv.height=D.h;
const bg=new Image();bg.src=D.bg;
let sprites=D.sprites.map((s,i)=>({...s,visible:true,scale:1,opacity:1,id:i}));
Promise.all(sprites.map(s=>new Promise(r=>{const im=new Image();im.onload=()=>{s.img=im;r()};im.src=s.uri;}))).then(()=>{bg.complete?draw():bg.onload=draw;});
const st={op:0.9,tw:4,order:'under',border:false,cmode:'plasma',scol:'#ffcc00'};
let sel=null,addMode=false;

function plasma(t){const i=Math.max(0,Math.min(255,Math.round((1-t)*255)));const c=D.lut[i];return `rgb(${c[0]},${c[1]},${c[2]})`;}
function drawTrace(){const T=D.trail;if(T.length<2)return;ctx.save();ctx.globalAlpha=st.op;ctx.lineCap='round';ctx.lineJoin='round';
 if(st.border){ctx.strokeStyle='#141414';ctx.lineWidth=st.tw+3;ctx.beginPath();ctx.moveTo(T[0][0],T[0][1]);for(let i=1;i<T.length;i++)ctx.lineTo(T[i][0],T[i][1]);ctx.stroke();}
 ctx.lineWidth=st.tw;
 for(let i=1;i<T.length;i++){ctx.strokeStyle=st.cmode==='plasma'?plasma(T[i][2]):st.scol;ctx.beginPath();ctx.moveTo(T[i-1][0],T[i-1][1]);ctx.lineTo(T[i][0],T[i][1]);ctx.stroke();}
 ctx.restore();}
function drawFish(){for(const s of sprites){if(!s.visible||!s.img)continue;ctx.globalAlpha=s.opacity;const w=s.w*s.scale,h=s.hh*s.scale;ctx.drawImage(s.img,s.x-(w-s.w)/2,s.y-(h-s.hh)/2,w,h);}ctx.globalAlpha=1;}
function draw(){ctx.clearRect(0,0,cv.width,cv.height);ctx.drawImage(bg,0,0,cv.width,cv.height);
 if(st.order==='under'){drawTrace();drawFish();}else{drawFish();drawTrace();}
 if(sel!==null){const s=sprites.find(x=>x.id===sel);if(s){const w=s.w*s.scale,h=s.hh*s.scale;ctx.strokeStyle='#2d6cdf';ctx.lineWidth=2;ctx.strokeRect(s.x-(w-s.w)/2,s.y-(h-s.hh)/2,w,h);}}
 renderList();}
function renderList(){document.getElementById('nfish').textContent=sprites.filter(s=>s.visible).length+' shown';
 const L=document.getElementById('fishlist');L.innerHTML='';sprites.forEach((s,k)=>{const d=document.createElement('div');d.className='fitem'+(s.id===sel?' sel':'');
 d.innerHTML=`<input type=checkbox ${s.visible?'checked':''}><span style="flex:1">fish ${k+1}</span>`;
 d.querySelector('input').onchange=e=>{s.visible=e.target.checked;draw();};
 d.querySelector('span').onclick=()=>{sel=s.id;syncSel();draw();};L.appendChild(d);});}
function syncSel(){const s=sprites.find(x=>x.id===sel);if(!s)return;fsc.value=s.scale*100;fop.value=s.opacity*100;scL.textContent=Math.round(s.scale*100)+'%';foL.textContent=Math.round(s.opacity*100)+'%';}

// canvas coords
function pos(e){const r=cv.getBoundingClientRect();return{x:(e.clientX-r.left)*cv.width/r.width,y:(e.clientY-r.top)*cv.height/r.height};}
let drag=null;
cv.onmousedown=e=>{const p=pos(e);
 if(addMode){const base=sprites.find(x=>x.id===sel)||sprites[0];if(base){const ns={...base,id:Date.now(),x:p.x-base.w/2,y:p.y-base.hh/2,visible:true};sprites.push(ns);sel=ns.id;addMode=false;document.getElementById('addmode').classList.remove('on');draw();syncSel();}return;}
 for(let i=sprites.length-1;i>=0;i--){const s=sprites[i];if(!s.visible)continue;const w=s.w*s.scale,h=s.hh*s.scale,x=s.x-(w-s.w)/2,y=s.y-(h-s.hh)/2;if(p.x>=x&&p.x<=x+w&&p.y>=y&&p.y<=y+h){sel=s.id;drag={dx:p.x-s.x,dy:p.y-s.y};syncSel();draw();return;}}
 sel=null;draw();};
cv.onmousemove=e=>{if(!drag||sel===null)return;const p=pos(e),s=sprites.find(x=>x.id===sel);s.x=p.x-drag.dx;s.y=p.y-drag.dy;draw();};
window.onmouseup=()=>drag=null;

// controls
op.oninput=e=>{st.op=e.target.value/100;oL.textContent=e.target.value+'%';draw();};
tw.oninput=e=>{st.tw=+e.target.value;wL.textContent=e.target.value+'px';draw();};
ord.onclick=()=>{st.order=st.order==='under'?'over':'under';ord.textContent='trace '+st.order+' fish';ord.classList.toggle('on',st.order==='under');draw();};
bd.onclick=()=>{st.border=!st.border;bd.classList.toggle('on',st.border);draw();};
cmode.onclick=()=>{st.cmode=st.cmode==='plasma'?'solid':'plasma';cmode.textContent=st.cmode;cmode.classList.toggle('on',st.cmode==='plasma');draw();};
scol.oninput=e=>{st.scol=e.target.value;draw();};
fsc.oninput=e=>{const s=sprites.find(x=>x.id===sel);if(s){s.scale=e.target.value/100;scL.textContent=e.target.value+'%';draw();}};
fop.oninput=e=>{const s=sprites.find(x=>x.id===sel);if(s){s.opacity=e.target.value/100;foL.textContent=e.target.value+'%';draw();}};
del.onclick=()=>{if(sel!==null){sprites=sprites.filter(x=>x.id!==sel);sel=null;draw();}};
dupe.onclick=()=>{const s=sprites.find(x=>x.id===sel);if(s){const ns={...s,id:Date.now(),x:s.x+30,y:s.y+30};sprites.push(ns);sel=ns.id;draw();syncSel();}};
document.getElementById('addmode').onclick=e=>{addMode=!addMode;e.target.classList.toggle('on',addMode);};
document.getElementById('title').textContent=D.run;
png.onclick=()=>{const s=sel;sel=null;draw();const a=document.createElement('a');a.download=D.run+'_edited.png';a.href=cv.toDataURL('image/png');a.click();sel=s;draw();};
cfg.onclick=()=>{const out={trace:st,fish:sprites.map(s=>({id:s.id,x:Math.round(s.x),y:Math.round(s.y),scale:s.scale,opacity:s.opacity,visible:s.visible}))};
 const a=document.createElement('a');a.download=D.run+'_settings.json';a.href='data:application/json,'+encodeURIComponent(JSON.stringify(out,null,2));a.click();};
document.getElementById('load').onchange=e=>{const f=e.target.files[0];if(!f)return;const rd=new FileReader();rd.onload=()=>{const j=JSON.parse(rd.result);Object.assign(st,j.trace);
 j.fish.forEach(fj=>{const s=sprites.find(x=>x.id===fj.id);if(s)Object.assign(s,fj);});
 op.value=st.op*100;tw.value=st.tw;draw();};rd.readAsText(f);};
</script></body></html>"""


if __name__ == "__main__":
    runs = sys.argv[1:] or REPORT_RUNS
    lut = plasma_lut()
    for r in runs:
        cfg = CONFIG[r]
        print(f"collecting {r} ...")
        bg, sprites, trail, w, h = collect(r, cfg)
        html = build_html(r, png_datauri(bg), sprites, trail, w, h, lut)
        out = os.path.join(OUT_DIR, f"{r}_editor.html")
        with open(out, "w") as fh:
            fh.write(html)
        print(f"  -> {out}  ({len(sprites)} fish, {len(trail)} trail pts, {len(html)//1024} KB)")
