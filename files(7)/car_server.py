#!/usr/bin/env python3
"""
Car server (single file) - camera relay + data recorder + Arduino serial bridge
===============================================================================
Roles:
  LAPTOP  (3 USB cameras + Arduino)  -> CAMERA page   http://localhost:5000/
        Camera access needs a secure context: use localhost, NOT the IP.
        Captures + uploads frames only. Does not drive.
  MOBILE  -> DRIVE page   http://<laptop-LAN-ip>:5000/?mode=drive
        Shows the relayed live feed + joystick. Drives the car, arms recording.

Install:  pip install flask pyserial
"""

import os
import time
import csv
import queue
import threading
from datetime import datetime
from flask import Flask, request, jsonify, Response

# ---- config -------------------------------------------------------------
SERIAL_PORT = "COM11"      # Windows COMx | Linux /dev/ttyUSB0 or /dev/ttyACM0 | Mac /dev/cu.usbserial-XXXX
BAUD        = 115200
SEND_HZ     = 20
STREAM_FPS  = 30
OUTPUT_DIR  = "dataset"
HOST        = "0.0.0.0"
PORT        = 5000
# -------------------------------------------------------------------------

app = Flask(__name__)

state_lock  = threading.Lock()

latest_control = {"steering": 0.0, "throttle": 0.0, "t": 0.0}
latest_frames  = {"left": None, "center": None, "right": None}
last_frame_ts  = {"left": 0.0, "center": 0.0, "right": 0.0}
recording   = False
sample_idx  = 0

ser = None


def open_serial():
    global ser
    try:
        import serial
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
        time.sleep(2.0)
        print(f"Serial open on {SERIAL_PORT}")
    except Exception as e:
        ser = None
        print(f"WARNING: serial not open ({e}). Control commands will be ignored.")


def serial_sender():
    period = 1.0 / SEND_HZ
    while True:
        with state_lock:
            c = dict(latest_control)
        if time.time() - c["t"] > 0.3:
            steer, thr = 0, 0
        else:
            steer = int(max(-1.0, min(1.0, c["steering"])) * 100)
            thr   = int(max(-1.0, min(1.0, c["throttle"])) * 100)
        if ser is not None:
            try:
                ser.write(f"{steer} {thr}\n".encode())
            except Exception:
                pass
        time.sleep(period)


writer_q = queue.Queue()


def recorder_thread():
    """All disk I/O lives here, fed by writer_q, so the HTTP handlers never
    block on file writes -- capture and control stay smooth under load."""
    csvf = None
    wr = None
    sdir = None
    while True:
        item = writer_q.get()
        try:
            kind = item[0]
            if kind == "start":
                sdir = item[1]
                for c in ("left", "center", "right"):
                    os.makedirs(os.path.join(sdir, c), exist_ok=True)
                csvf = open(os.path.join(sdir, "labels.csv"), "w", newline="")
                wr = csv.writer(csvf)
                wr.writerow(["index", "timestamp", "steering", "throttle",
                             "left", "center", "right"])
                print(f"Recording -> {sdir}")
            elif kind == "stop":
                if csvf:
                    csvf.flush()
                    csvf.close()
                csvf = None
                wr = None
            elif kind == "frame" and wr is not None:
                _, idx, imgs, steer, thr, ts = item
                paths = {}
                for c in ("left", "center", "right"):
                    rel = f"{c}/{idx:06d}.jpg"
                    with open(os.path.join(sdir, rel), "wb") as out:
                        out.write(imgs[c])
                    paths[c] = rel
                wr.writerow([idx, f"{ts:.3f}", f"{steer:.4f}", f"{thr:.4f}",
                             paths["left"], paths["center"], paths["right"]])
        except Exception as e:
            print("recorder error:", e)
        finally:
            writer_q.task_done()


def mjpeg_generator(cam):
    period = 1.0 / STREAM_FPS
    while True:
        with state_lock:
            data = latest_frames.get(cam)
        if data is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
        time.sleep(period)


@app.route("/")
def index():
    return Response(PAGE_HTML, mimetype="text/html")


@app.route("/control", methods=["POST"])
def control():
    d = request.get_json(force=True, silent=True) or {}
    with state_lock:
        latest_control["steering"] = float(d.get("steering", 0.0))
        latest_control["throttle"] = float(d.get("throttle", 0.0))
        latest_control["t"] = time.time()
    return ("", 204)


@app.route("/frame", methods=["POST"])
def frame():
    global sample_idx
    imgs = {}
    for cam in ("left", "center", "right"):
        f = request.files.get(cam)
        if f is not None:
            imgs[cam] = f.read()
    if not imgs:
        return ("no frames", 400)

    now = time.time()
    idx = None
    with state_lock:
        for k, v in imgs.items():
            latest_frames[k] = v
            last_frame_ts[k] = now
        rec   = recording
        steer = latest_control["steering"]
        thr   = latest_control["throttle"]
        # record only when all three cameras are present, so samples stay aligned
        if rec and len(imgs) == 3:
            idx = sample_idx
            sample_idx += 1

    if idx is not None:
        writer_q.put(("frame", idx, imgs, steer, thr, now))

    with state_lock:
        n = sample_idx
    return jsonify({"recording": rec, "n": n, "got": len(imgs)})


@app.route("/stream/<cam>")
def stream(cam):
    if cam not in ("left", "center", "right"):
        return ("bad cam", 404)
    return Response(mjpeg_generator(cam),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/latest/<cam>.jpg")
def latest(cam):
    with state_lock:
        data = latest_frames.get(cam)
    if data is None:
        return ("", 404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/record", methods=["POST"])
def record():
    global recording, sample_idx
    d = request.get_json(force=True, silent=True) or {}
    on = bool(d.get("on", False))
    with state_lock:
        was = recording
    if on and not was:
        sdir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        with state_lock:
            sample_idx = 0
            recording = True
        writer_q.put(("start", sdir))
    elif not on and was:
        with state_lock:
            recording = False
        writer_q.put(("stop",))
    with state_lock:
        n = sample_idx
    return jsonify({"recording": on, "n": n})


@app.route("/state")
def get_state():
    now = time.time()
    with state_lock:
        cams = {k: round((now - t) * 1000) if t else -1 for k, t in last_frame_ts.items()}
        return jsonify({"recording": recording, "n": sample_idx, "cams": cams})


PAGE_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Car control + camera stream</title>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no" />
<style>
  :root{ --bg:#0c1626; --panel:#13243d; --line:#24405f; --txt:#e8f0fb; --mut:#8fb0d6; --accent:#38a8ff; --gold:#f2b233; --ok:#5dcaa5; --bad:#e24b4a; }
  *{ box-sizing:border-box; }
  body{ font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:0; background:var(--bg); color:var(--txt); }
  header{ display:flex; align-items:center; gap:12px; padding:10px 14px; background:var(--panel); border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header h1{ font-size:1rem; margin:0; font-weight:600; }
  .grow{ flex:1; }
  button{ background:var(--line); color:var(--txt); border:1px solid var(--line); border-radius:8px; padding:8px 12px; cursor:pointer; font-size:.9rem; }
  button:active{ transform:translateY(1px); }
  button.rec{ background:#7a1f1f; border-color:#a32d2d; }
  button.rec.on{ background:#a32d2d; }
  .status{ font-size:.85rem; color:var(--mut); }
  .warn{ background:#5a1a1a; color:#ffd9d9; padding:10px 14px; font-size:.9rem; display:none; }
  .cams{ display:flex; gap:10px; padding:12px; flex-wrap:wrap; justify-content:center; }
  .cam{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:8px; width:300px; }
  .cam .lbl{ font-size:.8rem; color:var(--mut); margin-bottom:6px; display:flex; justify-content:space-between; }
  .cam .st{ font-size:.75rem; }
  .st.ok{ color:var(--ok); } .st.bad{ color:var(--bad); }
  video,img.feed{ width:100%; aspect-ratio:4/3; background:#000; border-radius:6px; object-fit:cover; display:block; }
  select{ width:100%; margin-top:6px; background:var(--bg); color:var(--txt); border:1px solid var(--line); border-radius:6px; padding:5px; font-size:.8rem; }
  .drivebar{ display:flex; gap:24px; align-items:center; justify-content:center; padding:6px 12px 22px; flex-wrap:wrap; }
  .pad{ position:relative; width:230px; height:230px; border-radius:50%; background:var(--panel); border:1px solid var(--line); touch-action:none; }
  .pad .cross-h,.pad .cross-v{ position:absolute; background:var(--line); }
  .pad .cross-h{ left:8%; right:8%; top:50%; height:1px; }
  .pad .cross-v{ top:8%; bottom:8%; left:50%; width:1px; }
  .knob{ position:absolute; width:74px; height:74px; border-radius:50%; background:var(--accent); left:50%; top:50%; transform:translate(-50%,-50%); box-shadow:0 0 0 4px rgba(56,168,255,.18); }
  .readout{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.9rem; color:var(--mut); min-width:170px; }
  .readout b{ color:var(--gold); }
  .hint{ font-size:.78rem; color:var(--mut); max-width:240px; line-height:1.4; }
</style>
</head>
<body>
<header>
  <h1>Car control</h1>
  <span id="mode" class="status"></span>
  <span class="grow"></span>
  <button id="refreshBtn">Refresh cams</button>
  <button id="recBtn" class="rec">Start recording</button>
  <span id="status" class="status"></span>
</header>

<div id="warn" class="warn"></div>
<div class="cams" id="cams"></div>

<div class="drivebar" id="drivebar" style="display:none">
  <div class="pad" id="pad">
    <div class="cross-h"></div><div class="cross-v"></div>
    <div class="knob" id="knob"></div>
  </div>
  <div>
    <div class="readout">steer <b id="rSteer">0.00</b><br>throttle <b id="rThr">0.00</b><br><span id="gpName" class="status"></span></div>
    <div class="hint">Drag the pad (or gamepad / arrow keys). Left-right = steering (what the model learns). Up-down = throttle. Release to center and stop.</div>
  </div>
</div>

<script>
const MODE = new URLSearchParams(location.search).get('mode') === 'drive' ? 'drive' : 'operator';
const CAP_W = 320, CAP_H = 240;
const FRAME_HZ = 30;
const CTRL_HZ = 20;
const CAMS = ['left','center','right'];
const sleep = ms => new Promise(r=>setTimeout(r,ms));
document.getElementById('mode').textContent = MODE === 'operator' ? 'camera device (laptop)' : 'remote drive';

let recording = false;
const recBtn = document.getElementById('recBtn');
recBtn.addEventListener('click', async ()=>{
  recording = !recording;
  try{ const r=await fetch('/record',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({on:recording})});
       const d=await r.json(); recording=d.recording; }catch(e){}
  recBtn.classList.toggle('on', recording);
  recBtn.textContent = recording ? 'Stop recording' : 'Start recording';
});
setInterval(async ()=>{
  try{ const r=await fetch('/state'); const d=await r.json();
    const cams=d.cams||{}; const live=CAMS.map(c=>cams[c]).filter(a=>a>=0 && a<1500).length;
    document.getElementById('status').textContent=(d.recording?'REC':'idle')+'  samples: '+d.n+'  cams live: '+live+'/3';
  }catch(e){}
}, 700);

// ================= OPERATOR (laptop, cameras only) =================
if(MODE === 'operator'){
  const warn = document.getElementById('warn');
  if(!navigator.mediaDevices || !window.isSecureContext){
    warn.style.display='block';
    warn.textContent = 'Cameras need a secure context. Open this page on the laptop at http://localhost:'+location.port+'/  (not the IP). Phones connect by IP with ?mode=drive.';
  }

  const cams = CAMS.map((name,i)=>({ name, index:i, video:null, select:null, st:null, stream:null }));
  const wrap = document.getElementById('cams');
  cams.forEach(cam=>{
    const d = document.createElement('div'); d.className='cam';
    d.innerHTML = '<div class="lbl"><span>'+cam.name+'</span><span class="st"></span></div>'
      + '<video autoplay playsinline muted></video><select></select>';
    wrap.appendChild(d);
    cam.video = d.querySelector('video');
    cam.select = d.querySelector('select');
    cam.st = d.querySelector('.st');
    cam.select.addEventListener('change', ()=>{ if(cam.select.value) startCam(cam, cam.select.value); });
  });

  function setSt(cam, txt, ok){ cam.st.textContent=txt; cam.st.className='st'+(ok===true?' ok':ok===false?' bad':''); }

  async function startCam(cam, deviceId){
    if(cam.stream){ cam.stream.getTracks().forEach(t=>t.stop()); cam.stream=null; }
    setSt(cam, 'opening...', null);
    try{
      cam.stream = await navigator.mediaDevices.getUserMedia({
        video:{ deviceId:{exact:deviceId}, width:{ideal:CAP_W}, height:{ideal:CAP_H}, frameRate:{ideal:30} },
        audio:false
      });
      cam.video.srcObject = cam.stream;
      setSt(cam, 'live', true);
    }catch(err){
      setSt(cam, 'FAILED: '+err.name, false);   // NotReadableError = USB busy/bandwidth
    }
  }

  // open cameras ONE AT A TIME -- opening 3 USB cams at once makes the later ones fail
  async function startAllSequential(){
    for(const cam of cams){
      if(cam.select.value){ await startCam(cam, cam.select.value); await sleep(350); }
    }
  }

  async function getDevices(){
    if(!navigator.mediaDevices) return;
    try{
      const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
      cams.forEach(cam=>{
        const cur = cam.select.value; cam.select.innerHTML='';
        devs.forEach((d,i)=>{ const o=document.createElement('option'); o.value=d.deviceId; o.textContent=d.label||('Camera '+(i+1)); cam.select.appendChild(o); });
        if(devs.length){ cam.select.value = cur || (devs[cam.index]?devs[cam.index].deviceId:devs[0].deviceId); }
      });
      if(devs.length < 3){ warn.style.display='block'; warn.textContent='Only '+devs.length+' camera(s) detected. Plug in all three (and check they are not on the same USB hub).'; }
      await startAllSequential();
    }catch(err){ document.getElementById('status').textContent='camera error: '+err.message; }
  }
  document.getElementById('refreshBtn').addEventListener('click', getDevices);

  const cv = document.createElement('canvas'); cv.width=CAP_W; cv.height=CAP_H;
  const ctx = cv.getContext('2d');
  function grab(cam){
    return new Promise(res=>{
      if(!cam.video.videoWidth) return res(null);
      ctx.drawImage(cam.video,0,0,CAP_W,CAP_H);
      cv.toBlob(b=>res(b), 'image/jpeg', 0.7);
    });
  }
  let inFlight=false;
  setInterval(async ()=>{
    if(inFlight) return; inFlight=true;
    try{
      const fd = new FormData(); let have=0;
      for(const cam of cams){ const b=await grab(cam); if(b){ fd.append(cam.name,b,cam.name+'.jpg'); have++; } }
      if(have) await fetch('/frame',{method:'POST',body:fd});
    }catch(e){}
    inFlight=false;
  }, 1000/FRAME_HZ);

  (async ()=>{
    try{
      const probe = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
      probe.getTracks().forEach(t=>t.stop());   // free the default camera before opening the real ones
    }catch(e){}
    getDevices();
  })();
}

// ================= DRIVE (mobile: view + joystick) =================
else {
  document.getElementById('refreshBtn').style.display='none';
  document.getElementById('drivebar').style.display='flex';
  const wrap = document.getElementById('cams');
  CAMS.forEach(name=>{
    const d=document.createElement('div'); d.className='cam';
    d.innerHTML='<div class="lbl"><span>'+name+'</span></div><img class="feed" alt="'+name+'">';
    wrap.appendChild(d);
    const img=d.querySelector('img');
    img.src='/stream/'+name;
    img.onerror=()=>{ setTimeout(()=>{ img.src='/stream/'+name+'?'+Date.now(); }, 1500); };
  });

  let steer=0, throttle=0;
  const pad=document.getElementById('pad'), knob=document.getElementById('knob');
  let dragging=false;
  function setFromPoint(x,y){
    const r=pad.getBoundingClientRect(), half=r.width/2;
    let nx=(x-(r.left+half))/half, ny=(y-(r.top+half))/half;
    const m=Math.hypot(nx,ny); if(m>1){nx/=m;ny/=m;}
    steer=nx; throttle=-ny;
    knob.style.left=(50+nx*42)+'%'; knob.style.top=(50+ny*42)+'%';
  }
  function reset(){ steer=0; throttle=0; knob.style.left='50%'; knob.style.top='50%'; }
  pad.addEventListener('pointerdown',e=>{ dragging=true; pad.setPointerCapture(e.pointerId); setFromPoint(e.clientX,e.clientY); });
  pad.addEventListener('pointermove',e=>{ if(dragging) setFromPoint(e.clientX,e.clientY); });
  pad.addEventListener('pointerup',()=>{ dragging=false; reset(); });
  pad.addEventListener('pointercancel',()=>{ dragging=false; reset(); });

  const keys={};
  addEventListener('keydown',e=>{ keys[e.key.toLowerCase()]=true; });
  addEventListener('keyup',e=>{ keys[e.key.toLowerCase()]=false; });
  function applyKeys(){ let s=0,t=0;
    if(keys['a']||keys['arrowleft'])s-=1; if(keys['d']||keys['arrowright'])s+=1;
    if(keys['w']||keys['arrowup'])t+=1; if(keys['s']||keys['arrowdown'])t-=1;
    if(s||t){ steer=s; throttle=t; knob.style.left=(50+s*42)+'%'; knob.style.top=(50-t*42)+'%'; return true; } return false; }
  function applyGamepad(){ const gps=navigator.getGamepads?navigator.getGamepads():[];
    for(const gp of gps){ if(!gp) continue;
      document.getElementById('gpName').textContent='gamepad: '+gp.id.slice(0,20);
      const dz=0.06; let sx=gp.axes[0]||0, sy=gp.axes[1]||0;
      if(Math.abs(sx)<dz)sx=0; if(Math.abs(sy)<dz)sy=0;
      if(sx||sy){ steer=sx; throttle=-sy; knob.style.left=(50+sx*42)+'%'; knob.style.top=(50+sy*42)+'%'; } return true; }
    return false; }

  let ctrlInFlight=false;
  setInterval(async ()=>{
    if(!dragging){ if(!applyGamepad()) applyKeys(); }
    document.getElementById('rSteer').textContent=steer.toFixed(2);
    document.getElementById('rThr').textContent=throttle.toFixed(2);
    if(ctrlInFlight) return; ctrlInFlight=true;
    try{ await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({steering:steer,throttle:throttle})}); }catch(e){}
    ctrlInFlight=false;
  }, 1000/CTRL_HZ);
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    open_serial()
    threading.Thread(target=serial_sender, daemon=True).start()
    threading.Thread(target=recorder_thread, daemon=True).start()
    print(f"CAMERA page (on the laptop):  http://localhost:{PORT}/")
    print(f"DRIVE page  (on the phone) :  http://<this-laptop-ip>:{PORT}/?mode=drive")
    app.run(host=HOST, port=PORT, threaded=True)
