"""
Aerial Ops Dashboard - backend
Capture le flux ESP32-CAM UNE fois, lance YOLO, et sert :
  /           -> le dashboard (index.html)
  /<fichier>  -> fichiers statiques (support.js, logo.png, hero.jpg ...)
  /raw        -> video brute (MJPEG)                 <- onglet Flux en direct
  /detect     -> video annotee YOLO (MJPEG)          <- onglet Detection
  /stats      -> chiffres live + journal (JSON)      -> panneau interface
  /set_model  -> change le modele YOLO a chaud
  /set_conf   -> change le seuil de confiance a chaud

Lancer :  python app.py   puis ouvrir  http://localhost:5000
"""

import os
import time
import base64
import threading

import cv2
import numpy as np
from flask import Flask, Response, send_from_directory, jsonify, request
from ultralytics import YOLO

# Segmentation U-Net des zones de danger (chargement paresseux : si les poids
# ou torch manquent, le reste de l'interface tourne quand meme).
import segmentation as seg

# ----------------------- CONFIG (a editer) -----------------------
STREAM_URL = "http://192.168.43.111:81/stream"
MY_MODEL   = "yolov8n.pt"

CONF   = 0.30     # seuil de confiance de depart (le curseur le modifie ensuite)
IMGSZ  = 640      # baisser a 416 si le CPU rame
DEVICE = "cpu"    # "cuda" si tu as un GPU
MAX_LOG = 30      # nb max d'entrees dans le journal de detection

HOST, PORT = "0.0.0.0", 5000
DASHBOARD_FILE = "index2.html"

# Nom dans le menu deroulant  ->  fichier de poids + classes a garder.
#   Ton modele VisDrone : 0 pedestrian, 1 people  -> [0, 1]
#   Modeles COCO officiels : 0 person             -> [0]
#   (Les poids officiels se telechargent tout seuls au 1er usage : internet requis.)
MODELS = {
    "Mon modele (best.pt)": {"weights": MY_MODEL,     "classes": [0]},
    "YOLOv8n":              {"weights": "yolov8n.pt", "classes": [0]},
    "YOLOv8m":              {"weights": "yolov8m.pt", "classes": [0]},
    "YOLOv8x":              {"weights": "yolov8x.pt", "classes": [0]},
    "YOLOv11m":             {"weights": "yolo11m.pt", "classes": [0]},
}
DEFAULT_MODEL = "Mon modele (best.pt)"
# -----------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# --- modele courant (protege par model_lock car remplacable a chaud) ---
model_lock = threading.Lock()
current_name = DEFAULT_MODEL
current_classes = MODELS[DEFAULT_MODEL]["classes"]
model = YOLO(MODELS[DEFAULT_MODEL]["weights"])

# --- buffers de frames + stats (protege par buf_lock) ---
buf_lock = threading.Lock()
latest_raw = None
latest_annotated = None
stats = {"count": 0, "fps": 0.0, "avg_conf": 0.0, "latency_ms": 0.0,
         "model": current_name, "conf": CONF, "dets": []}

# --- segmentation (overlay + stats live, protege par seg_lock) ---
seg_lock = threading.Lock()
latest_seg_stats = {"ready": False, "classes": [], "error": None}


def cls_name(names, cid):
    try:
        return names[cid]
    except Exception:
        return str(cid)


def capture_loop():
    """Seul consommateur du flux ESP32. Se reconnecte si ca coupe."""
    global latest_raw
    while True:
        cap = cv2.VideoCapture(STREAM_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("Flux introuvable, nouvelle tentative dans 2s...")
            time.sleep(2)
            continue
        print("Camera connectee:", STREAM_URL)
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lecture echouee, reconnexion...")
                break
            with buf_lock:
                latest_raw = frame
        cap.release()
        time.sleep(1)


def detect_loop():
    """Lance YOLO sur la frame la plus recente; remplit la video annotee + les stats."""
    global latest_annotated, stats
    last = time.time()
    while True:
        with buf_lock:
            frame = None if latest_raw is None else latest_raw.copy()
        if frame is None:
            time.sleep(0.05)
            continue

        with model_lock:
            m, cls, name = model, current_classes, current_name

        t0 = time.time()
        results = m(frame, verbose=False, device=DEVICE,
                    imgsz=IMGSZ, conf=CONF, classes=cls)
        infer_ms = (time.time() - t0) * 1000.0
        annotated = results[0].plot()   # UNIQUEMENT les cadres YOLO, aucun texte ajoute

        boxes = results[0].boxes
        names = results[0].names
        count = 0 if boxes is None else len(boxes)
        confs = [] if boxes is None or boxes.conf is None else boxes.conf.tolist()
        avg = (sum(confs) / len(confs)) if confs else 0.0

        # Journal : une entree par detection, triee par confiance decroissante
        dets = []
        if boxes is not None and count:
            order = sorted(range(count), key=lambda i: float(boxes.conf[i]), reverse=True)
            for rank, i in enumerate(order[:MAX_LOG], start=1):
                cid = int(boxes.cls[i])
                dets.append({
                    "id": f"P-{rank:02d}",
                    "cls": cls_name(names, cid),
                    "conf": round(float(boxes.conf[i]), 2),
                })

        now = time.time()
        fps = 1.0 / (now - last) if now > last else 0.0
        last = now

        with buf_lock:
            latest_annotated = annotated
            stats = {"count": count, "fps": round(fps, 1),
                     "avg_conf": round(avg, 2), "latency_ms": round(infer_ms, 1),
                     "model": name, "conf": CONF, "dets": dets}


def mjpeg(picker):
    """Diffuse en MJPEG la frame renvoyee par picker()."""
    while True:
        with buf_lock:
            frame = picker()
        if frame is None:
            time.sleep(0.03)
            continue
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpg.tobytes() + b"\r\n")
        time.sleep(0.03)


@app.route("/raw")
def raw():
    return Response(mjpeg(lambda: latest_raw),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detect")
def detect():
    return Response(mjpeg(lambda: latest_annotated),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ------------------------ SEGMENTATION U-NET ------------------------
SEG_PERIOD = 0.7   # intervalle min entre deux inferences U-Net (s) -> limite le CPU


def segment_mjpeg():
    """Segmente le flux en direct a la demande (uniquement quand l'onglet est ouvert).
    Met a jour latest_seg_stats pour la legende dynamique."""
    global latest_seg_stats
    last_overlay = None
    last_run = 0.0
    while True:
        with buf_lock:
            frame = None if latest_raw is None else latest_raw.copy()

        if frame is None:
            # pas de flux : on montre un message au lieu de planter
            ph = seg.placeholder_frame(
                "Flux camera indisponible.\nConnectez l'ESP32-CAM\nou utilisez le mode Image.")
            ok, jpg = cv2.imencode(".jpg", ph)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpg.tobytes() + b"\r\n")
            time.sleep(0.5)
            continue

        now = time.time()
        if now - last_run >= SEG_PERIOD or last_overlay is None:
            overlay, sstats = seg.segment_frame(frame)
            last_run = now
            if overlay is None:
                msg = seg.load_error() or "Modele non charge"
                overlay = seg.placeholder_frame("Segmentation indisponible :\n" + msg)
                with seg_lock:
                    latest_seg_stats = {"ready": False, "classes": [], "error": msg}
            else:
                with seg_lock:
                    latest_seg_stats = {"ready": True, "classes": sstats, "error": None}
            last_overlay = overlay

        ok, jpg = cv2.imencode(".jpg", last_overlay)
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg.tobytes() + b"\r\n")
        time.sleep(0.05)


@app.route("/segment")
def segment():
    """Flux MJPEG segmente (zones de danger) du direct."""
    return Response(segment_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/segment_stats")
def segment_stats_route():
    """Pourcentages par classe de danger du dernier overlay live (JSON)."""
    with seg_lock:
        return jsonify(latest_seg_stats)


@app.route("/segment_image", methods=["POST"])
def segment_image():
    """Segmente une image uploadee. Renvoie l'overlay (PNG base64) + les classes.
    Fonctionne meme sans ESP32-CAM -> ideal pour une demo de soutenance."""
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "aucun fichier 'image'"}), 400

    data = np.frombuffer(request.files["image"].read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"ok": False, "error": "image illisible"}), 400

    overlay, sstats = seg.segment_frame(bgr)
    if overlay is None:
        return jsonify({"ok": False, "error": seg.load_error() or "modele non charge"}), 503

    ok, png = cv2.imencode(".png", overlay)
    if not ok:
        return jsonify({"ok": False, "error": "encodage PNG echoue"}), 500

    b64 = base64.b64encode(png.tobytes()).decode("ascii")
    return jsonify({"ok": True,
                    "image": "data:image/png;base64," + b64,
                    "classes": sstats})


@app.route("/segment_ready")
def segment_ready():
    """Etat du modele de segmentation (pour l'UI)."""
    return jsonify({"ready": seg.is_ready(), "error": seg.load_error()})


@app.route("/stats")
def stats_route():
    with buf_lock:
        return jsonify(stats)


@app.route("/set_model", methods=["POST", "GET"])
def set_model():
    """Change le modele YOLO sans redemarrer le serveur."""
    global model, current_classes, current_name
    name = request.args.get("name", "")
    if name not in MODELS:
        return jsonify({"ok": False, "error": "modele inconnu",
                        "available": list(MODELS)}), 400
    spec = MODELS[name]
    try:
        new_model = YOLO(spec["weights"])   # peut telecharger si officiel + 1er usage
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    with model_lock:
        model = new_model
        current_classes = spec["classes"]
        current_name = name
    print("Modele change ->", name)
    return jsonify({"ok": True, "model": name})


@app.route("/set_conf", methods=["POST", "GET"])
def set_conf():
    """Change le seuil de confiance utilise par la detection."""
    global CONF
    try:
        v = float(request.args.get("v", ""))
    except ValueError:
        return jsonify({"ok": False, "error": "valeur invalide"}), 400
    CONF = max(0.0, min(1.0, v))
    return jsonify({"ok": True, "conf": CONF})


@app.route("/")
def index():
    return send_from_directory(HERE, DASHBOARD_FILE)


@app.route("/<path:fname>")
def static_files(fname):
    return send_from_directory(HERE, fname)


if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=detect_loop, daemon=True).start()
    print(f"\n  Dashboard ->  http://localhost:{PORT}\n")
    app.run(host=HOST, port=PORT, threaded=True)
