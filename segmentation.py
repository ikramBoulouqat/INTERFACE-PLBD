"""
segmentation.py — Segmentation U-Net des zones par degre de danger.

Encapsule le modele du notebook Kaggle (U-Net / encodeur ResNet34, 5 classes)
pour qu'app.py puisse l'utiliser sans dependre de torch au demarrage : si les
poids ou les librairies manquent, l'interface continue de tourner et les routes
de segmentation renvoient un message clair au lieu de planter.

Classes (identiques au notebook xView2) -> degre de danger :
    0  fond            -> transparent
    1  no-damage       -> vert      (danger faible)
    2  minor-damage    -> jaune     (danger modere)
    3  major-damage    -> orange    (danger eleve)
    4  destroyed       -> rouge     (danger critique)

IMPORTANT — le pretraitement doit etre IDENTIQUE a l'entrainement :
le notebook fait seulement `img/255.0` puis transpose en CHW, SANS
normalisation mean/std imagenet. On reproduit exactement ca ici.
"""

import os
import cv2
import numpy as np

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
WEIGHTS_PATH = os.environ.get(
    "UNET_WEIGHTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet_danger.pth"),
)
INPUT_SIZE = 256          # le modele a ete entraine en 256x256
DEVICE_PREF = "cpu"       # "cuda" si GPU dispo
ALPHA = 0.45              # opacite de l'overlay couleur

# index de classe -> (libelle, couleur BGR)  (BGR car OpenCV)
CLASSES = [
    ("Fond",          (0, 0, 0)),        # 0 - non dessine (transparent)
    ("Sans dommage",  (80, 175, 76)),    # 1 - vert
    ("Dommage leger", (60, 200, 240)),   # 2 - jaune
    ("Dommage majeur",(40, 130, 240)),   # 3 - orange
    ("Detruit",       (50, 50, 210)),    # 4 - rouge
]
N_CLASSES = len(CLASSES)

# ----------------------------------------------------------------------
# Chargement paresseux du modele (torch n'est importe qu'ici)
# ----------------------------------------------------------------------
_model = None
_device = None
_load_error = None


def _try_load_model():
    """Tente de charger le U-Net une seule fois. Renvoie (model, device) ou (None, None)."""
    global _model, _device, _load_error
    if _model is not None or _load_error is not None:
        return _model, _device

    try:
        import torch
        import segmentation_models_pytorch as smp
    except Exception as e:  # librairie absente
        _load_error = ("Librairies manquantes (torch / segmentation-models-pytorch). "
                       "Installez-les : pip install torch segmentation-models-pytorch")
        print("[segmentation]", _load_error, "|", e)
        return None, None

    if not os.path.exists(WEIGHTS_PATH):
        _load_error = (f"Poids introuvables : {WEIGHTS_PATH}. "
                       "Exportez unet_danger.pth depuis Kaggle (voir la cellule de sauvegarde).")
        print("[segmentation]", _load_error)
        return None, None

    try:
        device = torch.device("cuda" if (DEVICE_PREF == "cuda" and torch.cuda.is_available()) else "cpu")
        # encoder_weights=None : on charge NOS poids, pas ceux d'imagenet
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=3, classes=N_CLASSES)
        state = torch.load(WEIGHTS_PATH, map_location=device)
        # accepte un state_dict brut ou un checkpoint {'model': state_dict}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        model.load_state_dict(state)
        model.to(device).eval()
        _model, _device = model, device
        print(f"[segmentation] Modele charge sur {device} depuis {WEIGHTS_PATH}")
    except Exception as e:
        _load_error = f"Echec du chargement du modele : {e}"
        print("[segmentation]", _load_error)
        return None, None

    return _model, _device


def is_ready():
    """True si le modele est chargeable/charge."""
    m, _ = _try_load_model()
    return m is not None


def load_error():
    """Message d'erreur de chargement, ou None."""
    _try_load_model()
    return _load_error


# ----------------------------------------------------------------------
# Inference + overlay  (la partie pure numpy/cv2 est testable sans torch)
# ----------------------------------------------------------------------
def _predict_mask(bgr):
    """Image BGR (HxWx3) -> masque de classes (INPUT_SIZE x INPUT_SIZE), int."""
    model, device = _try_load_model()
    if model is None:                              # torch/poids absents : on degrade proprement
        return None
    import torch

    img = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE))
    img = img / 255.0                              # MEME pretraitement que l'entrainement
    img = np.transpose(img, (2, 0, 1))             # HWC -> CHW
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)                       # (1, N_CLASSES, H, W)
        mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    return mask.astype(np.int32)


def colorize_and_blend(bgr, mask):
    """
    Construit l'overlay couleur a partir du masque et l'incruste sur l'image.
    PURE numpy/cv2 (testable sans torch).
    Renvoie (image_overlay_BGR, stats) ou stats = liste de dicts par classe.
    """
    h, w = bgr.shape[:2]
    mask_full = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(1, N_CLASSES):                # on saute le fond (0)
        color[mask_full == cid] = CLASSES[cid][1]

    drawn = mask_full > 0                          # pixels a melanger
    out = bgr.copy()
    if drawn.any():
        blended = cv2.addWeighted(bgr, 1 - ALPHA, color, ALPHA, 0)
        out[drawn] = blended[drawn]

    # stats : pourcentage de pixels par classe de danger (hors fond)
    total = float(h * w)
    stats = []
    for cid in range(1, N_CLASSES):
        pct = round(100.0 * float((mask_full == cid).sum()) / total, 1)
        name, (b, g, r) = CLASSES[cid]
        stats.append({"id": cid, "name": name, "pct": pct,
                      "color": f"rgb({r},{g},{b})"})
    return out, stats


def segment_frame(bgr):
    """
    Point d'entree principal. Image BGR -> (overlay BGR, stats).
    Si le modele n'est pas pret, renvoie (None, None).
    """
    mask = _predict_mask(bgr)
    if mask is None:
        return None, None
    return colorize_and_blend(bgr, mask)


def placeholder_frame(text="Segmentation indisponible", size=(480, 640)):
    """Image grise avec un message — affichee si le modele n'est pas charge."""
    h, w = size
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(img, line, (24, 40 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return img
