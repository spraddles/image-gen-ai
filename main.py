import os, cv2, numpy as np
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
from rembg import remove

# --------- Paths ---------
IMG_DIR = Path("images")
BG_PATH = IMG_DIR / "background.png"
PERSONA_PATH = IMG_DIR / "persona.png"
SELFIE_PATH = IMG_DIR / "selfie.png"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "final.jpg"

assert BG_PATH.exists() and PERSONA_PATH.exists() and SELFIE_PATH.exists(), \
    "Missing images in images/background.png, images/persona.png, images/selfie.png"

# --------- Models / Tunables ---------
DET_SIZE = (1024, 1024)     # detector input; robust for small faces
LONG_MAX  = 2048            # resize long side for speed/quality balance
MASK_EXPAND_PX = 6
FEATHER_PX = 12
USE_POISSON = False         # try True if you want seamlessClone color bleeding
GRAIN_STRENGTH = 0.01       # set 0 to disable

# Init InsightFace (CPU)
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=DET_SIZE)

# Load local inswapper (downloaded already)
swapper = insightface.model_zoo.get_model(str(Path("models/inswapper_128.onnx")), download=False)

# --------- Helpers ---------
def read_bgr(p: Path):
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"Cannot read {p}")
    return img

def read_any(p: Path):
    """Reads BGR or BGRA; returns (bgr, alpha_or_None)."""
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read {p}")
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        alpha = img[:, :, 3]
        return bgr, alpha
    return img, None

def resize_long(img, m=LONG_MAX):
    h, w = img.shape[:2]
    s = m / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w*s), int(h*s)), cv2.INTER_AREA)

def composite_on_gray(bgr, alpha, gray=128):
    """Composite a transparent PNG onto flat gray to help face detection."""
    if alpha is None:
        return bgr
    bg = np.full_like(bgr, gray, dtype=np.uint8)
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    comp = (bgr.astype(np.float32) * a + bg.astype(np.float32) * (1 - a)).astype(np.uint8)
    return comp

def pad_and_upscale(img, pad_ratio=0.12, min_long=1024):
    """Pad edges and ensure enough pixels for reliable detection."""
    h, w = img.shape[:2]
    pad = int(max(h, w) * pad_ratio)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side < min_long:
        scale = min_long / long_side
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    return img

def largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

def yaw_pitch(face):
    kps3 = getattr(face, "landmark_3d_68", None)
    if kps3 is None:
        kps2 = face.kps
        yaw = float(kps2[1,0] - kps2[0,0])
        pitch = float(kps2[2,1] - (kps2[0,1] + kps2[1,1]) / 2)
        return yaw, pitch
    pts = kps3[:, :2]
    yaw = float(np.mean(pts[36:42,0]) - np.mean(pts[42:48,0]))
    pitch = float(np.mean(pts[27:31,1]) - np.mean(pts[30:35,1]))
    return yaw, pitch

def ellipse_mask_from_landmarks(img, face, expand_px=8):
    kps = face.kps
    cx, cy = kps.mean(axis=0)
    eye_dist = np.linalg.norm(kps[0]-kps[1])
    mouth_dist = np.linalg.norm(kps[3]-kps[4])
    a = max(eye_dist*1.25, mouth_dist*1.4) + expand_px
    b = a*1.25 + expand_px
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    m = (((X - cx)**2) / (a*a) + ((Y - cy)**2) / (b*b)) <= 1.0
    return m.astype(np.float32)

def feather(mask, px):
    if px <= 0: return mask.astype(np.float32)
    k = max(1, (px | 1))
    return (cv2.GaussianBlur((mask*255).astype(np.uint8), (k, k), 0).astype(np.float32)/255.0).clip(0, 1)

def lab_hist_match(src_bgr, ref_bgr, mask=None):
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is None: mask = np.ones(src_bgr.shape[:2], np.uint8)
    out = src_lab.copy()
    for c in range(3):
        s = src_lab[:,:,c][mask>0.5]; r = ref_lab[:,:,c][mask>0.5]
        if s.size < 50 or r.size < 50: continue
        s_mean, s_std = s.mean(), s.std() + 1e-5
        r_mean, r_std = r.mean(), r.std() + 1e-5
        out[:,:,c] = ((src_lab[:,:,c]-s_mean) * (r_std/s_std) + r_mean)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

def poisson(src_part, dst, bin_mask, center_xy):
    return cv2.seamlessClone(src_part, dst, bin_mask, center_xy, cv2.MIXED_CLONE)

def remove_bg_rgba(bgr):
    png = remove(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if isinstance(png, bytes):
        arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        arr = png
    if arr.shape[2] == 3:
        alpha = np.full(arr.shape[:2], 255, np.uint8)
        arr = np.dstack([arr, alpha])
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

def erode_feather_alpha(alpha, erode_px=2, feather_px=3):
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
        alpha = cv2.erode(alpha, k, iterations=1)
    if feather_px > 0:
        alpha = cv2.GaussianBlur(alpha, (feather_px*2+1, feather_px*2+1), 0)
    return alpha

def alpha_over_premul(fg_bgra, bg_bgr, x, y):
    fh, fw = fg_bgra.shape[:2]
    out = bg_bgr.copy()
    patch = out[y:y+fh, x:x+fw]
    a = (fg_bgra[:,:,3:4].astype(np.float32)/255.0)
    fg = fg_bgra[:,:,:3].astype(np.float32)
    comp = fg + patch.astype(np.float32) * (1.0 - a)
    out[y:y+fh, x:x+fw] = np.clip(comp, 0, 255).astype(np.uint8)
    return out

def add_micrograin(bgr, strength=0.015):
    if strength <= 0: return bgr
    h, w = bgr.shape[:2]
    noise = (np.random.randn(h, w, 1) * 255 * strength).astype(np.int16)
    out = np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out

# (Optional) quick debug to visualize detections
def debug_save_faces(im, faces, path: Path):
    vis = im.copy()
    for f in faces or []:
        x1, y1, x2, y2 = map(int, f.bbox)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        for (px,py) in f.kps:
            cv2.circle(vis, (int(px),int(py)), 2, (0,0,255), -1)
    cv2.imwrite(str(path), vis)

# --------- Load images ---------
bg = resize_long(read_bgr(BG_PATH), LONG_MAX)
persona = resize_long(read_bgr(PERSONA_PATH), LONG_MAX)

# Selfie: handle transparent PNGs and tight crops
selfie_bgr_raw, selfie_alpha = read_any(SELFIE_PATH)
selfie = composite_on_gray(selfie_bgr_raw, selfie_alpha, gray=128)
selfie = pad_and_upscale(selfie, pad_ratio=0.12, min_long=1024)
selfie = resize_long(selfie, LONG_MAX)

# --------- Detect faces ---------
src_faces = face_app.get(selfie)
tgt_faces = face_app.get(persona)
# Debug (optional):
# debug_save_faces(selfie, src_faces, OUT_DIR/"_dbg_selfie.jpg")
# debug_save_faces(persona, tgt_faces, OUT_DIR/"_dbg_persona.jpg")

if not src_faces: raise RuntimeError("No face in selfie.")
if not tgt_faces: raise RuntimeError("No face in persona.")

src = largest_face(src_faces)
tgt = largest_face(tgt_faces)

# --------- Pose sanity (optional info) ---------
# sy, sp = yaw_pitch(src); ty, tp = yaw_pitch(tgt)
# print("pose yaw/pitch:", (sy,sp), (ty,tp))

# --------- Build face mask on target ---------
mask_soft = ellipse_mask_from_landmarks(persona, tgt, expand_px=MASK_EXPAND_PX)
mask_bin  = (mask_soft > 0.5).astype(np.uint8) * 255
x1, y1, x2, y2 = map(int, tgt.bbox)
center = ((x1 + x2)//2, (y1 + y2)//2)

# --------- Swap ---------
swapped = swapper.get(persona.copy(), tgt, src, paste_back=True)

# --------- Local color match inside the face mask ---------
swapped_col = lab_hist_match(swapped, persona, mask=(mask_soft > 0.2).astype(np.uint8))

# --------- Blend the swapped face onto persona ---------
if USE_POISSON:
    src_part = persona.copy()
    idx = mask_soft > 0.01
    src_part[idx] = swapped_col[idx]
    blended_persona = poisson(src_part, persona, mask_bin, center)
else:
    feathered = feather(mask_soft, FEATHER_PX).astype(np.float32)
    blended_persona = (swapped_col * feathered[...,None] + persona * (1 - feathered[...,None])).astype(np.uint8)

# --------- Cut persona cleanly (remove halo) ---------
persona_rgba = remove_bg_rgba(blended_persona)
alpha = erode_feather_alpha(persona_rgba[:,:,3], erode_px=2, feather_px=3)
persona_rgba[:,:,3] = alpha

# --------- Composite onto background (centered) ---------
bh, bw = bg.shape[:2]; ph, pw = persona_rgba.shape[:2]
scale = min(0.92 * bh / ph, 0.92 * bw / pw)
new_size = (int(pw * scale), int(ph * scale))
persona_rgba = cv2.resize(persona_rgba, new_size, cv2.INTER_AREA)
ph, pw = persona_rgba.shape[:2]
x = (bw - pw) // 2; y = (bh - ph) // 2
composite = alpha_over_premul(persona_rgba, bg, x, y)

# Optional micro-grain to hide faint seams
final = add_micrograin(composite, strength=GRAIN_STRENGTH)

cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(OUT_PATH))
print(f"Saved: {OUT_PATH}")
