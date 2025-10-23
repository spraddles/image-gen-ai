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
DET_SIZE = (1024, 1024)
LONG_MAX  = 2048
MASK_EXPAND_PX = 6
FEATHER_PX = 12
USE_POISSON = False
GRAIN_STRENGTH = 0.01

HAIR_TINT_STRENGTH = 0.35   # 0..1
STUBBLE_SOFTEN     = 0.35   # 0..1
STUBBLE_LIFT       = 6      # +L in Lab
TEETH_FILL         = 2      # inpaint radius
TEETH_WHITEN       = 6      # +L in Lab

# Init InsightFace (CPU)
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
swapper = insightface.model_zoo.get_model(str(Path("models/inswapper_128.onnx")), download=False)

# --------- Helpers ---------
def read_bgr(p: Path):
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"Cannot read {p}")
    return img

def read_any(p: Path):
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None: raise RuntimeError(f"Cannot read {p}")
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR); alpha = img[:, :, 3]
        return bgr, alpha
    return img, None

def resize_long(img, m=LONG_MAX):
    h, w = img.shape[:2]; s = m / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w*s), int(h*s)), cv2.INTER_AREA)

def composite_on_gray(bgr, alpha, gray=128):
    if alpha is None: return bgr
    bg = np.full_like(bgr, gray, np.uint8)
    a = (alpha.astype(np.float32)/255.0)[..., None]
    return (bgr.astype(np.float32)*a + bg.astype(np.float32)*(1-a)).astype(np.uint8)

def pad_and_upscale(img, pad_ratio=0.12, min_long=1024):
    h, w = img.shape[:2]; pad = int(max(h, w)*pad_ratio)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    h, w = img.shape[:2]; long_side = max(h, w)
    if long_side < min_long:
        s = min_long/long_side; img = cv2.resize(img, (int(w*s), int(h*s)), cv2.INTER_CUBIC)
    return img

def largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

def ellipse_mask_from_landmarks(img, face, expand_px=8):
    kps = face.kps
    cx, cy = kps.mean(axis=0)
    eye_dist = np.linalg.norm(kps[0]-kps[1])
    mouth_dist = np.linalg.norm(kps[3]-kps[4])
    a = max(eye_dist*1.25, mouth_dist*1.4) + expand_px
    b = a*1.25 + expand_px
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    return ((((X-cx)**2)/(a*a) + ((Y-cy)**2)/(b*b)) <= 1.0).astype(np.float32)

def feather(mask, px):
    if px <= 0: return mask.astype(np.float32)
    k = max(1, (px | 1))
    return (cv2.GaussianBlur((mask*255).astype(np.uint8), (k, k), 0).astype(np.float32)/255.0).clip(0,1)

def lab_hist_match(src_bgr, ref_bgr, mask=None):
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is None: mask = np.ones(src_bgr.shape[:2], np.uint8)
    out = src_lab.copy()
    for c in range(3):
        s = src_lab[:,:,c][mask>0.5]; r = ref_lab[:,:,c][mask>0.5]
        if s.size < 50 or r.size < 50: continue
        s_mean, s_std = s.mean(), s.std()+1e-5
        r_mean, r_std = r.mean(), r.std()+1e-5
        out[:,:,c] = ((src_lab[:,:,c]-s_mean)*(r_std/s_std) + r_mean)
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
        alpha = cv2.erode(alpha, k, 1)
    if feather_px > 0:
        alpha = cv2.GaussianBlur(alpha, (feather_px*2+1, feather_px*2+1), 0)
    return alpha

def alpha_over_premul(fg_bgra, bg_bgr, x, y):
    fh, fw = fg_bgra.shape[:2]; out = bg_bgr.copy()
    patch = out[y:y+fh, x:x+fw]
    a = (fg_bgra[:,:,3:4].astype(np.float32)/255.0)
    fg = fg_bgra[:,:,:3].astype(np.float32)
    comp = fg + patch.astype(np.float32)*(1.0-a)
    out[y:y+fh, x:x+fw] = np.clip(comp, 0, 255).astype(np.uint8)
    return out

def add_micrograin(bgr, strength=0.015):
    if strength <= 0: return bgr
    h,w = bgr.shape[:2]
    noise = (np.random.randn(h,w,1)*255*strength).astype(np.int16)
    return np.clip(bgr.astype(np.int16)+noise,0,255).astype(np.uint8)

# ---- Region masks from 68 landmarks (for post-corrections) ----
def get_lm68(face):
    pts = getattr(face, "landmark_3d_68", None)
    return None if pts is None else pts[:, :2].astype(np.float32)

def poly_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    return m

def build_lower_face_mask(shape, lm):
    jaw = lm[0:17]; mouth = lm[48:60]
    pts = np.vstack([jaw, mouth[::-1]])
    return (cv2.GaussianBlur(poly_mask(shape, pts),(21,21),0)/255.0).astype(np.float32)

def build_mouth_inner_mask(shape, lm):
    return (cv2.GaussianBlur(poly_mask(shape, lm[60:68]),(9,9),0)/255.0).astype(np.float32)

def build_hair_band_mask(shape, lm, strength=1.0):
    brow_y = int(np.mean(lm[17:27,1]))
    x_min = int(np.min(lm[:,0])); x_max = int(np.max(lm[:,0]))
    y_top = max(0, int(brow_y - (lm[8,1]-brow_y)*0.35))
    m = np.zeros(shape[:2], np.uint8)
    cv2.rectangle(m, (x_min, 0), (x_max, y_top), int(255*strength), -1)
    return (cv2.GaussianBlur(m,(31,31),0)/255.0).astype(np.float32)

def tint_toward(src_bgr, ref_bgr, src_mask, strength=0.35, ref_mask=None):
    """
    Shift src chroma (a/b in Lab) toward mean chroma of ref within ref_mask.
    src_mask and ref_mask are in their own image spaces (no shape coupling).
    """
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Gather region means independently (no boolean-shape mismatch)
    if ref_mask is None:
        ref_sel = np.ones(ref_bgr.shape[:2], np.uint8)
    else:
        ref_sel = (ref_mask > 0.3).astype(np.uint8)

    src_sel = (src_mask > 0.3).astype(np.uint8)

    for c in (1, 2):  # a, b channels
        s_vals = src_lab[:,:,c][src_sel == 1]
        r_vals = ref_lab[:,:,c][ref_sel == 1]
        if s_vals.size < 50 or r_vals.size < 50: 
            continue
        delta = (float(np.mean(r_vals)) - float(np.mean(s_vals))) * strength
        chan = src_lab[:,:,c]
        chan[src_sel == 1] = np.clip(chan[src_sel == 1] + delta, 0, 255)
        src_lab[:,:,c] = chan

    return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def lighten_in_lab(bgr, mask, delta_L):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:,:,0] = np.clip(lab[:,:,0] + delta_L*mask, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

# --------- Load images ---------
bg = resize_long(read_bgr(BG_PATH), LONG_MAX)
persona = resize_long(read_bgr(PERSONA_PATH), LONG_MAX)

# Selfie: handle transparent PNGs & tight crops
selfie_raw, selfie_a = read_any(SELFIE_PATH)
selfie = composite_on_gray(selfie_raw, selfie_a, gray=128)
selfie = pad_and_upscale(selfie, pad_ratio=0.12, min_long=1024)
selfie = resize_long(selfie, LONG_MAX)

# --------- Detect faces ---------
src_faces = face_app.get(selfie)
tgt_faces = face_app.get(persona)
if not src_faces: raise RuntimeError("No face in selfie.")
if not tgt_faces: raise RuntimeError("No face in persona.")
src = largest_face(src_faces); tgt = largest_face(tgt_faces)

# --------- Build face mask on target ---------
mask_soft = ellipse_mask_from_landmarks(persona, tgt, expand_px=MASK_EXPAND_PX)
mask_bin  = (mask_soft > 0.5).astype(np.uint8) * 255
x1,y1,x2,y2 = map(int, tgt.bbox); center = ((x1+x2)//2, (y1+y2)//2)

# --------- Swap ---------
swapped = swapper.get(persona.copy(), tgt, src, paste_back=True)

# --------- Local color match inside the face mask ---------
swapped_col = lab_hist_match(swapped, persona, mask=(mask_soft > 0.2).astype(np.uint8))

# --------- Blend the swapped face onto persona ---------
if USE_POISSON:
    src_part = persona.copy(); idx = mask_soft > 0.01
    src_part[idx] = swapped_col[idx]
    blended_persona = poisson(src_part, persona, mask_bin, center)
else:
    feathered = feather(mask_soft, FEATHER_PX).astype(np.float32)
    blended_persona = (swapped_col*feathered[...,None] + persona*(1-feathered[...,None])).astype(np.uint8)

# --------- Post-corrections (hair, stubble, teeth) ---------
lm = get_lm68(tgt)
if lm is not None:
    # 1) Hair tint toward selfie hair color (use separate masks per image)
    hair_mask_persona = build_hair_band_mask(blended_persona.shape, lm, strength=1.0)
    src_faces2 = face_app.get(selfie)
    if src_faces2:
        lm_src = get_lm68(largest_face(src_faces2))
        if lm_src is not None:
            hair_mask_selfie = build_hair_band_mask(selfie.shape, lm_src, strength=1.0)
        else:
            hair_mask_selfie = None
        blended_persona = tint_toward(
            blended_persona, selfie,
            src_mask=hair_mask_persona,
            strength=HAIR_TINT_STRENGTH,
            ref_mask=hair_mask_selfie
        )

    # 2) Stubble cleanup (lower face)
    lower_mask = build_lower_face_mask(blended_persona.shape, lm)
    if STUBBLE_SOFTEN > 0:
        soft = cv2.bilateralFilter(blended_persona, d=7, sigmaColor=50, sigmaSpace=7)
        blended_persona = (soft*(lower_mask[...,None]*STUBBLE_SOFTEN) +
                           blended_persona*(1 - lower_mask[...,None]*STUBBLE_SOFTEN)).astype(np.uint8)
    if STUBBLE_LIFT != 0:
        blended_persona = lighten_in_lab(blended_persona, lower_mask, STUBBLE_LIFT)

    # 3) Teeth cleanup
    mouth_inner = build_mouth_inner_mask(blended_persona.shape, lm)
    if TEETH_FILL > 0:
        gray = cv2.cvtColor(blended_persona, cv2.COLOR_BGR2GRAY)
        dark = ((gray < 70) & (mouth_inner > 0.4)).astype(np.uint8)*255
        if np.count_nonzero(dark) > 0:
            blended_persona = cv2.inpaint(blended_persona, dark, TEETH_FILL, cv2.INPAINT_TELEA)
    if TEETH_WHITEN != 0:
        blended_persona = lighten_in_lab(blended_persona, mouth_inner, TEETH_WHITEN)

# --------- Cut persona cleanly (remove halo) ---------
persona_rgba = remove_bg_rgba(blended_persona)
alpha = erode_feather_alpha(persona_rgba[:,:,3], erode_px=2, feather_px=3)
persona_rgba[:,:,3] = alpha

# --------- Composite onto background ---------
bh,bw = bg.shape[:2]; ph,pw = persona_rgba.shape[:2]
scale = min(0.92*bh/ph, 0.92*bw/pw)
persona_rgba = cv2.resize(persona_rgba, (int(pw*scale), int(ph*scale)), cv2.INTER_AREA)
ph,pw = persona_rgba.shape[:2]
x = (bw-pw)//2; y = (bh-ph)//2
composite = alpha_over_premul(persona_rgba, bg, x, y)

# Optional micro-grain
final = add_micrograin(composite, strength=GRAIN_STRENGTH)

cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(OUT_PATH))
print(f"Saved: {OUT_PATH}")
