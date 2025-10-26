import os, cv2, numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import insightface
from insightface.app import FaceAnalysis
from rembg import remove

# --------- Paths ---------
IMG_DIR = Path("images")
BG_PATH = IMG_DIR / "background.png"
PERSONA_PATH = IMG_DIR / "persona.png"
SELFIE_PATH = IMG_DIR / "selfie-1.jpg"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "final.jpg"

assert BG_PATH.exists() and PERSONA_PATH.exists() and SELFIE_PATH.exists(), \
    "Missing images in images/background.png, images/persona.png, images/selfie.png"

# --------- Models / Tunables ---------
DET_SIZE = (640, 640)       # Smaller size for better detection
LONG_MAX  = 2048            
MASK_EXPAND_PX = 6
FEATHER_PX = 12
USE_POISSON = False         
GRAIN_STRENGTH = 0.01       

# --------- Helpers: IO & geometry ---------
def read_any_exif(p: Path):
    """
    EXIF-aware reader. Returns (BGR, alpha_or_None).
    Works for PNG/JPEG/WebP; preserves alpha if present.
    """
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode == "RGBA":
            arr = np.array(im)  # RGBA
            bgr = arr[:, :, :3][:, :, ::-1].copy()
            alpha = arr[:, :, 3].copy()
            return bgr, alpha
        elif im.mode == "RGB":
            arr = np.array(im)  # RGB
            bgr = arr[:, :, ::-1].copy()
            return bgr, None
        elif im.mode == "L":
            arr = np.array(im)
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return bgr, None
        else:
            im = im.convert("RGBA")
            arr = np.array(im)
            bgr = arr[:, :, :3][:, :, ::-1].copy()
            alpha = arr[:, :, 3].copy()
            return bgr, alpha

def read_bgr(p: Path):
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"Cannot read {p}")
    return img

def composite_on_gray(bgr, alpha, gray=128):
    """Composite a transparent PNG onto flat gray to help face detection."""
    if alpha is None:
        return bgr
    bg = np.full_like(bgr, gray, dtype=np.uint8)
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    comp = (bgr.astype(np.float32) * a + bg.astype(np.float32) * (1 - a)).astype(np.uint8)
    return comp

def resize_long(img, m=LONG_MAX):
    h, w = img.shape[:2]
    s = m / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w*s), int(h*s)), cv2.INTER_AREA)

def pad_and_upscale(img, pad_ratio=0.18, min_long=1280):
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
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

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
        s_mean, s_std = s.mean(), max(s.std(), 1e-5)
        r_mean, r_std = r.mean(), max(r.std(), 1e-5)
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

def debug_save_faces(im, faces, path: Path):
    vis = im.copy()
    for f in faces or []:
        x1, y1, x2, y2 = map(int, f.bbox)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        for (px,py) in f.kps:
            cv2.circle(vis, (int(px),int(py)), 2, (0,0,255), -1)
    cv2.imwrite(str(path), vis)

def save_debug_image(img, path: Path, title="debug"):
    """Save debug image with info"""
    cv2.imwrite(str(path), img)
    print(f"Saved debug image: {path}")

# --------- Enhanced Face Detection ---------
def setup_face_detector(det_size=(640, 640), threshold=0.1):
    """Setup face detector with low threshold"""
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=-1, det_size=det_size)
    
    # Set very low detection threshold
    det = face_app.models.get('detection', None)
    if det is not None and hasattr(det, 'threshold'):
        try:
            det.threshold = float(threshold)
            print(f"Set detection threshold to: {threshold}")
        except Exception as e:
            print(f"Could not set threshold: {e}")
    
    return face_app

def preprocess_image_for_detection(img):
    """Apply multiple preprocessing techniques"""
    results = []
    
    # Original
    results.append(("original", img))
    
    # Brightness/contrast adjustments
    alpha = 1.2  # Contrast control
    beta = 30    # Brightness control
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    results.append(("brightness_contrast", adjusted))
    
    # CLAHE for contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    results.append(("clahe", clahe_img))
    
    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    results.append(("sharpened", sharpened))
    
    # Gaussian blur (sometimes helps with noisy images)
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    results.append(("blurred", blurred))
    
    return results

def detect_faces_comprehensive(face_app, img, debug_name="image"):
    """Try multiple detection strategies"""
    all_faces = []
    best_faces = []
    
    print(f"\n=== Detecting faces in {debug_name} ===")
    print(f"Image shape: {img.shape}")
    
    # Save original for debugging
    save_debug_image(img, OUT_DIR / f"{debug_name}_original.jpg", "Original")
    
    # Try different preprocessing techniques
    preprocessed = preprocess_image_for_detection(img)
    
    for name, processed_img in preprocessed:
        try:
            # Save preprocessed image for debugging
            save_debug_image(processed_img, OUT_DIR / f"{debug_name}_{name}.jpg", name)
            
            # Detect faces
            faces = face_app.get(processed_img)
            print(f"{name}: {len(faces)} faces detected")
            
            if faces:
                all_faces.extend(faces)
                if len(faces) > len(best_faces):
                    best_faces = faces
                    
                # Save debug image with faces
                debug_img = processed_img.copy()
                for face in faces:
                    x1, y1, x2, y2 = map(int, face.bbox)
                    cv2.rectangle(debug_img, (x1,y1), (x2,y2), (0,255,0), 3)
                    cv2.putText(debug_img, f"{name}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                save_debug_image(debug_img, OUT_DIR / f"{debug_name}_{name}_faces.jpg", f"{name} faces")
                
        except Exception as e:
            print(f"Error in {name} preprocessing: {e}")
            continue
    
    # Try different scales
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    for scale in scales:
        try:
            h, w = img.shape[:2]
            if scale != 1.0:
                scaled_img = cv2.resize(img, (int(w*scale), int(h*scale)), 
                                     interpolation=cv2.INTER_CUBIC)
            else:
                scaled_img = img
                
            faces = face_app.get(scaled_img)
            print(f"scale_{scale}: {len(faces)} faces detected")
            
            if faces:
                # Adjust bbox coordinates back to original scale
                for face in faces:
                    face.bbox = [coord/scale for coord in face.bbox]
                    if hasattr(face, 'kps'):
                        face.kps = face.kps / scale
                all_faces.extend(faces)
                if len(faces) > len(best_faces):
                    best_faces = faces
                    
        except Exception as e:
            print(f"Error in scale {scale}: {e}")
            continue
    
    # Remove duplicates (faces with very similar bounding boxes) - FIXED VERSION
    unique_faces = []
    for face in all_faces:
        is_duplicate = False
        for unique_face in unique_faces:
            # Check if bounding boxes overlap significantly
            x1_1, y1_1, x2_1, y2_1 = face.bbox
            x1_2, y1_2, x2_2, y2_2 = unique_face.bbox
            
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            
            # Calculate intersection
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)
            
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                overlap = intersection / min(area1, area2)
                if overlap > 0.5:  # 50% overlap
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_faces.append(face)
    
    print(f"Total unique faces detected: {len(unique_faces)}")
    return unique_faces if unique_faces else best_faces

# --------- Main Execution ---------
try:
    # Initialize with very low threshold
    face_app = setup_face_detector(det_size=(640, 640), threshold=0.1)
    
    # Load local inswapper
    swapper = insightface.model_zoo.get_model(str(Path("models/inswapper_128.onnx")), download=False)
    
    # Load images
    bg = resize_long(read_bgr(BG_PATH), LONG_MAX)
    persona = resize_long(read_bgr(PERSONA_PATH), LONG_MAX)
    
    # Selfie: EXIF-aware + handle transparency / tight crops
    selfie_bgr_raw, selfie_alpha = read_any_exif(SELFIE_PATH)
    selfie = composite_on_gray(selfie_bgr_raw, selfie_alpha, gray=128)
    selfie = pad_and_upscale(selfie, pad_ratio=0.25, min_long=800)  # More padding, lower min size
    selfie = resize_long(selfie, LONG_MAX)
    
    print(f"Selfie final size: {selfie.shape}")
    print(f"Persona final size: {persona.shape}")
    
    # Detect faces with comprehensive approach
    src_faces = detect_faces_comprehensive(face_app, selfie, "selfie")
    tgt_faces = detect_faces_comprehensive(face_app, persona, "persona")
    
    if not src_faces:
        # Ultimate fallback - try manual face detection with OpenCV
        print("\nTrying OpenCV fallback detection...")
        gray = cv2.cvtColor(selfie, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        if len(faces) > 0:
            print(f"OpenCV found {len(faces)} faces!")
            # Convert OpenCV format to insightface-like format
            class SimpleFace:
                def __init__(self, bbox):
                    self.bbox = bbox
            src_faces = [SimpleFace([x, y, x+w, y+h]) for (x, y, w, h) in faces]
            
            # Debug save OpenCV detection
            debug_img = selfie.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            save_debug_image(debug_img, OUT_DIR / "selfie_opencv_faces.jpg", "OpenCV Faces")
    
    if not src_faces:
        raise RuntimeError("""
No face detected in selfie image. Please check:
1. The selfie image is clear and the face is visible
2. The face is not too small in the image
3. Try a front-facing photo with good lighting
4. Check the debug images in outputs/ folder to see what the detector sees
        """)
    
    if not tgt_faces:
        raise RuntimeError("No face detected in persona image.")
    
    src = largest_face(src_faces)
    tgt = largest_face(tgt_faces)
    
    print(f"Using source face bbox: {src.bbox}")
    print(f"Using target face bbox: {tgt.bbox}")
    
    # Continue with face swap...
    mask_soft = ellipse_mask_from_landmarks(persona, tgt, expand_px=MASK_EXPAND_PX)
    mask_bin  = (mask_soft > 0.5).astype(np.uint8) * 255
    x1, y1, x2, y2 = map(int, tgt.bbox)
    center = ((x1 + x2)//2, (y1 + y2)//2)
    
    # Swap
    print("Performing face swap...")
    swapped = swapper.get(persona.copy(), tgt, src, paste_back=True)
    
    # Color match
    swapped_col = lab_hist_match(swapped, persona, mask=(mask_soft > 0.2).astype(np.uint8))
    
    # Blend
    if USE_POISSON:
        src_part = persona.copy()
        idx = mask_soft > 0.01
        src_part[idx] = swapped_col[idx]
        blended_persona = poisson(src_part, persona, mask_bin, center)
    else:
        feathered = feather(mask_soft, FEATHER_PX).astype(np.float32)
        blended_persona = (swapped_col * feathered[...,None] + persona * (1 - feathered[...,None])).astype(np.uint8)
    
    # Remove background
    print("Removing background...")
    persona_rgba = remove_bg_rgba(blended_persona)
    alpha = erode_feather_alpha(persona_rgba[:,:,3], erode_px=2, feather_px=3)
    persona_rgba[:,:,3] = alpha
    
    # Composite
    print("Compositing final image...")
    bh, bw = bg.shape[:2]; ph, pw = persona_rgba.shape[:2]
    scale = min(0.92 * bh / ph, 0.92 * bw / pw)
    new_size = (int(pw * scale), int(ph * scale))
    persona_rgba = cv2.resize(persona_rgba, new_size, cv2.INTER_AREA)
    ph, pw = persona_rgba.shape[:2]
    x = (bw - pw) // 2; y = (bh - ph) // 2
    composite = alpha_over_premul(persona_rgba, bg, x, y)
    
    # Final touch
    final = add_micrograin(composite, strength=GRAIN_STRENGTH)
    
    cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(OUT_PATH))
    print(f"Success! Saved: {OUT_PATH}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting tips:")
    print("1. Check that your selfie image has a clear, visible face")
    print("2. Try a different selfie image")
    print("3. Check the debug images in outputs/ folder")
    print("4. Make sure the face isn't obscured or at extreme angles")