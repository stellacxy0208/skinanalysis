import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

mp_face_mesh = mp.solutions.face_mesh

# Face oval & exclusion regions (MediaPipe indices)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109
]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]


@dataclass
class AnalysisResult:
    spots: np.ndarray
    wrinkles: np.ndarray
    texture: np.ndarray
    pores: np.ndarray
    red_areas: np.ndarray
    porphyrin: np.ndarray  # approximate


def _normalize_0_255(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = np.percentile(x, 2), np.percentile(x, 98)
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn)
    y = np.clip(y, 0, 1)
    return (y * 255).astype(np.uint8)


def _landmarks_to_points(landmarks, idxs, w, h):
    pts = []
    for i in idxs:
        lm = landmarks[i]
        pts.append((int(lm.x * w), int(lm.y * h)))
    return np.array(pts, dtype=np.int32)


def build_skin_mask(bgr: np.ndarray) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lms = res.multi_face_landmarks[0].landmark
        face_oval = _landmarks_to_points(lms, FACE_OVAL, w, h)
        left_eye = _landmarks_to_points(lms, LEFT_EYE, w, h)
        right_eye = _landmarks_to_points(lms, RIGHT_EYE, w, h)
        mouth = _landmarks_to_points(lms, MOUTH, w, h)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_oval], 255)
        cv2.fillPoly(mask, [left_eye], 0)
        cv2.fillPoly(mask, [right_eye], 0)
        cv2.fillPoly(mask, [mouth], 0)

        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        mask = (mask > 30).astype(np.uint8) * 255
        return mask


def compute_maps(bgr: np.ndarray, skin_mask: np.ndarray) -> AnalysisResult:
    mask = (skin_mask > 0).astype(np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)

    # Spots (pigment-ish): dark + local chroma contrast
    L_blur = cv2.GaussianBlur(L, (0, 0), 2)
    dark = np.clip((L_blur.astype(np.float32) - L.astype(np.float32)), 0, None)

    B_blur = cv2.GaussianBlur(B, (0, 0), 3)
    b_contrast = cv2.absdiff(B, B_blur).astype(np.float32)

    spots_score = 0.7 * dark + 0.3 * b_contrast
    spots_u8 = (_normalize_0_255(spots_score) * mask).astype(np.uint8)
    _, spots_bin = cv2.threshold(spots_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    spots_bin = cv2.morphologyEx(spots_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Wrinkles: gradients
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)
    grad_u8 = (_normalize_0_255(grad) * mask).astype(np.uint8)
    _, wrinkles_bin = cv2.threshold(grad_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    wrinkles_bin = cv2.morphologyEx(wrinkles_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    # Texture: high frequency energy
    blur_small = cv2.GaussianBlur(gray, (0, 0), 1)
    blur_large = cv2.GaussianBlur(gray, (0, 0), 5)
    highpass = cv2.absdiff(blur_small, blur_large).astype(np.float32)
    texture_u8 = (_normalize_0_255(highpass) * mask).astype(np.uint8)
    _, texture_bin = cv2.threshold(texture_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    texture_bin = cv2.morphologyEx(texture_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    # Pores: Laplacian-ish blobs
    log = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.6), cv2.CV_32F, ksize=3)
    log = np.abs(log)
    pores_u8 = (_normalize_0_255(log) * mask).astype(np.uint8)
    if np.any(mask):
        thr = int(np.clip(np.percentile(pores_u8[mask > 0], 85), 80, 230))
    else:
        thr = 200
    _, pores_bin = cv2.threshold(pores_u8, thr, 255, cv2.THRESH_BINARY)
    pores_bin = cv2.morphologyEx(pores_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    # Red areas: LAB A-channel
    A_norm = _normalize_0_255(A.astype(np.float32))
    red_u8 = (A_norm * mask).astype(np.uint8)
    if np.any(mask):
        thr_red = int(np.clip(np.percentile(red_u8[mask > 0], 80), 90, 230))
    else:
        thr_red = 200
    _, red_bin = cv2.threshold(red_u8, thr_red, 255, cv2.THRESH_BINARY)
    red_bin = cv2.morphologyEx(red_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # "Porphyrin" approx: saturation + red dominance
    b, g, r = cv2.split(bgr)
    r_dom = np.clip(r.astype(np.float32) - 0.5 * g.astype(np.float32) - 0.5 * b.astype(np.float32), 0, None)
    por_score = 0.55 * r_dom + 0.45 * S.astype(np.float32)
    por_u8 = (_normalize_0_255(por_score) * mask).astype(np.uint8)
    if np.any(mask):
        thr_por = int(np.clip(np.percentile(por_u8[mask > 0], 82), 100, 240))
    else:
        thr_por = 210
    _, por_bin = cv2.threshold(por_u8, thr_por, 255, cv2.THRESH_BINARY)
    por_bin = cv2.morphologyEx(por_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    return AnalysisResult(
        spots=spots_bin,
        wrinkles=wrinkles_bin,
        texture=texture_bin,
        pores=pores_bin,
        red_areas=red_bin,
        porphyrin=por_bin,
    )


def _overlay_points(base_bgr, mask_u8, color=(0, 255, 255), alpha=0.75):
    out = base_bgr.copy()
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return out
    step = max(1, int(len(xs) / 8000))
    xs, ys = xs[::step], ys[::step]
    for x, y in zip(xs, ys):
        cv2.circle(out, (int(x), int(y)), 1, color, -1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(out, alpha, base_bgr, 1 - alpha, 0)


def _overlay_mask(base_bgr, mask_u8, color=(0, 255, 255), alpha=0.35):
    out = base_bgr.copy()
    color_layer = np.zeros_like(out)
    color_layer[:] = color
    m = (mask_u8 > 0).astype(np.uint8)
    out = out * (1 - (m[..., None] * alpha)) + color_layer * (m[..., None] * alpha)
    return out.astype(np.uint8)


def _normalize_heat(gray_u8):
    return _normalize_0_255(gray_u8)
    def compute_uv_spots_visia_approx(bgr: np.ndarray, skin_mask: np.ndarray):
    """
    VISIA-like UV Spots approximation from a single RGB photo.
    Returns:
      uv_bg_u8: grayscale 0-255 background (negative look)
      uv_spots_bin: binary mask for yellow dots
    """
    mask = (skin_mask > 0).astype(np.uint8)

    # 1) grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 2) illumination correction (remove large-scale lighting)
    blur_big = cv2.GaussianBlur(gray, (0, 0), 18)
    corrected = cv2.subtract(gray, blur_big)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3) local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)

    # 4) background negative look (VISIA-ish)
    uv_bg = 255 - enhanced
    uv_bg = (uv_bg * mask).astype(np.uint8)

    # 5) spot candidate map: Difference of Gaussians (blob emphasis)
    g1 = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
    g2 = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
    dog = cv2.subtract(g2, g1)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dog = (dog * mask).astype(np.uint8)

    # 6) threshold using percentile (more "device-like" than OTSU)
    vals = dog[mask > 0]
    thr = int(np.percentile(vals, 92)) if vals.size else 200
    _, spots_bin = cv2.threshold(dog, thr, 255, cv2.THRESH_BINARY)

    # 7) cleanup
    spots_bin = cv2.morphologyEx(spots_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    return uv_bg, spots_bin


def make_panel(bgr: np.ndarray, skin_mask: np.ndarray, res: AnalysisResult) -> np.ndarray:
    base = bgr.copy()
    pad = 18

    def make_tile(title, overlay_mask_u8=None, heat_u8=None, tint_color=(0, 255, 255), mode="dots"):
        tile = base.copy()
        if overlay_mask_u8 is not None:
            if mode == "dots":
                tile = _overlay_points(tile, overlay_mask_u8, color=tint_color, alpha=0.85)
            else:
                tile = _overlay_mask(tile, overlay_mask_u8, color=tint_color, alpha=0.35)
        if heat_u8 is not None:
            # If heat_u8 is grayscale, keep VISIA-like BW look (no JET colormap)
                heat_norm = _normalize_heat(heat_u8)
            if len(heat_norm.shape) == 2:
                 hm = cv2.cvtColor(heat_norm, cv2.COLOR_GRAY2BGR)
            else:
                 hm = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
                 tile = cv2.addWeighted(tile, 0.55, hm, 0.45, 0)


    # Top row
    t1 = make_tile(f"Spots ({int(res.spots.sum()/255)})", res.spots, tint_color=(0, 255, 255), mode="dots")
    t2 = make_tile(f"Wrinkles ({int(res.wrinkles.sum()/255)})", res.wrinkles, tint_color=(0, 255, 0), mode="dots")
    t3 = make_tile(f"Texture ({int(res.texture.sum()/255)})", res.texture, tint_color=(0, 200, 255), mode="dots")
    t4 = make_tile(f"Pores ({int(res.pores.sum()/255)})", res.pores, tint_color=(255, 120, 0), mode="dots")

    # Bottom row (approx)
    uv_bg_u8, uv_spots_bin = compute_uv_spots_visia_approx(base, skin_mask)

    lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
    brown_like = _normalize_0_255(lab[:, :, 2])
    brown_like = (brown_like * (skin_mask > 0)).astype(np.uint8)

    b1 = make_tile("UV Spots (approx)", heat_u8=uv_bg_u8)
    b1 = _overlay_points(b1, uv_spots_bin, color=(0, 255, 255), alpha=0.9)
    b2 = make_tile("Brown Spots (approx)", heat_u8=brown_like)
    b3 = make_tile(f"Red Areas ({int(res.red_areas.sum()/255)})", res.red_areas, tint_color=(0, 0, 255), mode="mask")
    b4 = make_tile(f"Porphyrins (approx) ({int(res.porphyrin.sum()/255)})", res.porphyrin, tint_color=(255, 0, 255), mode="mask")

    top = np.hstack([t1, t2, t3, t4])
    bot = np.hstack([b1, b2, b3, b4])

    bg = np.zeros((top.shape[0] + bot.shape[0] + pad * 3, top.shape[1] + pad * 2, 3), dtype=np.uint8)
    bg[:] = (20, 20, 20)

    bg[pad:pad + top.shape[0], pad:pad + top.shape[1]] = top
    bg[pad * 2 + top.shape[0]:pad * 2 + top.shape[0] + bot.shape[0], pad:pad + bot.shape[1]] = bot
    return bg


def run_analysis(input_path: str, output_path: str = "analysis_panel.png", max_width: int = 900) -> str:
    bgr = cv2.imread(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    h, w = bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    skin_mask = build_skin_mask(bgr)
    if skin_mask is None:
        raise RuntimeError("No face detected. Try a clearer, front-facing photo with good lighting.")

    res = compute_maps(bgr, skin_mask)
    panel = make_panel(bgr, skin_mask, res)
    cv2.imwrite(output_path, panel)
    return output_path

