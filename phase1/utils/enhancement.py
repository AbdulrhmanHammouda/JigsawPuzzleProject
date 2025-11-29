import cv2
import numpy as np

# -------------------------
# Helpers / Utilities
# -------------------------
def _to_uint8(img):
    """Ensure uint8 and clip values."""
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def compute_luminance_stats(img):
    """Return mean/std of luminance (Y channel) for debugging/adaptive choices."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    return float(L.mean()), float(L.std())


# -------------------------
# 1) DENOISE (conservative)
# -------------------------
def denoise_image(img, method="fastnlm", h=10, templateWindowSize=7, searchWindowSize=21):
    """
    Denoise using either:
     - 'fastnlm' (cv2.fastNlMeansDenoisingColored) : good for color images, preserves edges
     - 'bilateral' : preserves edges, but slower for very large images
    Safe defaults are conservative to avoid over-smoothing.
    """
    if method == "bilateral":
        # smaller d to avoid over-smoothing: keeps edges
        return cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)
    else:
        # fast non-local means (recommended default)
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, searchWindowSize)


# -------------------------
# 2) CLAHE (safe / adaptive)
# -------------------------
def apply_safe_clahe(img, clipLimit=1.2, tileGridSize=(8,8), min_clip=0.5, max_clip=3.0):
    """
    Apply CLAHE on L channel in LAB color space with conservative defaults.
    This function checks image luminance and reduces clipLimit if the image is already bright,
    avoiding local over-amplification.
    """
    # get luminance stats
    meanL, stdL = compute_luminance_stats(img)

    # adapt clipLimit based on luminance: brighter images -> smaller clipLimit
    # user-provided clipLimit acts as base
    if meanL > 150:
        adaptive_clip = max(min_clip, clipLimit * 0.6)
    elif meanL > 120:
        adaptive_clip = max(min_clip, clipLimit * 0.85)
    else:
        adaptive_clip = clipLimit

    adaptive_clip = float(np.clip(adaptive_clip, min_clip, max_clip))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=adaptive_clip, tileGridSize=tileGridSize)
    L2 = clahe.apply(L)

    merged = cv2.merge([L2, A, B])
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out


# -------------------------
# 3) Unsharp Mask (gentle sharpening)
# -------------------------
def unsharp_mask(img, sigma=1.0, strength=0.7):
    """
    Gentle unsharp mask:
      blurred = Gaussian(img, sigma)
      result = img + strength * (img - blurred)
    Strength default 0.7 (conservative).
    Avoids the harsh artifacts of simple high-gain convolution kernels.
    """
    img_f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0,0), sigmaX=sigma, sigmaY=sigma)
    mask = img_f - blurred
    result = img_f + strength * mask
    return _to_uint8(result)


# -------------------------
# 4) Small saturation boost (safe)
# -------------------------
def boost_saturation(img, factor=1.03):
    """
    Slight saturation boost in HSV space with clipping; default minimal (3%).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# -------------------------
# 5) Gamma correction (optional)
# -------------------------
def adjust_gamma(img, gamma=1.0):
    """
    Gamma correction. gamma < 1 darkens, >1 brightens.
    Default 1.0 (no change). Use e.g. gamma=0.95 to prevent blowout.
    """
    if abs(gamma - 1.0) < 1e-6:
        return img
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


# -------------------------
# 6) FULL ENHANCEMENT PIPELINE (conservative defaults)
# -------------------------
def enhance_full_image(img,
    denoise_method="fastnlm",
denoise_h=8,
clahe_clip=0.8,
clahe_grid=(8,8),
sharpen_sigma=1.0,
sharpen_strength=0.5,
sat_factor=1.02,
gamma=0.97
, return_intermediates=False):
    """
    A safer enhancement pipeline with tunable parameters.
    Default settings are conservative to avoid blowouts.
    Returns dict with 'denoised','clahe','sharpened','final'
    """
    # 1. denoise
    den = denoise_image(img) if denoise_method else img.copy()

    # 2. CLAHE (adaptive)
    clahe_img = apply_safe_clahe(den, clipLimit=clahe_clip, tileGridSize=clahe_grid)

    # 3. gentle unsharp mask
    sharpened = unsharp_mask(clahe_img, sigma=sharpen_sigma, strength=sharpen_strength)

    # 4. optional gamma correction (use gamma < 1 to slightly darken bright images)
    gamma_corr = adjust_gamma(sharpened, gamma=gamma)

    # 5. slight saturation boost
    final = boost_saturation(gamma_corr, factor=sat_factor)

    outputs = {
        "denoised": den,
        "clahe": clahe_img,
        "sharpened": sharpened,
        "gamma": gamma_corr,
        "final": final
    }

    return outputs if return_intermediates else {"final": final}


# -------------------------
# Example recommended defaults to try interactively
# -------------------------
# - For most images: clahe_clip=1.2, clahe_grid=(8,8), sharpen_sigma=1.0, sharpen_strength=0.6
# - If you see blown highlights: reduce clahe_clip to 0.8-1.0 and/or reduce sharpen_strength
# - If colors over-saturate: reduce sat_factor to 1.01 or set to 1.0
# - If images are too bright after enhancement: try gamma = 0.95
#
# Use return_intermediates=True to debug intermediate results:
# outs = enhance_full_image_safe(img, return_intermediates=True)
# plt.imshow(cv2.cvtColor(outs['final'], cv2.COLOR_BGR2RGB))
