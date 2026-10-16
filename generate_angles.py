"""Generate realistic multi-angle camera shots of the real glass plate with optical glare and perspective tilts."""

import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BASE_IMG_PATH = DATA_DIR / "real_glass_plate_topdown.png"

# Angles to generate:
# 1. view_angle_left: Camera tilted from left (yaw ~ +25°, pitch ~ 15°) + specular reflection
# 2. view_angle_right: Camera tilted from right (yaw ~ -25°, pitch ~ 18°) + glare streak
# 3. view_angle_top: Camera angled from top forward (pitch ~ 30°) + soft ambient glow
# 4. view_angle_bottom: Camera angled from bottom forward (pitch ~ -28°) + bright flash glare


def add_glare(image: np.ndarray, center: tuple[int, int], radius: tuple[int, int], alpha: float, angle_deg: float = 0.0) -> np.ndarray:
    """Add a realistic physical specular glare onto the image."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, center, radius, angle_deg, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius[0] * 0.45, sigmaY=radius[1] * 0.45)
    
    # Screen blend white glare
    img_f = image.astype(np.float32)
    glare_color = np.array([255.0, 255.0, 250.0], dtype=np.float32)
    
    alpha_3d = (mask * alpha)[:, :, None]
    # Screen: 255 - (255 - img) * (255 - glare) / 255
    screened = 255.0 - (255.0 - img_f) * (255.0 - glare_color) / 255.0
    blended = img_f * (1.0 - alpha_3d) + screened * alpha_3d
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_view(
    base: np.ndarray,
    quad_dest: np.ndarray,
    glare_params: list[dict],
    out_w: int = 1200,
    out_h: int = 900,
    border_color: tuple[int, int, int] = (25, 20, 15),
) -> np.ndarray:
    """Warp the base plate into a tilted perspective quad on a dark wooden background, then add glare."""
    h, w = base.shape[:2]
    src_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    M = cv2.getPerspectiveTransform(src_corners, quad_dest)
    
    # Create dark textured wooden table background
    table = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    table[:] = border_color
    noise = np.random.normal(0, 5, (out_h, out_w, 3)).astype(np.int16)
    table = np.clip(table.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    warped_plate = cv2.warpPerspective(base, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    plate_mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, M, (out_w, out_h), flags=cv2.INTER_NEAREST)
    
    # Soften plate boundary slightly
    mask_blurred = cv2.GaussianBlur(plate_mask.astype(np.float32) / 255.0, (5, 5), 0)[:, :, None]
    composite = (warped_plate.astype(np.float32) * mask_blurred + table.astype(np.float32) * (1.0 - mask_blurred)).astype(np.uint8)
    
    # Add specular glares
    result = composite.copy()
    for gp in glare_params:
        result = add_glare(result, gp["center"], gp["radius"], gp["alpha"], gp.get("angle", 0.0))
        
    return result


def main():
    base = cv2.imread(str(BASE_IMG_PATH))
    if base is None:
        raise FileNotFoundError(f"Base image not found at {BASE_IMG_PATH}")
    
    h, w = base.shape[:2]
    
    # 1. View 1: Left Angled Shot (Yaw ~ +25°)
    q_left = np.float32([
        [240, 120],  # Top-left (farther, smaller)
        [1060, 80],  # Top-right
        [1140, 820], # Bottom-right (closer, larger)
        [160, 780]   # Bottom-left
    ])
    g_left = [
        {"center": (700, 380), "radius": (180, 70), "alpha": 0.82, "angle": 25.0},
        {"center": (450, 600), "radius": (90, 40), "alpha": 0.65, "angle": -15.0},
    ]
    view_left = generate_view(base, q_left, g_left)
    cv2.imwrite(str(DATA_DIR / "real_glass_angle_left.png"), view_left)
    print("Saved real_glass_angle_left.png")

    # 2. View 2: Right Angled Shot (Yaw ~ -25°)
    q_right = np.float32([
        [140, 90],   # Top-left
        [980, 130],  # Top-right (farther, smaller)
        [1050, 780], # Bottom-right
        [80, 830]    # Bottom-left (closer, larger)
    ])
    g_right = [
        {"center": (480, 320), "radius": (190, 80), "alpha": 0.85, "angle": -30.0},
        {"center": (850, 520), "radius": (80, 35), "alpha": 0.70, "angle": 20.0},
    ]
    view_right = generate_view(base, q_right, g_right)
    cv2.imwrite(str(DATA_DIR / "real_glass_angle_right.png"), view_right)
    print("Saved real_glass_angle_right.png")

    # 3. View 3: Top-Tilted Shot (Pitch ~ 30°)
    q_top = np.float32([
        [260, 80],   # Top-left
        [940, 80],   # Top-right
        [1120, 820], # Bottom-right
        [80, 820]    # Bottom-left
    ])
    g_top = [
        {"center": (600, 480), "radius": (220, 90), "alpha": 0.80, "angle": 5.0},
    ]
    view_top = generate_view(base, q_top, g_top)
    cv2.imwrite(str(DATA_DIR / "real_glass_angle_pitch.png"), view_top)
    print("Saved real_glass_angle_pitch.png")

    print("All multi-angle camera shots successfully created!")


if __name__ == "__main__":
    main()
