"""课程训练图实际使用的八种视图。number 是原图在排序后的训练集中的序号。"""

import cv2
import numpy as np

VIEWS = ("full", "flip", "center", "jitter", "geometry2", "illumination2", "capture2", "color_swap")
MEAN = np.array([0.485, 0.456, 0.406]) * 255
STD = np.array([0.229, 0.224, 0.225]) * 255


def image_view(image, view, number):
    rgb = cv2.resize(image, (224, 224))
    if view == "flip":
        rgb = rgb[:, ::-1]
    elif view == "center":
        rgb = cv2.resize(rgb[14:210, 14:210], (224, 224))
    elif view == "jitter":
        rng = np.random.default_rng(42 + number)
        side = int(rng.integers(180, 220))
        y, x = rng.integers(0, 225 - side, size=2)
        rgb = cv2.resize(rgb[y:y + side, x:x + side], (224, 224))
        if rng.random() < 0.5:
            rgb = rgb[:, ::-1]
    elif view == "geometry2":
        rng = np.random.default_rng(33013 + number)
        angle = rng.uniform(-13, 13)
        affine = cv2.getRotationMatrix2D((111.5, 111.5), angle, rng.uniform(0.90, 1.11))
        affine[:, 2] += rng.uniform(-9, 9, size=2)
        rgb = cv2.warpAffine(rgb, affine, (224, 224), borderMode=cv2.BORDER_REFLECT_101)
        if rng.random() < 0.5:
            rgb = rgb[:, ::-1]
    elif view == "illumination2":
        rng = np.random.default_rng(71111 + number)
        rgb = rgb.astype(np.float32) / 255
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[:, :, None]
        rgb = np.clip(gray + rng.uniform(0.25, 1.35) * (rgb - gray), 0, 1)
        gain = rng.uniform(0.84, 1.14, size=3)
        rgb = (np.clip(rgb ** rng.uniform(0.72, 1.37) * gain, 0, 1) * 255).astype(np.uint8)
    elif view == "capture2":
        rng = np.random.default_rng(81861 + number)
        side = int(rng.integers(112, 218))
        rgb = cv2.resize(cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA), (224, 224))
        ok, encoded = cv2.imencode(".jpg", rgb[:, :, ::-1],
                                   [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(46, 85))])
        if not ok:
            raise ValueError("JPEG 编码失败")
        rgb = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[:, :, ::-1]
    elif view == "color_swap":
        rgb = rgb[:, :, ::-1]
    elif view != "full":
        raise ValueError(f"未知增强方式 {view}")
    return ((rgb - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)
