import os
import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ai_focus.py
# Usage: python ai_focus.py --img path/to/image.png --model path/to/model.h5 --out out.png
# Requires: opencv-python, numpy, tensorflow (keras)


def increase_contrast_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def block_average_gray_color(bgr, block_size=6):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # pad so dims divisible by block_size
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    gray_p = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    H, W = gray_p.shape
    # reshape into blocks and compute mean per block
    gray_blocks = gray_p.reshape(H//block_size, block_size, W//block_size, block_size)
    means = gray_blocks.mean(axis=(1,3)).astype(np.uint8)  # shape (Hb, Wb)
    # expand means back to block image
    expanded = np.repeat(np.repeat(means, block_size, axis=0), block_size, axis=1)
    expanded = expanded[:h, :w]  # crop to original
    # make 3-channel color image where each block has the gray as rgb
    colored = cv2.cvtColor(expanded, cv2.COLOR_GRAY2BGR)
    return colored

def prepare_for_model(img_bgr, model):
    # model may accept fixed size; resize to model input if necessary
    input_shape = model.input_shape  # e.g. (None, H, W, C) or (None, C)
    if len(input_shape) >= 4:
        _, H, W, C = input_shape
        if H is None or W is None:
            arr = img_bgr.astype(np.float32) / 255.0
            if arr.shape[-1] != C and C == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)[..., None]
            return np.expand_dims(arr, 0)
        resized = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        if C == 1:
            arr = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)[..., None].astype(np.float32) / 255.0
        return np.expand_dims(arr, 0)
    else:
        # fallback: flatten input
        arr = img_bgr.astype(np.float32) / 255.0
        return np.expand_dims(arr, 0)

def detect_bright_dots(processed_bgr, model, threshold=0.5, min_distance=3):
    inp = prepare_for_model(processed_bgr, model)
    pred = model.predict(inp)
    # try to extract a heatmap-like output
    heat = np.squeeze(pred)
    # if output has channel dim, take first channel
    if heat.ndim == 3:
        heat = heat[..., 0]
    # resize heatmap to image size if needed
    if heat.shape != processed_bgr.shape[:2]:
        heat = cv2.resize(heat, (processed_bgr.shape[1], processed_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    # normalize heat to 0..1
    if heat.max() > 0:
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    # find local maxima above threshold
    thresh = (heat > threshold).astype(np.uint8)
    if thresh.sum() == 0:
        return [], heat
    # local maxima via dilation
    kernel = np.ones((2*min_distance+1, 2*min_distance+1), np.uint8)
    dil = cv2.dilate(heat, kernel)
    peaks = (heat == dil) & (heat > threshold)
    ys, xs = np.where(peaks)
    scores = heat[ys, xs]
    detections = sorted([(int(x), int(y), float(s)) for x,y,s in zip(xs, ys, scores)], key=lambda x: -x[2])
    return detections, heat

def overlay_detections(img_bgr, detections, heat=None):
    out = img_bgr.copy()
    for x,y,s in detections:
        color = (0, int(255*s), int(255*(1-s)))
        cv2.circle(out, (x,y), 6, color, 2)
        cv2.putText(out, f"{s:.2f}", (x+8,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    if heat is not None:
        heat_vis = (np.clip(heat,0,1)*255).astype(np.uint8)
        heat_vis = cv2.applyColorMap(heat_vis, cv2.COLORMAP_JET)
        blend = cv2.addWeighted(out, 0.6, heat_vis, 0.4, 0)
        return blend
    return out

def main():
    parser = argparse.ArgumentParser(description="Contrast -> 6x6 block grayscale -> detect bright dots via NN")
    parser.add_argument("--img", required=True, help="input image path")
    parser.add_argument("--model", required=True, help="keras .h5 model path (model outputs heatmap)")
    parser.add_argument("--out", default="out.png", help="output visualization")
    parser.add_argument("--threshold", type=float, default=0.4, help="detection threshold on model heatmap (0..1)")
    args = parser.parse_args()

    if not os.path.exists(args.img):
        raise SystemExit("Image not found: " + args.img)
    if not os.path.exists(args.model):
        raise SystemExit("Model not found: " + args.model)

    img = cv2.imread(args.img, cv2.IMREAD_COLOR)
    img_contrast = increase_contrast_clahe(img)
    img_block = block_average_gray_color(img_contrast, block_size=6)

    model = load_model(args.model)
    detections, heat = detect_bright_dots(img_block, model, threshold=args.threshold)

    vis = overlay_detections(img_block, detections, heat=heat)
    cv2.imwrite(args.out, vis)
    # print detections
    for x,y,s in detections:
        print(f"{x},{y},{s:.4f}")

if __name__ == "__main__":
    main()