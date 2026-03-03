import argparse
import os
import sys
import cv2
import numpy as np





# Initialize the meter
cvtimer = cv2.TickMeter()





def count_non_black_pixels(img: np.ndarray) -> int:
    if img is None:
        return 0
    arr = img.astype(np.uint8)
    if arr.ndim == 2:
        return int(np.count_nonzero(arr != 0))
    if arr.ndim == 3:
        mask = np.any(arr != 0, axis=2)
        return int(np.count_nonzero(mask))
    raise ValueError("Unsupported image shape: " + str(arr.shape))


def block_average_gray(gray: np.ndarray, block_size: int = 6) -> np.ndarray:
    h, w = gray.shape[:2]
    out = gray.copy()
    bs = int(block_size)
    if bs <= 0:
        return out

    for y in range(0, h, bs):
        y2 = min(y + bs, h)
        for x in range(0, w, bs):
            x2 = min(x + bs, w)
            block = gray[y:y2, x:x2]
            mean_val = int(np.round(block.mean()))
            out[y:y2, x:x2] = mean_val

    return out


def increase_contrast(image: np.ndarray, method: str = 'clahe', clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Increase image contrast.

    Parameters:
        image: BGR or grayscale image as numpy array.
        method: 'clahe' (default) uses CLAHE on L-channel (color) or directly on gray.
                'hist' uses histogram equalization for grayscale or Y channel for color.
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid size.

    Returns:
        Contrast-enhanced image with same shape and dtype as input.
    """
    # Grayscale image
    if image.ndim == 2 or image.shape[2] == 1:
        gray = image if image.ndim == 2 else image[:, :, 0]
        if method == 'hist':
            return cv2.equalizeHist(gray)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    # Color image (BGR)
    if method == 'hist':
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    # CLAHE on L channel in LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def threshold_to_black(img: np.ndarray, thresh: int = 150) -> np.ndarray:
    if img is None:
        return img

    # Ensure uint8
    img_u8 = img.astype(np.uint8)

    if img_u8.ndim == 3 and img_u8.shape[2] == 3:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
        mask = gray < int(thresh)
        out = img_u8.copy()
        out[mask] = (0, 0, 0)
        return out
    else:
        # single-channel
        out = img_u8.copy()
        out[out < int(thresh)] = 0
        return out




def build_parser():
    p = argparse.ArgumentParser(description='Block-average (pixelate) grayscale image')
    p.add_argument('image', help='Input image path')
    p.add_argument('--out', '-o', help='Output image path (default: input_blockavg_NxN.png)', default=None)
    p.add_argument('--block-size', type=int, default=6, help='Block size N (default 6)')
    p.add_argument('--threshold', type=int, default=180, help='Gray Scale Threshold')
    p.add_argument('--show', action='store_true', help='Show result in a window')
    return p


def main():
    args = build_parser().parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        print('Error: could not read image:', args.image, file=sys.stderr)
        sys.exit(2)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    cvtimer.start()
    out_img = increase_contrast(gray, method='clahe', clip_limit=4.0, tile_grid_size=(8, 8))
    out_img = block_average_gray(out_img, block_size=args.block_size)
    out_img = threshold_to_black(out_img, thresh=args.threshold)
    non_black_pixel = count_non_black_pixels(out_img)

    print(f'Non-black pixels: {non_black_pixel * 100/ 3000:.2f}%')
    cvtimer.stop()
    if non_black_pixel * 100/ 3000 > 100:
        print('The Camera is focused')
    elif non_black_pixel * 100/ 3000 > 75:
        print('The Camera is likely focused')
    else:
        print('The Camera is likely not focused')

    
    print(f"Time in milliseconds: {cvtimer.getTimeMilli()}")


    if args.out:
        out_path = args.out
    else:
        base = os.path.splitext(os.path.basename(args.image))[0]
        out_path = os.path.join(os.path.dirname(args.image) or '.', f"{base}_blockavg_{args.block_size}x{args.block_size}.png")

    cv2.imwrite(out_path, out_img)
    print('Wrote:', out_path)

    if args.show:
        cv2.imshow('blockavg', out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()