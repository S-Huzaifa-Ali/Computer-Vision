"""
Computer Vision (AI-4002) - Lab 06 solutions
Wavelets, Boundary Detection, Hough Transform & SIFT

Usage (each task is a sub-command):
  python lab06_solutions.py screens   --image lab.jpg [--expected 12]
  python lab06_solutions.py assets    --image lab.jpg --refs monitor=mon.jpg keyboard=kb.jpg
  python lab06_solutions.py wavelet   [--csv sensor.csv]
  python lab06_solutions.py objvideo  --ref object.jpg --video clip.mp4   (or --video 0 for webcam)
  python lab06_solutions.py panorama  --images a.jpg b.jpg c.jpg
  python lab06_solutions.py lanes     --image road.jpg   (or --video road.mp4)
  python lab06_solutions.py coins     --image coins.jpg
  python lab06_solutions.py security  --video 0 [--zone x1 y1 x2 y2]

Requires: opencv-python, numpy, matplotlib, PyWavelets
"""
import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    return img


def show(img_bgr, title="", save=None):
    """Display a BGR image with matplotlib (and optionally save it)."""
    if save:
        cv2.imwrite(save, img_bgr)
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def sift_match(des_ref, des_scene, ratio=0.75):
    """FLANN KNN matching + Lowe's ratio test."""
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    flann = cv2.FlannBasedMatcher(index_params, dict(checks=50))
    knn = flann.knnMatch(des_ref, des_scene, k=2)
    return [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]


def locate_with_homography(kp_ref, kp_scene, good, ref_shape, min_matches=10):
    """RANSAC homography. Returns (4x2 polygon in scene, inlier mask) or (None, None)."""
    if len(good) < min_matches:
        return None, None
    src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask.sum() < min_matches:
        return None, None
    h, w = ref_shape[:2]
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    poly = cv2.perspectiveTransform(corners, H)
    # reject degenerate / non-convex projections
    if not cv2.isContourConvex(np.int32(poly)):
        return None, None
    return poly.reshape(4, 2), mask


# ----------------------------------------------------------------------------
# Lab-timing Task 1: Computer screen detection (Hough Line Transform)
# ----------------------------------------------------------------------------
def detect_screens(image_path, expected=None, on_threshold=90, save="screens_out.jpg"):
    """
    Pipeline: gray -> blur -> Canny -> HoughLinesP -> keep near-horizontal /
    near-vertical lines (screen bezels) -> draw them on a mask -> close gaps ->
    contours -> 4-sided boxes = screens.  Brightness inside a box decides ON/OFF,
    and gaps in row spacing (or expected count) reveal missing screens.
    """
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=min(img.shape[:2]) // 12, maxLineGap=20)
    mask = np.zeros_like(gray)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang < 10 or ang > 170 or 80 < ang < 100:      # horizontal or vertical only
                cv2.line(mask, (x1, y1), (x2, y2), 255, 5)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img.shape[0] * img.shape[1]
    screens = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ar = w / float(h)
        if 0.005 * img_area < area < 0.5 * img_area and 0.8 < ar < 2.5:
            roi = gray[y:y + h, x:x + w]
            on = roi.mean() > on_threshold
            screens.append(dict(box=(x, y, w, h), on=on, brightness=float(roi.mean())))

    # ---- anomaly: missing screens ----
    missing_slots = []
    if screens:
        screens.sort(key=lambda s: (s["box"][1], s["box"][0]))
        # group into rows by y-centre
        rows, med_h = [], np.median([s["box"][3] for s in screens])
        for s in screens:
            cy = s["box"][1] + s["box"][3] / 2
            for r in rows:
                if abs(r["cy"] - cy) < med_h * 0.6:
                    r["items"].append(s)
                    break
            else:
                rows.append(dict(cy=cy, items=[s]))
        med_w = np.median([s["box"][2] for s in screens])
        for r in rows:
            r["items"].sort(key=lambda s: s["box"][0])
            for a, b in zip(r["items"], r["items"][1:]):
                gap = b["box"][0] - (a["box"][0] + a["box"][2])
                if gap > 1.2 * med_w:                         # a whole screen-width is empty
                    n_missing = int(round(gap / (med_w * 1.1)))
                    for k in range(n_missing):
                        mx = a["box"][0] + a["box"][2] + int((k + 0.5) * gap / n_missing - med_w / 2)
                        missing_slots.append((mx, int(r["cy"] - med_h / 2), int(med_w), int(med_h)))

    out = img.copy()
    for s in screens:
        x, y, w, h = s["box"]
        col = (0, 200, 0) if s["on"] else (0, 165, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 3)
        cv2.putText(out, "ON" if s["on"] else "OFF", (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    for x, y, w, h in missing_slots:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(out, "MISSING", (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    n_on = sum(s["on"] for s in screens)
    print(f"Screens found: {len(screens)}  (ON: {n_on}, OFF: {len(screens) - n_on})")
    print(f"Missing slots (from row gaps): {len(missing_slots)}")
    if expected is not None and len(screens) < expected:
        print(f"ANOMALY: expected {expected} screens, only {len(screens)} detected "
              f"-> {expected - len(screens)} missing")
    show(out, "Screen detection (Hough lines)", save)
    return screens, missing_slots


# ----------------------------------------------------------------------------
# Lab-timing Task 2: Asset tracking with SIFT
# ----------------------------------------------------------------------------
def track_assets(scene_path, refs, max_instances=10, save="assets_out.jpg"):
    """
    refs: dict {asset_name: reference_image_path}.
    For each reference, SIFT-match against the lab image, estimate a homography
    with RANSAC, draw the box, then mask the found region and search again so
    several copies of the same asset (e.g. 20 identical monitors) are counted.
    """
    scene = load_image(scene_path)
    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    out = scene.copy()
    inventory = {}
    rng = np.random.default_rng(1)

    for name, path in refs.items():
        ref = load_image(path)
        kp_r, des_r = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        search_mask = np.full(gray_scene.shape, 255, np.uint8)
        count = 0
        for _ in range(max_instances):
            kp_s, des_s = sift.detectAndCompute(gray_scene, search_mask)
            if des_s is None or len(kp_s) < 2:
                break
            good = sift_match(des_r, des_s)
            poly, _ = locate_with_homography(kp_r, kp_s, good, ref.shape)
            if poly is None:
                break
            count += 1
            cv2.polylines(out, [np.int32(poly)], True, color, 3)
            x, y = np.int32(poly[0])
            cv2.putText(out, f"{name} #{count}", (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.fillPoly(search_mask, [np.int32(poly)], 0)       # don't find the same one again
        inventory[name] = count

    print("\n=== Asset inventory ===")
    for k, v in inventory.items():
        print(f"{k:15s}: {v}")
    show(out, "Asset tracking (SIFT)", save)
    return inventory


# ----------------------------------------------------------------------------
# Lab-timing Task 3: Anomaly detection in sensor data (wavelets)
# ----------------------------------------------------------------------------
def wavelet_anomaly_detection(data=None, wavelet="db4", level=4, k=3.0, save="wavelet_out.png"):
    """
    1. Decompose with DWT.  2. Soft-threshold detail coefficients (universal
    threshold, sigma from MAD of finest level).  3. Reconstruct = denoised signal.
    4. residual = data - denoised.  5. Anomaly where |residual| > k * robust sigma.
    """
    import pywt

    if data is None:                                   # sample sensor dataset
        rng = np.random.default_rng(42)
        t = np.arange(1000)
        data = np.sin(2 * np.pi * t / 200) + 0.3 * rng.standard_normal(1000)
        for i in rng.choice(1000, 20, replace=False):  # injected faults
            data[i] += rng.choice([-1, 1]) * rng.uniform(2.0, 3.5)
    data = np.asarray(data, float)

    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet)[: len(data)]

    residual = data - denoised
    res_sigma = np.median(np.abs(residual - np.median(residual))) / 0.6745
    anomalies = np.where(np.abs(residual - np.median(residual)) > k * res_sigma)[0]

    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    ax[0].plot(data, label="Sensor Data")
    ax[0].plot(denoised, "--", label="Denoised Signal")
    ax[0].set_title("Sensor Data and Denoised Signal")
    ax[0].legend(loc="lower right")
    ax[1].plot(residual, "r", label="Residuals")
    ax[1].scatter(anomalies, residual[anomalies], c="green", zorder=3, label="Anomalies")
    ax[1].set_title("Residuals and Detected Anomalies")
    ax[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save, dpi=120)
    plt.show()
    print(f"Detected {len(anomalies)} anomalies at indices: {anomalies.tolist()}")
    return anomalies


# ----------------------------------------------------------------------------
# Lab Task 1: Object recognition in video (SIFT)
# ----------------------------------------------------------------------------
def object_recognition_video(ref_path, video, out_path="objvideo_out.mp4", show_window=False):
    """Detect the reference object in every frame; draw a bounding polygon."""
    ref = load_image(ref_path)
    sift = cv2.SIFT_create()
    kp_r, des_r = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)

    cap = cv2.VideoCapture(int(video) if str(video).isdigit() else video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    found_frames = total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        kp_f, des_f = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        label, color = "Object NOT found", (0, 0, 255)
        if des_f is not None and len(kp_f) > 2:
            good = sift_match(des_r, des_f)
            poly, _ = locate_with_homography(kp_r, kp_f, good, ref.shape)
            if poly is not None:
                found_frames += 1
                cv2.polylines(frame, [np.int32(poly)], True, (0, 255, 0), 3)
                label, color = f"Object found ({len(good)} matches)", (0, 255, 0)
        cv2.putText(frame, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        writer.write(frame)
        if show_window:
            cv2.imshow("Object recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Object detected in {found_frames}/{total} frames. Saved -> {out_path}")


# ----------------------------------------------------------------------------
# Lab Task 2: Panorama stitching (SIFT + RANSAC homography)
# ----------------------------------------------------------------------------
def stitch_pair(base, new):
    """Warp `new` into the coordinate frame of `base` and blend."""
    sift = cv2.SIFT_create()
    kp_b, des_b = sift.detectAndCompute(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), None)
    kp_n, des_n = sift.detectAndCompute(cv2.cvtColor(new, cv2.COLOR_BGR2GRAY), None)
    good = sift_match(des_n, des_b, ratio=0.7)                 # new -> base
    if len(good) < 10:
        raise RuntimeError(f"Not enough matches ({len(good)}) - are the images overlapping?")
    src = np.float32([kp_n[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)

    hb, wb = base.shape[:2]
    hn, wn = new.shape[:2]
    corners_new = cv2.perspectiveTransform(
        np.float32([[0, 0], [0, hn], [wn, hn], [wn, 0]]).reshape(-1, 1, 2), H)
    corners_base = np.float32([[0, 0], [0, hb], [wb, hb], [wb, 0]]).reshape(-1, 1, 2)
    allc = np.concatenate([corners_base, corners_new])
    xmin, ymin = np.int32(allc.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(allc.max(axis=0).ravel() + 0.5)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)

    pano = cv2.warpPerspective(new, T @ H, (xmax - xmin, ymax - ymin))
    base_canvas = np.zeros_like(pano)
    base_canvas[-ymin:hb - ymin, -xmin:wb - xmin] = base
    # simple blend: where both have content average, otherwise take whichever exists
    m_new = pano.sum(axis=2) > 0
    m_base = base_canvas.sum(axis=2) > 0
    both = m_new & m_base
    result = np.where(m_base[..., None], base_canvas, pano)
    result[both] = (0.5 * base_canvas[both] + 0.5 * pano[both]).astype(np.uint8)
    return result


def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect((gray > 5).astype(np.uint8))
    return img[y:y + h, x:x + w]


def make_panorama(paths, save="panorama_out.jpg"):
    imgs = [load_image(p) for p in paths]
    pano = imgs[0]
    for nxt in imgs[1:]:                       # images must be in left-to-right order
        pano = stitch_pair(pano, nxt)
    pano = crop_black(pano)
    print("Panorama size:", pano.shape[1], "x", pano.shape[0])
    show(pano, "Panorama (SIFT stitching)", save)
    return pano


# ----------------------------------------------------------------------------
# Lab Task 3: Lane detection (Canny + ROI + Hough)
# ----------------------------------------------------------------------------
def _lane_lines(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

    roi = np.zeros_like(edges)
    poly = np.array([[(int(0.05 * w), h), (int(0.45 * w), int(0.60 * h)),
                      (int(0.55 * w), int(0.60 * h)), (int(0.95 * w), h)]], np.int32)
    cv2.fillPoly(roi, poly, 255)
    edges = cv2.bitwise_and(edges, roi)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=100)
    left, right = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if abs(slope) < 0.4:                      # ignore near-horizontal clutter
                continue
            (left if slope < 0 else right).append((slope, intercept))

    def extrapolate(group):
        if not group:
            return None
        m, b = np.mean(group, axis=0)
        y1, y2 = h, int(0.62 * h)
        return (int((y1 - b) / m), y1, int((y2 - b) / m), y2)

    return extrapolate(left), extrapolate(right)


def detect_lanes_frame(frame):
    out = frame.copy()
    layer = np.zeros_like(frame)
    left, right = _lane_lines(frame)
    for ln in (left, right):
        if ln:
            cv2.line(layer, ln[:2], ln[2:], (0, 0, 255), 10)
    if left and right:                                 # shade the lane between the lines
        pts = np.array([[left[:2], left[2:], right[2:], right[:2]]], np.int32)
        cv2.fillPoly(layer, pts, (0, 120, 0))
    return cv2.addWeighted(out, 1.0, layer, 0.6, 0)


def lane_detection(image=None, video=None, save="lanes_out.jpg"):
    if image:
        show(detect_lanes_frame(load_image(image)), "Lane detection (Hough)", save)
        return
    cap = cv2.VideoCapture(int(video) if str(video).isdigit() else video)
    w, h = int(cap.get(3)), int(cap.get(4))
    wr = cv2.VideoWriter("lanes_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                         cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        wr.write(detect_lanes_frame(f))
    cap.release()
    wr.release()
    print("Saved -> lanes_out.mp4")


# ----------------------------------------------------------------------------
# Lab Task 4: Coin detection and counting (Hough Circle Transform)
# ----------------------------------------------------------------------------
def count_coins(image_path, save="coins_out.jpg", min_r=15, max_r=120):
    img = load_image(image_path)
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=min_r * 1.6, param1=120, param2=40,
                               minRadius=min_r, maxRadius=max_r)
    out = img.copy()
    n = 0
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        n = len(circles)
        for i, (x, y, r) in enumerate(sorted(circles.tolist(), key=lambda c: (c[1], c[0])), 1):
            cv2.circle(out, (x, y), r, (0, 255, 0), 3)
            cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(out, str(i), (x - 10, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        print("Coin radii (px):", sorted(circles[:, 2].tolist()))
    cv2.putText(out, f"Total coins: {n}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    print(f"Total coins detected: {n}")
    show(out, "Coin detection (Hough circles)", save)
    return n


# ----------------------------------------------------------------------------
# Lab Task 5: Smart security system (boundary detection in a security zone)
# ----------------------------------------------------------------------------
def smart_security(video, zone=None, out_path="security_out.mp4", show_window=False, min_area=1500):
    """
    Background subtraction -> object masks -> contours/boundaries (also Canny for
    display).  If any object's contour overlaps the zone by more than `min_area`
    pixels, raise the alarm: zone turns red, banner shown, console/beep alert.
    """
    cap = cv2.VideoCapture(int(video) if str(video).isdigit() else video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video}")
    w, h = int(cap.get(3)), int(cap.get(4))
    if zone is None:
        zone = (int(0.35 * w), int(0.3 * h), int(0.65 * w), int(0.8 * h))   # x1,y1,x2,y2
    zx1, zy1, zx2, zy2 = zone
    zone_mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(zone_mask, (zx1, zy1), (zx2, zy2), 255, -1)

    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                         cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=True)
    alarm_frames, frame_no = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        fg = bg.apply(frame)
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]          # drop shadows (127)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg = cv2.dilate(fg, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        intruder = False
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            obj = np.zeros((h, w), np.uint8)
            cv2.drawContours(obj, [c], -1, 255, -1)
            inside = cv2.countNonZero(cv2.bitwise_and(obj, zone_mask)) > min_area * 0.2
            cv2.drawContours(frame, [c], -1, (0, 0, 255) if inside else (0, 255, 0), 2)  # boundary
            intruder |= inside

        zone_col = (0, 0, 255) if intruder else (255, 200, 0)
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), zone_col, 3)
        if intruder and frame_no > 30:                  # ignore first frames (bg learning)
            alarm_frames += 1
            cv2.putText(frame, "ALARM: UNAUTHORIZED OBJECT IN ZONE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            if alarm_frames == 1:
                print(f"[frame {frame_no}] ALARM TRIGGERED!\a")
        wr.write(frame)
        if show_window:
            cv2.imshow("Smart Security", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    wr.release()
    cv2.destroyAllWindows()
    print(f"Done. Alarm active in {alarm_frames} frames. Saved -> {out_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Lab 06 solutions")
    sp = p.add_subparsers(dest="task", required=True)

    a = sp.add_parser("screens"); a.add_argument("--image", required=True)
    a.add_argument("--expected", type=int)
    a = sp.add_parser("assets"); a.add_argument("--image", required=True)
    a.add_argument("--refs", nargs="+", required=True, help="name=path ...")
    a = sp.add_parser("wavelet"); a.add_argument("--csv")
    a = sp.add_parser("objvideo"); a.add_argument("--ref", required=True)
    a.add_argument("--video", required=True); a.add_argument("--show", action="store_true")
    a = sp.add_parser("panorama"); a.add_argument("--images", nargs="+", required=True)
    a = sp.add_parser("lanes"); a.add_argument("--image"); a.add_argument("--video")
    a = sp.add_parser("coins"); a.add_argument("--image", required=True)
    a = sp.add_parser("security"); a.add_argument("--video", required=True)
    a.add_argument("--zone", nargs=4, type=int); a.add_argument("--show", action="store_true")

    args = p.parse_args()
    if args.task == "screens":
        detect_screens(args.image, args.expected)
    elif args.task == "assets":
        refs = dict(r.split("=", 1) for r in args.refs)
        track_assets(args.image, refs)
    elif args.task == "wavelet":
        data = np.loadtxt(args.csv, delimiter=",").ravel() if args.csv else None
        wavelet_anomaly_detection(data)
    elif args.task == "objvideo":
        object_recognition_video(args.ref, args.video, show_window=args.show)
    elif args.task == "panorama":
        make_panorama(args.images)
    elif args.task == "lanes":
        lane_detection(args.image, args.video)
    elif args.task == "coins":
        count_coins(args.image)
    elif args.task == "security":
        smart_security(args.video, tuple(args.zone) if args.zone else None, show_window=args.show)


if __name__ == "__main__":
    main()
