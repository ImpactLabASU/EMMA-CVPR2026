
try:
    import os as _os
    if __name__ == "__main__":
        # why: don't suppress in-file mains by default; what: default to 0 unless user overrides
        _os.environ.setdefault("EMMA_AUTOMATE_MODE", "0")
except Exception:
    pass

# droneExtract.py - extracts the x, y, z coordinates of the drone

# Standard library imports
import os, sys, shutil, subprocess, threading, time, uuid, webbrowser
import numpy as np
import cv2


import argparse, os, csv, time, math
import numpy as np
import cv2

# Option A: Ultralytics YOLO (recommended if you have a .pt/.onnx drone model)
try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False

# --------------------------- Kalman (x,y,z,vx,vy,vz) ---------------------------

class Kalman3D:
    def __init__(self, dt=1/30.0, process_var=5.0, meas_var_xy=25.0, meas_var_z=0.5):
        # State: [x, y, z, vx, vy, vz]
        self.dt = dt
        self.x = np.zeros((6, 1), dtype=float)
        self.P = np.eye(6, dtype=float) * 1000.0

        self.F = np.eye(6, dtype=float)
        for i in range(3):
            self.F[i, i+3] = dt

        self.H = np.zeros((3, 6), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        q = process_var
        self.Q = np.eye(6, dtype=float) * q
        self.R = np.diag([meas_var_xy, meas_var_xy, meas_var_z]).astype(float)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        # z: shape (3,), values [x, y, z]
        z = np.asarray(z, dtype=float).reshape(3, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()

# --------------------------- Depth estimation helpers --------------------------

class DepthEstimator:
    """
    Two modes:
      - 'size': Z = focal_px * real_width_m / bbox_width_px   (needs focal_px & real_width_m)
                 If missing, returns ~1/width (relative depth proxy).
      - 'midas': relative depth via ONNX MiDaS. Provide MIDAS_ONNX=/path/to/midas_small.onnx
                 Returns median depth over bbox (relative units).
    """
    def __init__(self, mode, focal_px=None, real_width_m=None):
        self.mode = mode
        self.focal_px = focal_px
        self.real_width_m = real_width_m

        self.net = None
        if mode == 'midas':
            onnx_path = os.getenv('MIDAS_ONNX', 'midas_small.onnx')
            if os.path.exists(onnx_path):
                self.net = cv2.dnn.readNet(onnx_path)
            else:
                print(f"[WARN] MiDaS ONNX not found at {onnx_path} - Z will be None. "
                      f"Set MIDAS_ONNX env var or place file in working dir.")

    def z_from_bbox(self, frame, bbox_xyxy):
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        if self.mode == 'size':
            if self.focal_px and self.real_width_m:
                return float(self.focal_px * self.real_width_m / w)
            # Fallback: relative inverse-width
            return float(1.0 / w)

        if self.mode == 'midas' and self.net is not None:
            if frame is None or frame.size == 0:
                return None
            # Whole-frame depth (small model for speed), then crop region median
            inp = cv2.resize(frame, (256, 256))
            blob = cv2.dnn.blobFromImage(
                inp, 1/255.0, (256, 256),
                mean=(0.485, 0.456, 0.406), swapRB=True, crop=False
            )
            self.net.setInput(blob)
            depth = self.net.forward().squeeze()
            depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
            region = depth[y1:y2, x1:x2]
            if region.size == 0:
                return None
            return float(np.median(region))

        return None


# --------------------------- Video Stabilizer (translation) --------------------

class VideoStabilizer:
    def __init__(self, smoothing_radius=30):
        self.smoothing_radius = smoothing_radius
        self.prev_gray = None
        self.trajectory = []  # list of (dx, dy)
        self.transform = None

    def stabilize_frame(self, frame):
        """Stabilize frame by compensating translation using feature tracking."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.transform = np.eye(2, 3, dtype=np.float32)
            return frame, self.transform

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
        )

        if prev_pts is None:
            self.transform = np.eye(2, 3, dtype=np.float32)
            self.prev_gray = gray
            return frame, self.transform

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        idx = np.where(status == 1)[0]
        if len(idx) < 10:
            self.transform = np.eye(2, 3, dtype=np.float32)
            self.prev_gray = gray
            return frame, self.transform

        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        M, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        if M is None:
            self.transform = np.eye(2, 3, dtype=np.float32)
            self.prev_gray = gray
            return frame, self.transform

        dx, dy = float(M[0, 2]), float(M[1, 2])
        self.trajectory.append((dx, dy))
        if len(self.trajectory) > self.smoothing_radius:
            self.trajectory.pop(0)

        smooth_dx = np.mean([t[0] for t in self.trajectory])
        smooth_dy = np.mean([t[1] for t in self.trajectory])

        # Compensate difference
        diff_x = smooth_dx - dx
        diff_y = smooth_dy - dy
        self.transform = np.array([[1, 0, -diff_x], [0, 1, -diff_y]], dtype=np.float32)

        stabilized = cv2.warpAffine(frame, self.transform, (frame.shape[1], frame.shape[0]))
        self.prev_gray = gray
        return stabilized, self.transform


# --------------------------- Template Tracker -------------------------------

class TemplateTracker:
    def __init__(self, template_size=(64, 64), match_threshold=0.6):
        self.template = None
        self.template_size = template_size
        self.last_center = None
        self.tracking_lost = True
        self.match_threshold = match_threshold
        self.scale_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
        self.scale_history = []

    def init_template(self, frame, bbox_xyxy):
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            self.tracking_lost = True
            return False
        self.template = cv2.resize(crop, self.template_size)
        self.last_center = ( (x1 + x2)//2, (y1 + y2)//2 )
        self.tracking_lost = False
        self.scale_history.clear()
        return True

    def track(self, frame):
        if self.template is None or self.tracking_lost or self.last_center is None:
            return None

        h, w = frame.shape[:2]
        cx, cy = self.last_center
        search_radius = 300
        x1 = max(0, cx - search_radius); y1 = max(0, cy - search_radius)
        x2 = min(w, cx + search_radius); y2 = min(h, cy + search_radius)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            self.tracking_lost = True
            return None

        best_val, best_loc, best_scale = -1.0, None, 1.0
        for s in self.scale_factors:
            tw = int(self.template_size[0] * s)
            th = int(self.template_size[1] * s)
            if tw < 10 or th < 10 or tw > region.shape[1] or th > region.shape[0]:
                continue
            tpl = cv2.resize(self.template, (tw, th))
            res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val; best_loc = max_loc; best_scale = s

        if best_val < self.match_threshold or best_loc is None:
            self.tracking_lost = True
            return None

        # Compute bbox in full-frame coords
        mx, my = best_loc
        mx += x1; my += y1
        tw = int(self.template_size[0] * best_scale)
        th = int(self.template_size[1] * best_scale)
        cx_new = mx + tw//2
        cy_new = my + th//2
        self.last_center = (cx_new, cy_new)

        # Smooth scale
        self.scale_history.append(best_scale)
        if len(self.scale_history) > 5:
            self.scale_history.pop(0)
        smooth_scale = float(np.mean(self.scale_history))

        tw = int(self.template_size[0] * smooth_scale)
        th = int(self.template_size[1] * smooth_scale)
        x1b, y1b = int(cx_new - tw//2), int(cy_new - th//2)
        x2b, y2b = x1b + tw, y1b + th
        return (x1b, y1b, x2b, y2b), float(best_val)

    def update_template(self, frame, bbox_xyxy, variance_thresh=0.1):
        if len(self.scale_history) < 3:
            return
        if np.var(self.scale_history[-3:]) > variance_thresh:
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                self.template = cv2.resize(crop, self.template_size)


# --------------------------- Feature Tracker (ORB) ----------------------------

class FeatureTracker:
    def __init__(self):
        self.prev_kp = None
        self.prev_desc = None
        self.bbox = None
        self.tracking_lost = True
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def init_features(self, frame, bbox_xyxy):
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            self.tracking_lost = True
            return False
        kp, desc = self.orb.detectAndCompute(roi, None)
        if desc is None or len(kp) < 10:
            self.tracking_lost = True
            return False
        # shift kp to full-frame coords
        for k in kp:
            k.pt = (k.pt[0] + x1, k.pt[1] + y1)
        self.prev_kp = kp
        self.prev_desc = desc
        self.bbox = (x1, y1, x2, y2)
        self.tracking_lost = False
        return True

    def track(self, frame):
        if self.prev_kp is None or self.tracking_lost:
            return None
        kp, desc = self.orb.detectAndCompute(frame, None)
        if desc is None or len(kp) < 10:
            self.tracking_lost = True
            return None
        matches = self.matcher.match(self.prev_desc, desc)
        matches = sorted(matches, key=lambda m: m.distance)
        if len(matches) < 5:
            self.tracking_lost = True
            return None

        src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None:
            self.tracking_lost = True
            return None

        x1, y1, x2, y2 = self.bbox
        corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
        transformed = cv2.transform(corners, M)[0]
        nx1 = int(np.min(transformed[:, 0])); ny1 = int(np.min(transformed[:, 1]))
        nx2 = int(np.max(transformed[:, 0])); ny2 = int(np.max(transformed[:, 1]))
        self.bbox = (nx1, ny1, nx2, ny2)

        # Keep only inlier matches for next round
        inliers = (mask.ravel() == 1)
        in_kp_idx = [matches[i].trainIdx for i in range(len(matches)) if inliers[i]]
        if len(in_kp_idx) >= 5:
            self.prev_kp = [kp[i] for i in in_kp_idx]
            self.prev_desc = desc[in_kp_idx]
            self.tracking_lost = False
            # Confidence proxy: inlier ratio
            conf = float(np.mean(inliers))
            return (nx1, ny1, nx2, ny2), conf

        self.tracking_lost = True
        return None


# --------------------------- Drone Detector/Tracker ---------------------------

# ---------- Drone Detector wrapper ---------------------------
#class DroneDetector:
#    def __init__(self, weights, conf=0.25, cls_names=('drone',)):
#        self.cls_names = set([c.lower() for c in cls_names])
#        self.conf = conf
#        self.is_yolo = False
#        self.model = None

#        if _HAS_ULTRALYTICS and weights is not None:
#            self.model = YOLO(weights)
#            self.is_yolo = True
#        else:
#            raise RuntimeError("Ultralytics not available or weights not provided. Install `ultralytics` and provide a drone model.")

#    def detect(self, frame):
#        """Return best drone bbox (x1,y1,x2,y2,conf) or None."""
#        if self.is_yolo:
#            res = self.model.predict(source=frame, conf=self.conf, verbose=False)
#            best = None
#            for r in res:
#                boxes = r.boxes
#                if boxes is None:
#                    continue
#                for b in boxes:
#                    cls_id = int(b.cls[0].item()) if b.cls is not None else None
#                    name = self.model.names.get(cls_id, '').lower() if cls_id is not None else ''
#                    # If your model has a single class (drone), name check can be skipped
#                    if (len(self.cls_names)==0) or (name in self.cls_names) or (len(self.model.names)==1):
#                        conf = float(b.conf[0].item()) if b.conf is not None else 0.0
#                        if best is None or conf > best[-1]:
#                            x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
#                            best = (x1,y1,x2,y2,conf)
#            return best
#        return None

class DroneDetector:
    def __init__(self, weights, conf=0.25, imgsz=1280, roi_expansion=200):
        self.conf = conf
        self.imgsz = imgsz
        self.is_yolo = False
        self.model = None

        self.last_detection = None     # (x1,y1,x2,y2,conf)
        self.detection_history = []    # recent detections for smoothing
        self.max_history = 5
        self.expected_position = None
        self.velocity = None
        self.roi = None
        self.roi_expansion = roi_expansion
        self.tracking_mode = "detection"  # "detection" | "template" | "feature"
        self.frames_since_detection = 0
        self.max_tracking_frames = 60
        self.frame_count = 0

        # stabilizer and trackers
        self.stabilizer = VideoStabilizer()
        self.template_tracker = TemplateTracker()
        self.feature_tracker = FeatureTracker()

        # YOLO
        if _HAS_ULTRALYTICS and weights:
            self.model = YOLO(weights)
            self.is_yolo = True
            print(f"[INFO] Loaded YOLO weights: {weights}")
        else:
            raise RuntimeError("Ultralytics not available or weights not provided.")

        # cache current frame size each call
        self.w = None
        self.h = None

    def _is_in_roi(self, bbox):
        if self.roi is None:
            return True
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        rx1, ry1, rx2, ry2 = self.roi
        return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

    def _update_roi(self, det):
        if det is None or self.w is None or self.h is None:
            return
        x1, y1, x2, y2, _ = det
        new_x1 = max(0, int(x1) - self.roi_expansion)
        new_y1 = max(0, int(y1) - self.roi_expansion)
        new_x2 = min(self.w, int(x2) + self.roi_expansion)
        new_y2 = min(self.h, int(y2) + self.roi_expansion)

        if self.roi is None:
            self.roi = (new_x1, new_y1, new_x2, new_y2)
            return

        ox1, oy1, ox2, oy2 = self.roi
        rx1 = int(0.7 * ox1 + 0.3 * new_x1)
        ry1 = int(0.7 * oy1 + 0.3 * new_y1)
        rx2 = int(0.7 * ox2 + 0.3 * new_x2)
        ry2 = int(0.7 * oy2 + 0.3 * new_y2)

        # enforce minimum ROI size
        min_roi = int(0.2 * min(self.w, self.h))
        if (rx2 - rx1) < min_roi:
            cx = (rx1 + rx2) // 2
            rx1 = max(0, cx - min_roi // 2); rx2 = min(self.w, cx + min_roi // 2)
        if (ry2 - ry1) < min_roi:
            cy = (ry1 + ry2) // 2
            ry1 = max(0, cy - min_roi // 2); ry2 = min(self.h, cy + min_roi // 2)

        self.roi = (rx1, ry1, rx2, ry2)

    def _smooth_detection(self, det):
        if det is None:
            return None
        x1, y1, x2, y2, conf = det
        self.detection_history.append((x1, y1, x2, y2, conf))
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        if len(self.detection_history) < 3:
            return det

        weights = np.linspace(0.3, 1.0, len(self.detection_history))
        weights = weights / weights.sum()
        sx1 = sum(d[0] * w for d, w in zip(self.detection_history, weights))
        sy1 = sum(d[1] * w for d, w in zip(self.detection_history, weights))
        sx2 = sum(d[2] * w for d, w in zip(self.detection_history, weights))
        sy2 = sum(d[3] * w for d, w in zip(self.detection_history, weights))
        sc  = sum(d[4] * w for d, w in zip(self.detection_history, weights))
        return (sx1, sy1, sx2, sy2, sc)

    def _update_motion_model(self, det):
        if det is None:
            return
        x1, y1, x2, y2, _ = det
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if self.last_detection is not None:
            px1, py1, px2, py2, _ = self.last_detection
            pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            self.velocity = (cx - pcx, cy - pcy)
            self.expected_position = (cx + self.velocity[0], cy + self.velocity[1])
        else:
            self.velocity = (0.0, 0.0)
            self.expected_position = (cx, cy)
        self._update_roi(det)

    def _run_detection(self, frame):
        """Run YOLO and pick best candidate with dynamic size/edge rules."""
        h, w = frame.shape[:2]
        img_area = w * h
        edge_thresh = int(0.02 * min(w, h))  # 2% border
        min_area_px = max(200, int(0.00005 * img_area))
        max_area_px = int(0.5 * img_area)

        # Run model - detects ANY object type (car, truck, bus, person, etc.)
        results = self.model.predict(
            source=frame, imgsz=self.imgsz, conf=self.conf,
            iou=0.6, agnostic_nms=True, verbose=False
        )

        # Debug: raw boxes count
        raw = sum(0 if r.boxes is None else len(r.boxes) for r in results)
        # print(f"[DEBUG] YOLO raw boxes: {raw} @ frame {self.frame_count}")

        candidates = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                bw, bh = x2 - x1, y2 - y1
                area = bw * bh
                if area < min_area_px or area > max_area_px:
                    continue
                # edge guard
                if (x1 < edge_thresh or y1 < edge_thresh or
                    x2 > w - edge_thresh or y2 > h - edge_thresh):
                    continue
                conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                roi_bonus = 0.1 if self._is_in_roi((x1, y1, x2, y2)) else 0.0

                # motion consistency (optional lightweight)
                cons = 0.0
                if self.last_detection is not None:
                    px1, py1, px2, py2, _ = self.last_detection
                    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    dist = math.hypot(cx - pcx, cy - pcy)
                    cons = max(0.0, 1.0 - dist / 50.0)  # decay with distance

                score = conf + 0.4 * cons + roi_bonus
                candidates.append((x1, y1, x2, y2, conf, score))

        if not candidates:
            return None
        x1, y1, x2, y2, conf, _ = max(candidates, key=lambda t: t[5])
        return (x1, y1, x2, y2, conf)

    def _initialize_trackers(self, frame, det):
        if det is None:
            return
        self.template_tracker.init_template(frame, det[:4])
        self.feature_tracker.init_features(frame, det[:4])
        self._update_roi(det)

    def detect(self, frame):
        """Main per-frame strategy: template → feature → detection fallback."""
        self.h, self.w = frame.shape[:2]
        self.frame_count += 1
        self.frames_since_detection += 1

        # Try template tracking first
        if self.tracking_mode == "template" and self.template_tracker.template is not None:
            out = self.template_tracker.track(frame)
            if out is not None:
                bbox, t_conf = out
                self.template_tracker.update_template(frame, bbox)
                det = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(t_conf))
                self.last_detection = det
                self.frames_since_detection = 0
                return det
            else:
                self.tracking_mode = "feature"

        # Try feature tracking
        if self.tracking_mode == "feature" and not self.feature_tracker.tracking_lost:
            out = self.feature_tracker.track(frame)
            if out is not None:
                bbox, f_conf = out
                det = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(f_conf))
                self.last_detection = det
                self.frames_since_detection = 0
                return det
            else:
                self.tracking_mode = "detection"

        # Fallback: YOLO detection
        det = self._run_detection(frame)
        if det is not None:
            det = self._smooth_detection(det)
            self._initialize_trackers(frame, det)
            self.last_detection = det
            self.frames_since_detection = 0
            self._update_motion_model(det)
            # choose next mode
            if det[4] >= 0.3:
                self.tracking_mode = "template"
            elif det[4] >= 0.2:
                self.tracking_mode = "feature"
            else:
                self.tracking_mode = "detection"
            return det

        # If nothing found, hold last_detection (or None)
        return self.last_detection

# ---------- Main pipeline -----------------------------------
def main():
    ap = argparse.ArgumentParser(description="Drone detect + center keypoint + track x,y,z")
    # why: allow zero-arg run using repo defaults; what: default paths relative to this file
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(_here)
    _root = os.path.dirname(_repo)
    ap.add_argument("--video", default=os.path.join(_here, "DroneVideo.mp4"), help="input video path")
    ap.add_argument("--weights", default=os.path.join(_root, "yolov8n.pt"), help="YOLO weights (.pt/.onnx) trained for drones")
    ap.add_argument("--out_video", default=os.path.join(_here, "annotated.mp4"), help="output annotated video")
    ap.add_argument("--out_csv", default=os.path.join(_here, "data", "trajectory.csv"), help="output CSV of frame,time_s,x,y,z")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence")
    ap.add_argument("--depth_mode", choices=["size","midas"], default="size", help="Z estimation mode")
    ap.add_argument("--focal_px", type=float, default=None, help="Camera focal length in pixels (size mode)")
    ap.add_argument("--real_width_m", type=float, default=None, help="Real drone width in meters (size mode)")
    ap.add_argument("--kalman_dt", type=float, default=None, help="Override KF dt (default: from FPS)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3: fps = 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))

    detector = DroneDetector(weights=args.weights, conf=args.conf, imgsz=1280)
    depth_est = None
    if args.depth_mode != "none":
        depth_est = DepthEstimator(args.depth_mode, focal_px=args.focal_px, real_width_m=args.real_width_m)

    kf = Kalman3D(dt=(args.kalman_dt if args.kalman_dt else 1.0 / fps))

    # Prepare CSV
    out_dir = os.path.dirname(os.path.abspath(args.out_csv)) or "."
    os.makedirs(out_dir, exist_ok=True)
    csv_f = open(args.out_csv, "w", newline="")
    csvw = csv.writer(csv_f)
    csvw.writerow(["frame", "time_s", "x", "y", "z", "conf"])

    x_series, y_series, z_series = [], [], []

    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_time = frame_idx / fps

        # 1) Stabilize frame (work in stabilized space for detection/tracking)
        stabilized, T = detector.stabilizer.stabilize_frame(frame)

        # 2) Detect/track in stabilized frame
        det = detector.detect(stabilized)

        if det is not None:
            x1, y1, x2, y2, conf = det
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # depth based on stabilized frame (bbox coords match)
            if depth_est is not None:
                z = depth_est.z_from_bbox(stabilized, (x1, y1, x2, y2))
                if z is None or math.isnan(z) or math.isinf(z):
                    z = 0.0
            else:
                z = 0.0

            # 3) Kalman smoothing
            kf.predict()
            xs = kf.update(np.array([cx, cy, z], dtype=float)).squeeze()
            xk, yk, zk = float(xs[0]), float(xs[1]), float(xs[2])
            x_series.append(xk); y_series.append(yk); z_series.append(zk)

            # 4) Map stabilized coords back to original frame for drawing
            if T is not None and not np.array_equal(T, np.eye(2, 3)):
                T_inv = cv2.invertAffineTransform(T)
                pts = np.array([[[x1, y1], [x2, y2], [xk, yk]]], dtype=np.float32)
                pts_orig = cv2.transform(pts, T_inv)[0]
                ox1, oy1 = pts_orig[0]; ox2, oy2 = pts_orig[1]; oxk, oyk = pts_orig[2]
            else:
                ox1, oy1, ox2, oy2 = x1, y1, x2, y2
                oxk, oyk = xk, yk

            # 5) Draw on the original frame
            cv2.rectangle(frame, (int(ox1), int(oy1)), (int(ox2), int(oy2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(oxk), int(oyk)), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"x={oxk:.1f}, y={oyk:.1f}, z={zk:.3f}, conf={conf:.2f}",
                        (int(max(0, ox1)), max(20, int(oy1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # 6) Optional: draw ROI (transform back too)
            if detector.roi is not None:
                rx1, ry1, rx2, ry2 = detector.roi
                if T is not None and not np.array_equal(T, np.eye(2, 3)):
                    T_inv = cv2.invertAffineTransform(T)
                    roi_pts = np.array([[[rx1, ry1], [rx2, ry2]]], dtype=np.float32)
                    roi_orig = cv2.transform(roi_pts, T_inv)[0]
                    orx1, ory1 = roi_orig[0]; orx2, ory2 = roi_orig[1]
                    cv2.rectangle(frame, (int(orx1), int(ory1)), (int(orx2), int(ory2)), (255, 255, 0), 1)
                else:
                    cv2.rectangle(frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255, 255, 0), 1)

            csvw.writerow([frame_idx, f"{frame_time:.3f}", f"{xk:.2f}", f"{yk:.2f}", f"{zk:.5f}", f"{conf:.3f}"])

        else:
            # No detection: predict forward and draw predicted point
            xs = kf.predict().squeeze()
            xk, yk, zk = float(xs[0]), float(xs[1]), float(xs[2])
            x_series.append(xk); y_series.append(yk); z_series.append(zk)
            # Map to original space
            if T is not None and not np.array_equal(T, np.eye(2, 3)):
                T_inv = cv2.invertAffineTransform(T)
                pt = np.array([[[xk, yk]]], dtype=np.float32)
                pt_orig = cv2.transform(pt, T_inv)[0][0]
                oxk, oyk = pt_orig
            else:
                oxk, oyk = xk, yk
            cv2.circle(frame, (int(oxk), int(oyk)), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"(pred) x={oxk:.1f}, y={oyk:.1f}, z={zk:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2, cv2.LINE_AA)
            csvw.writerow([frame_idx, f"{frame_time:.3f}", f"{xk:.2f}", f"{yk:.2f}", f"{zk:.5f}", "NaN"])

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[PROGRESS] Processed {frame_idx} frames")

    # Cleanup
    writer.release()
    cap.release()
    csv_f.close()

    # Export only transposed (N x 1701) matrices under canonical names xData/yData/zData
    # why: caller expects trainSim-like orientation; what: write M.T as xData.txt etc.
    try:
        out_dir = os.path.dirname(os.path.abspath(args.out_csv)) or "."
        # why: ensure directory still exists before saving matrices; what: create if missing
        os.makedirs(out_dir, exist_ok=True)
        if len(x_series) > 0:
            row_x = np.asarray(x_series, dtype=float)
            row_y = np.asarray(y_series, dtype=float)
            row_z = np.asarray(z_series, dtype=float)
            Mx = np.tile(row_x, (1701,1))
            My = np.tile(row_y, (1701,1))
            Mz = np.tile(row_z, (1701,1))
            np.savetxt(os.path.join(out_dir, "xData.txt"), Mx.T, fmt='%.6f')
            np.savetxt(os.path.join(out_dir, "yData.txt"), My.T, fmt='%.6f')
            np.savetxt(os.path.join(out_dir, "zData.txt"), Mz.T, fmt='%.6f')
            # optional: cleanup old files if present
            for obsolete in ("xData_T.txt","yData_T.txt","zData_T.txt"):
                try:
                    os.remove(os.path.join(out_dir, obsolete))
                except Exception:
                    pass
    except Exception as e:
        print("[WARN] Failed to write x/y/z data:", e)
    print(f"[DONE] Wrote: {args.out_video} and {args.out_csv}")

if __name__ == "__main__" and os.getenv("EMMA_AUTOMATE_MODE") != "1" and os.getenv("EMMA_ENABLE_VISION", "0") == "1":
    main()

#video code logic ends here 

# droneExtractAudio.py - extracts the audio features of the drone

import argparse, os, csv, time, math
import numpy as np
import cv2

# YOLO
from ultralytics import YOLO

# Audio utils (handle MoviePy 1.x and 2.x)
try:
    from moviepy.editor import VideoFileClip
except Exception:
    from moviepy import VideoFileClip
import librosa
import soundfile as sf

# ------------ Kalman for (x,y,z) ------------
class Kalman3D:
    def __init__(self, dt=1/30.0, process_var=5.0, meas_var_xy=25.0, meas_var_z=0.5):
        self.dt = dt
        self.x = np.zeros((6,1))
        self.P = np.eye(6)*1000.0
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i+3] = dt
        self.H = np.zeros((3,6))
        self.H[0,0]=self.H[1,1]=self.H[2,2]=1.0
        q=process_var
        self.Q=np.eye(6)*q
        self.R=np.diag([meas_var_xy,meas_var_xy,meas_var_z])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        z = z.reshape(3,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()

# ------------ Depth (simple) ------------
class DepthEstimator:
    """Size-based proxy: Z ~ f * W_real / w_pixels; if missing, uses 1/w_pixels."""
    def __init__(self, focal_px=None, real_width_m=None):
        self.focal_px=focal_px; self.real_width_m=real_width_m
    def z_from_bbox(self, bbox):
        x1,y1,x2,y2=[int(v) for v in bbox]
        w = max(1, x2-x1)
        if self.focal_px and self.real_width_m:
            return float(self.focal_px*self.real_width_m/w)
        return float(1.0/w)


# ------------ Audio feature extractor ------------
class AudioRotorFeatures:
    """
    Computes short-time audio features aligned to video timeline:
      - RMS energy (proxy for rotor intensity)
      - Spectral centroid (Hz)
      - Strongest tonal peak in 80..3000 Hz (Hz)
      - Estimated RPM = peak_hz * 60 / blade_count (if provided)
    """
    def __init__(self, wav_path=None, sr_target=22050, blade_count=None):
        self.wav_path = wav_path
        self.sr_target = sr_target
        self.blade_count = blade_count
        self.t = None
        self.rms = None
        self.centroid_hz = None
        self.peak_hz = None
        self.est_rpm = None

    @staticmethod
    def extract_wav_from_video(video_path, out_wav, fps=44100):
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            raise RuntimeError("No audio track in video.")
        # MoviePy 2.x: verbose/logger args removed
        clip.audio.write_audiofile(out_wav, fps=fps)
        return out_wav

    def compute(self, wav_path):
        y, sr = librosa.load(wav_path, sr=self.sr_target, mono=True)
        # STFT
        n_fft = 2048
        hop_length = 512
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2  # power
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # RMS (time-domain)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length).flatten()

        # Spectral centroid (weighted average frequency)
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr).flatten()

        # Strongest tonal peak (within rotor band 80..3000 Hz)
        lo = np.searchsorted(freqs, 80)
        hi = np.searchsorted(freqs, 3000)
        band = S[lo:hi, :]
        peak_bin = np.argmax(band, axis=0)
        peak_freq = freqs[lo + peak_bin]

        # Estimated RPM if blade_count known: rpm ≈ peak_freq * 60 / blades
        if self.blade_count and self.blade_count > 0:
            est_rpm = (peak_freq * 60.0) / float(self.blade_count)
        else:
            est_rpm = np.full_like(peak_freq, np.nan, dtype=float)

        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        self.t = times
        self.rms = rms
        self.centroid_hz = centroid
        self.peak_hz = peak_freq
        self.est_rpm = est_rpm

    def value_at_time(self, t_sec):
        """Linear interpolate features at time t_sec."""
        if self.t is None:
            return (np.nan, np.nan, np.nan, np.nan)
        def interp(arr):
            return float(np.interp(t_sec, self.t, arr))
        return (
            interp(self.rms),
            interp(self.centroid_hz),
            interp(self.peak_hz),
            interp(self.est_rpm),
        )

def main():
    ap = argparse.ArgumentParser(description="Drone detect+track with audio rotor intensity")
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_video", default="annotated.mp4")
    ap.add_argument("--out_csv", default="trajectory_with_audio.csv")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--focal_px", type=float, default=None)
    ap.add_argument("--real_width_m", type=float, default=None)
    ap.add_argument("--blade_count", type=int, default=None, help="propeller blade count (for RPM)")
    args = ap.parse_args()

    # Video IO
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))

    # Models
    detector = DroneDetector(weights=args.weights, conf=args.conf, imgsz=args.imgsz)
    depth_est = DepthEstimator(focal_px=args.focal_px, real_width_m=args.real_width_m)
    kf = Kalman3D(dt=1.0/float(fps))

    # Audio pipeline
    tmp_wav = "tmp_audio.wav"
    try:
        AudioRotorFeatures.extract_wav_from_video(args.video, tmp_wav, fps=44100)
        arf = AudioRotorFeatures(tmp_wav, sr_target=22050, blade_count=args.blade_count)
        arf.compute(tmp_wav)
        have_audio = True
    except Exception as e:
        print(f"[WARN] Audio extraction/processing failed: {e}")
        have_audio = False
        arf = None

    # CSV
    with open(args.out_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame","time_s","x","y","z","conf","audio_rms","centroid_hz","peak_hz","est_rpm"])

        frame_idx = 0
        peak_series = []  # why: build uMotor from audio peak; what: collect peak_hz per frame
        while True:
            ok, frame = cap.read()
            if not ok: break
            t = frame_idx / fps

            det = detector.detect(frame)
            if det is not None:
                x1,y1,x2,y2,conf = det
                cx = (x1+x2)/2.0; cy=(y1+y2)/2.0
                z = depth_est.z_from_bbox((x1,y1,x2,y2))
                kf.predict(); kf.update(np.array([cx, cy, z], dtype=float))
                xs = kf.x.squeeze(); xk, yk, zk = float(xs[0]), float(xs[1]), float(xs[2])

                # Draw
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                cv2.circle(frame, (int(xk), int(yk)), 5, (0,0,255), -1)
                overlay = f"x={xk:.1f} y={yk:.1f} z={zk:.3f} conf={conf:.2f}"
                cv2.putText(frame, overlay, (int(x1), max(20,int(y1)-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

                # Audio features aligned to this frame time
                if have_audio:
                    rms, cen, peak, rpm = arf.value_at_time(t)
                else:
                    rms=cen=peak=rpm=np.nan
                peak_series.append(float(peak) if np.isfinite(peak) else np.nan)

                # Write CSV
                wcsv.writerow([frame_idx, f"{t:.3f}", f"{xk:.2f}", f"{yk:.2f}", f"{zk:.5f}", f"{conf:.3f}",
                               f"{rms:.6f}", f"{cen:.2f}", f"{peak:.2f}", f"{rpm:.2f}"])

                # Show audio on frame (optional)
                if have_audio:
                    cv2.putText(frame, f"RMS:{rms:.4f}  Cent:{cen:.0f}Hz  Peak:{peak:.0f}Hz  RPM:{rpm:.0f}",
                                (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            else:
                # Predict only
                xs = kf.predict().squeeze()
                xk, yk, zk = float(xs[0]), float(xs[1]), float(xs[2])
                cv2.circle(frame, (int(xk), int(yk)), 5, (0,255,255), -1)
                if have_audio:
                    rms, cen, peak, rpm = arf.value_at_time(t)
                else:
                    rms=cen=peak=rpm=np.nan
                peak_series.append(float(peak) if np.isfinite(peak) else np.nan)
                wcsv.writerow([frame_idx, f"{t:.3f}", f"{xk:.2f}", f"{yk:.2f}", f"{zk:.5f}", "NaN",
                               f"{rms:.6f}", f"{cen:.2f}", f"{peak:.2f}", f"{rpm:.2f}"])
                cv2.putText(frame, "(pred only)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1

    writer.release(); cap.release()
    if os.path.exists(tmp_wav):
        try: os.remove(tmp_wav)
        except: pass
    print(f"[DONE] Saved: {args.out_video} and {args.out_csv}")

    # ---------- Generate uMotor1..4 from scaled peak_hz ----------
    # why: normalize rotor tone to base 75 at smallest peak; what: scale peaks by (75/min_peak)
    try:
        # why: keep motor files alongside x/y/z; what: save into Dronepipeline/data
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(out_dir, exist_ok=True)
        if len(peak_series) > 0:
            arr = np.asarray(peak_series, dtype=float)
            # find smallest positive finite
            valid = arr[np.isfinite(arr) & (arr > 0)]
            if valid.size == 0:
                raise RuntimeError("No valid peak_hz values to scale")
            min_peak = float(np.min(valid))
            scale = 75.0 / min_peak
            scaled = arr * scale
            # fill invalid with 75 (scaled baseline)
            scaled[~(np.isfinite(arr) & (arr > 0))] = 75.0
            # shape: N x 1701 (transpose style like xData)
            M = np.tile(scaled.reshape(-1), (1701, 1)).T
            np.savetxt(os.path.join(out_dir, "uMotor1.txt"), M, fmt='%.6f')
            np.savetxt(os.path.join(out_dir, "uMotor2.txt"), M, fmt='%.6f')
            np.savetxt(os.path.join(out_dir, "uMotor3.txt"), M, fmt='%.6f')
            np.savetxt(os.path.join(out_dir, "uMotor4.txt"), M, fmt='%.6f')
    except Exception as e:
        print("[WARN] Failed to write uMotor files:", e)
    
if __name__ == "__main__" and os.getenv("EMMA_AUTOMATE_MODE") != "1" and os.getenv("EMMA_ENABLE_VISION", "0") == "1":
    main()

#EMMA optimized - EMMA_drone_torch_ltc_optimized.py Gives the theta coefficients and gif output of simulation






# Standard library imports
import os
import csv
import argparse

# Scientific computing and deep learning imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ncps.torch import LTC  # Official LTC (Liquid Time-Constant) implementation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# Setup and Configuration
# ----------------------------
# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Patient/experiment ID for data organization
numID = 1

# ----------------------------
# Helper Functions
# ----------------------------
def cut_in_sequences(x, y, seq_len, inc=1):
    """
    Slice a long 1D/2D series into overlapping windows for sequence-based learning.
    
    This function creates sequences from the input data for the LTC model.
    For drone data: input shape (32, 1701) -> output shape (seq_len, num_sequences, 1701)
    
    Args:
        x: Input data array (e.g., z-position trajectory)
        y: Target data array (e.g., z-position trajectory) 
        seq_len: Length of each sequence (e.g., 16 timesteps)
        inc: Increment step for creating overlapping sequences
        
    Returns:
        sequences_x: Input sequences with shape (seq_len, num_sequences, features)
        sequences_y: Target sequences with shape (seq_len, num_sequences, features)
    """
    sequences_x, sequences_y = [], []
    for s in range(0, x.shape[0] - seq_len, inc):
        start, end = s, s + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])
    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)

# Global variable to store the number of features per timestep
# This will be set when data is loaded (typically 1701 for drone trajectory)
Nloop = 0

# ----------------------------
# Custom Drone Loss Function 
# ----------------------------
class Custom_CE_Loss(nn.Module):
    """
    Custom loss function that integrates 6DOF drone physics simulation.
    
    This is the core of the parameter estimation system. Instead of using a simple
    MSE loss, this function:
    1. Takes predicted drone parameters from the neural network
    2. Runs a complete 6DOF physics simulation using these parameters
    3. Compares the simulated trajectory with the actual drone trajectory
    4. Returns the physics-based loss for gradient descent
    
    The physics simulation includes:
    - 6DOF dynamics (position, velocity, orientation, angular velocity)
    - Quaternion-based orientation representation
    - Motor dynamics (second-order system)
    - Gyroscopic precession effects
    - Thrust and torque calculations
    - Drag forces and constraints
    
    This approach ensures that the learned parameters are physically meaningful
    and can be used for actual drone control.
    """
    
    def __init__(self, labels, logits, uMotor1, uMotor2, uMotor3, uMotor4, maxMotor, minMotor):
        """
        Initialize the physics-based loss function.
        
        Args:
            labels: Actual z-position trajectory data [T, B, 1]
            logits: Predicted drone parameters from neural network [T, B, 12]
            uMotor1-4: Motor input commands for each motor [T, B, 1]
            maxMotor: Maximum motor speed limits [T, B, 1]
            minMotor: Minimum motor speed limits [T, B, 1]
        """
        super().__init__()
        # Store actual trajectory data for comparison
        self.y_true2 = labels    # [T, B, 1] - actual z position data (matches TF)
        
        # Store predicted parameters from neural network
        self.y_pred2 = logits    # [T, B, 12] - 12 drone parameters
        
        # Store motor input data for physics simulation
        self.y_uMotor1 = uMotor1  # Motor 1 input commands
        self.y_uMotor2 = uMotor2  # Motor 2 input commands
        self.y_uMotor3 = uMotor3  # Motor 3 input commands
        self.y_uMotor4 = uMotor4  # Motor 4 input commands
        
        # Store motor speed constraints
        self.y_maxMotor = maxMotor  # Maximum motor speeds
        self.y_minMotor = minMotor  # Minimum motor speeds

    def forward(self):
        """
        Complete 6DOF drone dynamics simulation with physics-based loss.
        
        This method performs the following steps:
        1. Extract predicted parameters from neural network output
        2. Convert normalized parameters to physical values
        3. Initialize drone state variables
        4. Run physics simulation for T timesteps
        5. Calculate loss between simulated and actual trajectories
        
        Returns:
            total_loss: Combined physics-based loss and parameter penalty
        """
        # Get device and tensor dimensions
        dev = self.y_pred2.device
        T, B, _ = self.y_pred2.shape  # T=timesteps, B=batch_size, 12=parameters

        # ========================================
        # STEP 1: Extract and Convert Parameters
        # ========================================
        # The neural network outputs normalized values [0,1] for each parameter
        # We convert these to physical values with ±95% variation around nominal values
        
        maxChange = 95.0  # Maximum percentage change from nominal values
        getp = lambda k: self.y_pred2[:,:,k]  # Extract parameter k for all timesteps [T,B]
        
        # Convert normalized predictions to physical parameters
        # Each parameter is scaled from [0,1] to [nominal*(1-0.95), nominal*(1+0.95)]
        dxm = (1 + (0.5 - getp(0)) * maxChange / 100.0) * 0.16  # X-arm length (m)
        dym = (1 + (0.5 - getp(1)) * maxChange / 100.0) * 0.16  # Y-arm length (m)
        dzm = (1 + (0.5 - getp(2)) * maxChange / 100.0) * 0.05  # Z-arm length (m)
        
        # Moment of inertia components (kg⋅m²)
        IBxx = (1 + (0.5 - getp(3)) * maxChange / 100.0) * 0.0123  # X-axis inertia
        IByy = (1 + (0.5 - getp(4)) * maxChange / 100.0) * 0.0123  # Y-axis inertia
        IBzz = (1 + (0.5 - getp(5)) * maxChange / 100.0) * 0.0123  # Z-axis inertia
        
        # Aerodynamic and propulsion parameters
        Cd = (1 + (0.5 - getp(6)) * maxChange / 100.0) * 0.1      # Drag coefficient
        kTh = (1 + (0.5 - getp(7)) * maxChange / 100.0) * 1.076e-5 # Thrust coefficient
        kTo = (1 + (0.5 - getp(8)) * maxChange / 100.0) * 1.632e-7 # Torque coefficient
        
        # Motor dynamics parameters
        tau2 = (1 + (0.5 - getp(9)) * maxChange / 100.0) * 0.015  # Motor time constant (s)
        kp = (1 + (0.5 - getp(10)) * maxChange / 100.0) * 1.0     # Motor gain
        damp = (1 + (0.5 - getp(11)) * maxChange / 100.0) * 1.0   # Motor damping

        # ========================================
        # STEP 2: Physical Constants
        # ========================================
        # These are fixed physical constants that don't change during training
        mB = torch.tensor(1.2, device=dev)   # Drone mass (kg)
        g = torch.tensor(9.81, device=dev)   # Gravitational acceleration (m/s²)
        eps = torch.tensor(1e-12, device=dev) # Small epsilon for numerical stability
        
        # Gyroscopic precession parameters (from quad.py)
        IRzz = torch.tensor(0.0001, device=dev)  # Rotor inertia (kg⋅m²)
        uP = 1  # Enable gyroscopic precession (matches quad.py)
        
        # ========================================
        # STEP 3: Coordinate System and Data Setup
        # ========================================
        # We use NED (North-East-Down) coordinate system to match quad.py reference
        # This is important for consistency with the reference physics implementation
        
        # Get actual drone position data for comparison
        if self.y_true2.dim() == 3:
            actual_z = self.y_true2[:,:,0]  # [T,B] - actual z position from [T,B,1]
        else:
            actual_z = self.y_true2  # [T,B] - actual z position
        
        # ========================================
        # STEP 4: Initialize Drone State Variables
        # ========================================
        # All state variables are initialized as [B] tensors (one value per batch)
        # These will be updated during the simulation loop
        
        # Position state (3D position in NED coordinates)
        x_x = torch.zeros(B, device=dev)  # North position (m)
        x_y = torch.zeros(B, device=dev)  # East position (m)
        x_z = torch.zeros(B, device=dev)  # Down position (m) - starts at zero
        
        # Orientation state (quaternion representation: w, x, y, z)
        # Quaternions are used instead of Euler angles to avoid gimbal lock
        quat0 = torch.ones(B, device=dev)   # w component (scalar part)
        quat1 = torch.zeros(B, device=dev)  # x component (i part)
        quat2 = torch.zeros(B, device=dev)  # y component (j part)
        quat3 = torch.zeros(B, device=dev)  # z component (k part)
        # Initial quaternion [1,0,0,0] represents no rotation (identity)
        
        # Linear velocity state (3D velocity in NED coordinates)
        xdot = torch.zeros(B, device=dev)  # North velocity (m/s)
        ydot = torch.zeros(B, device=dev)  # East velocity (m/s)
        zdot = torch.zeros(B, device=dev)  # Down velocity (m/s)
        
        # Angular velocity state (3D angular velocity in body frame)
        p = torch.zeros(B, device=dev)  # Roll rate (rad/s) - rotation about x-axis
        q = torch.zeros(B, device=dev)  # Pitch rate (rad/s) - rotation about y-axis
        r = torch.zeros(B, device=dev)  # Yaw rate (rad/s) - rotation about z-axis
        
        # ========================================
        # STEP 5: Initialize Motor States
        # ========================================
        # Each motor has two state variables: speed (w) and acceleration (wdot)
        # Initial motor speeds are calculated to hover (balance gravity)
        
        # Calculate hover speed for each motor (all motors contribute equally to lift)
        hover_speed = torch.sqrt(torch.clamp(mB * g / (4 * kTh.mean() + eps), min=1e-6))
        
        # Motor 1 state (front-right motor)
        w_hover1 = hover_speed.clone()  # Motor speed (rad/s)
        wdot_hover1 = torch.zeros(B, device=dev)  # Motor acceleration (rad/s²)
        
        # Motor 2 state (front-left motor)
        w_hover2 = hover_speed.clone()
        wdot_hover2 = torch.zeros(B, device=dev)
        
        # Motor 3 state (rear-left motor)
        w_hover3 = hover_speed.clone()
        wdot_hover3 = torch.zeros(B, device=dev)
        
        # Motor 4 state (rear-right motor)
        w_hover4 = hover_speed.clone()
        wdot_hover4 = torch.zeros(B, device=dev)
        
        # ========================================
        # STEP 6: Simulation Setup
        # ========================================
        # Set up simulation parameters and storage arrays
        
        limitLoop = T  # Number of simulation steps (matches data timesteps)
        tau = 0.005    # Time step (s) - smaller for better accuracy
        max_inc = 400.0  # Maximum change per timestep for numerical stability
        
        # Initialize arrays to store predicted trajectory
        predicted_x = torch.zeros((limitLoop, B), device=dev)  # North position
        predicted_y = torch.zeros((limitLoop, B), device=dev)  # East position
        predicted_z = torch.zeros((limitLoop, B), device=dev)  # Down position
        
        # Store initial states (t=0)
        predicted_x[0] = x_x
        predicted_y[0] = x_y
        predicted_z[0] = x_z
        
        # ========================================
        # STEP 7: Main Physics Simulation Loop
        # ========================================
        # This is the core of the physics simulation
        # For each timestep, we:
        # 1. Get motor inputs from data
        # 2. Update motor dynamics
        # 3. Calculate thrust and torque
        # 4. Update 6DOF dynamics
        # 5. Store predicted states
        
        for i in range(1, limitLoop):
            # Current timestep index - use actual timestep
            t_idx = i  # Use actual timestep index
            
            # ========================================
            # STEP 7.1: Get Motor Inputs from Data
            # ========================================
            # Extract motor input commands for current timestep
            # These come from the actual flight data and represent the pilot's commands
            
            if self.y_uMotor1.dim() == 3:
                # Handle 3D tensors [T, B, Nloop] - take first sequence
                uMotor1_curr = self.y_uMotor1[t_idx, :, 0]  # [B] - Motor 1 input
                uMotor2_curr = self.y_uMotor2[t_idx, :, 0]  # [B] - Motor 2 input
                uMotor3_curr = self.y_uMotor3[t_idx, :, 0]  # [B] - Motor 3 input
                uMotor4_curr = self.y_uMotor4[t_idx, :, 0]  # [B] - Motor 4 input
            else:
                # Handle 2D tensors [T, B]
                uMotor1_curr = self.y_uMotor1[t_idx]  # [B] - Motor 1 input
                uMotor2_curr = self.y_uMotor2[t_idx]  # [B] - Motor 2 input
                uMotor3_curr = self.y_uMotor3[t_idx]  # [B] - Motor 3 input
                uMotor4_curr = self.y_uMotor4[t_idx]  # [B] - Motor 4 input
            
            # ========================================
            # STEP 7.2: Motor Dynamics (Second-Order System)
            # ========================================
            # Each motor is modeled as a second-order system:
            # τ²ẅ + 2ζτẇ + w = kp*u
            # where τ=time constant, ζ=damping, kp=gain, u=input, w=speed
            
            # Get motor parameters for current timestep
            damp_curr = damp[t_idx]    # Damping ratio ζ
            tau2_curr = tau2[t_idx]    # Time constant τ
            kp_curr = kp[t_idx]        # Motor gain kp
            
            # Calculate motor accelerations using second-order dynamics
            # wddot = (-2*ζ*τ*ẇ - w + kp*u) / τ²
            wddotM1 = (-2.0 * damp_curr * tau2_curr * wdot_hover1 - w_hover1 + 
                       kp_curr * uMotor1_curr / (tau2_curr ** 2))
            wddotM2 = (-2.0 * damp_curr * tau2_curr * wdot_hover2 - w_hover2 + 
                       kp_curr * uMotor2_curr / (tau2_curr ** 2))
            wddotM3 = (-2.0 * damp_curr * tau2_curr * wdot_hover3 - w_hover3 + 
                       kp_curr * uMotor3_curr / (tau2_curr ** 2))
            wddotM4 = (-2.0 * damp_curr * tau2_curr * wdot_hover4 - w_hover4 + 
                       kp_curr * uMotor4_curr / (tau2_curr ** 2))
            
            # ========================================
            # STEP 7.3: Apply Motor Speed Constraints
            # ========================================
            # Real motors have physical limits on their speed
            # We clamp the motor speeds to stay within these limits
            
            if self.y_minMotor.dim() == 3:
                # Handle 3D tensors [T, B, Nloop] - take first sequence
                minMotor_curr = self.y_minMotor[t_idx, :, 0]  # [B] - minimum motor speeds
                maxMotor_curr = self.y_maxMotor[t_idx, :, 0]  # [B] - maximum motor speeds
            else:
                # Handle 2D tensors [T, B]
                minMotor_curr = self.y_minMotor[t_idx]  # [B] - minimum motor speeds
                maxMotor_curr = self.y_maxMotor[t_idx]  # [B] - maximum motor speeds
            
            # Clamp motor speeds to physical limits
            w_hover1M1 = torch.clamp(w_hover1, minMotor_curr, maxMotor_curr)  # Motor 1
            w_hover2M1 = torch.clamp(w_hover2, minMotor_curr, maxMotor_curr)  # Motor 2
            w_hover3M1 = torch.clamp(w_hover3, minMotor_curr, maxMotor_curr)  # Motor 3
            w_hover4M1 = torch.clamp(w_hover4, minMotor_curr, maxMotor_curr)  # Motor 4
            
            # ========================================
            # STEP 7.4: Calculate Thrust and Torque
            # ========================================
            # Thrust and torque are proportional to the square of motor speed
            # Thrust = kTh * w² (upward force)
            # Torque = kTo * w² (rotational force)
            
            # Get thrust and torque coefficients for current timestep
            kTh_curr = kTh[t_idx]  # Thrust coefficient
            kTo_curr = kTo[t_idx]  # Torque coefficient
            
            # Calculate thrust for each motor (proportional to speed squared)
            ThrM1 = kTh_curr * (w_hover1M1 ** 2)  # Motor 1 thrust (N)
            ThrM2 = kTh_curr * (w_hover2M1 ** 2)  # Motor 2 thrust (N)
            ThrM3 = kTh_curr * (w_hover3M1 ** 2)  # Motor 3 thrust (N)
            ThrM4 = kTh_curr * (w_hover4M1 ** 2)  # Motor 4 thrust (N)
            
            # Calculate torque for each motor (proportional to speed squared)
            TorM1 = kTo_curr * (w_hover1M1 ** 2)  # Motor 1 torque (N⋅m)
            TorM2 = kTo_curr * (w_hover2M1 ** 2)  # Motor 2 torque (N⋅m)
            TorM3 = kTo_curr * (w_hover3M1 ** 2)  # Motor 3 torque (N⋅m)
            TorM4 = kTo_curr * (w_hover4M1 ** 2)
            
            # ========================================
            # STEP 7.5: 6DOF Dynamics (Newton's Laws)
            # ========================================
            # This is the core of the physics simulation
            # We apply Newton's laws to calculate accelerations from forces
            
            # Get drag coefficient for current timestep
            Cd_curr = Cd[t_idx]
            
            # Linear acceleration in North direction (x-axis in NED)
            # F = ma, so a = F/m
            # Forces: drag (opposes motion) + thrust (from quaternion rotation)
            dummyxdot = (xdot + tau * (
                Cd_curr * torch.sign(-xdot) * (xdot ** 2) +  # Drag force (opposes velocity)
                2 * (quat0 * quat2 + quat1 * quat3) *        # Thrust component (from quaternion)
                (ThrM1 + ThrM2 + ThrM3 + ThrM4)              # Total thrust
            ) / mB)
            
            # Linear acceleration in East direction (y-axis in NED)
            dummyydot = (ydot + tau * (
                Cd_curr * torch.sign(-ydot) * (ydot ** 2) -  # Drag force (opposes velocity)
                2 * (quat0 * quat1 - quat2 * quat3) *        # Thrust component (from quaternion)
                (ThrM1 + ThrM2 + ThrM3 + ThrM4)              # Total thrust
            ) / mB)
            
            # Linear acceleration in Down direction (z-axis in NED)
            # This includes gravity and thrust in the vertical direction
            dummyzdot = (zdot + tau * (
                -Cd_curr * torch.sign(zdot) * (zdot ** 2) +  # Drag force (opposes velocity)
                (ThrM1 + ThrM2 + ThrM3 + ThrM4) *            # Total thrust
                (quat0 ** 2 - quat1 ** 2 - quat2 ** 2 + quat3 ** 2) -  # Vertical thrust component
                g * mB                                        # Gravity (always downward)
            ) / mB)
            
            # Quaternion dynamics - matches TF exactly
            dummyq0 = (quat0 + tau * (
                -0.5 * p * quat1 - 
                0.5 * q * quat2 - 
                0.5 * r * quat3
            ))
            
            dummyq1 = (quat1 + tau * (
                0.5 * p * quat0 - 
                0.5 * q * quat3 + 
                0.5 * r * quat2
            ))
            
            dummyq2 = (quat2 + tau * (
                0.5 * p * quat3 + 
                0.5 * q * quat0 - 
                0.5 * r * quat1
            ))
            
            dummyq3 = (quat3 + tau * (
                -0.5 * p * quat2 + 
                0.5 * q * quat1 + 
                0.5 * r * quat0
            ))
            
            # Angular velocity dynamics - matches quad.py reference exactly
            # Use parameter values at current timestep
            IBxx_curr = IBxx[t_idx]
            IByy_curr = IByy[t_idx]
            IBzz_curr = IBzz[t_idx]
            dxm_curr = dxm[t_idx]
            dym_curr = dym[t_idx]
            
            # Calculate gyroscopic precession terms (from quad.py)
            gyro_p_term = -uP * IRzz * (w_hover1M1 - w_hover2M1 + w_hover3M1 - w_hover4M1) * q
            gyro_q_term = uP * IRzz * (w_hover1M1 - w_hover2M1 + w_hover3M1 - w_hover4M1) * p
            
            dummyp = (p + tau * (
                (IByy_curr - IBzz_curr) * q * r + 
                gyro_p_term + 
                (ThrM1 - ThrM2 - ThrM3 + ThrM4) * dym_curr
            ) / IBxx_curr)
            
            dummyq = (q + tau * (
                (IBzz_curr - IBxx_curr) * p * r + 
                gyro_q_term + 
                (ThrM1 + ThrM2 - ThrM3 - ThrM4) * dxm_curr
            ) / IByy_curr)
            
            # Fixed R dynamics - correct signs from quad.py
            dummyr = (r + tau * (
                (IBxx_curr - IByy_curr) * p * q - 
                TorM1 + TorM2 - TorM3 + TorM4
            ) / IBzz_curr)
            
            # Position updates with improved integration
            # Using simple Euler for now, but could be upgraded to RK4
            dummy_x_x = x_x + tau * dummyxdot
            dummy_x_y = x_y + tau * dummyydot
            dummy_x_z = x_z + tau * dummyzdot
            
            # Motor state updates
            dummy_w_hover1 = w_hover1M1 + tau * wdot_hover1
            dummy_w_hover2 = w_hover2M1 + tau * wdot_hover2
            dummy_w_hover3 = w_hover3M1 + tau * wdot_hover3
            dummy_w_hover4 = w_hover4M1 + tau * wdot_hover4
            
            dummy_wdot_hover1 = wdot_hover1 + tau * wddotM1
            dummy_wdot_hover2 = wdot_hover2 + tau * wddotM2
            dummy_wdot_hover3 = wdot_hover3 + tau * wddotM3
            dummy_wdot_hover4 = wdot_hover4 + tau * wddotM4
            
            # Stability check for z position - matches TF exactly
            mask = torch.logical_and(
                torch.isfinite(dummy_x_z),
                torch.abs(dummy_x_z - x_z) <= max_inc
            )
            dummy_x_z = torch.where(mask, dummy_x_z, x_z)
            
            # Store predicted states
            predicted_x[i] = dummy_x_x
            predicted_y[i] = dummy_x_y
            predicted_z[i] = dummy_x_z
            
            # Update states for next iteration (matches TF exactly)
            x_x = dummy_x_x
            x_y = dummy_x_y
            x_z = dummy_x_z
            quat0 = dummyq0
            quat1 = dummyq1
            quat2 = dummyq2
            quat3 = dummyq3
            xdot = dummyxdot
            ydot = dummyydot
            zdot = dummyzdot
            p = dummyp
            q = dummyq
            r = dummyr
            w_hover1 = dummy_w_hover1
            w_hover2 = dummy_w_hover2
            w_hover3 = dummy_w_hover3
            w_hover4 = dummy_w_hover4
            wdot_hover1 = dummy_wdot_hover1
            wdot_hover2 = dummy_wdot_hover2
            wdot_hover3 = dummy_wdot_hover3
            wdot_hover4 = dummy_wdot_hover4
        
        # ========================================
        # STEP 8: Calculate Physics-Based Loss
        # ========================================
        # The loss function compares the simulated trajectory with the actual trajectory
        # This is what drives the parameter estimation - the neural network learns
        # parameters that make the simulation match the real drone behavior
        
        # Calculate MSE loss for entire z-position trajectory
        # This is the primary loss that measures how well our simulation matches reality
        mse_loss = torch.mean((predicted_z - actual_z) ** 2)
        
        # ========================================
        # STEP 9: Parameter Constraint Penalty
        # ========================================
        # Add penalties to ensure learned parameters are physically reasonable
        # This prevents the network from learning unrealistic values
        
        param_penalty = 0.0
        
        # Arm length constraints (must be positive and reasonable)
        param_penalty += torch.mean(torch.relu(-dxm))  # dxm > 0
        param_penalty += torch.mean(torch.relu(-dym))  # dym > 0
        param_penalty += torch.mean(torch.relu(-dzm))  # dzm > 0
        param_penalty += torch.mean(torch.relu(dxm - 0.5))  # dxm < 0.5m (reasonable upper bound)
        param_penalty += torch.mean(torch.relu(dym - 0.5))  # dym < 0.5m
        
        # Inertia constraints (must be positive and reasonable)
        param_penalty += torch.mean(torch.relu(-IBxx))  # IBxx > 0
        param_penalty += torch.mean(torch.relu(-IByy))  # IByy > 0
        param_penalty += torch.mean(torch.relu(-IBzz))  # IBzz > 0
        param_penalty += torch.mean(torch.relu(IBxx - 0.1))  # IBxx < 0.1 kg⋅m²
        param_penalty += torch.mean(torch.relu(IByy - 0.1))  # IByy < 0.1 kg⋅m²
        param_penalty += torch.mean(torch.relu(IBzz - 0.1))  # IBzz < 0.1 kg⋅m²
        
        # Physical parameter constraints
        param_penalty += torch.mean(torch.relu(-kTh))  # kTh > 0
        param_penalty += torch.mean(torch.relu(-kTo))  # kTo > 0
        param_penalty += torch.mean(torch.relu(-Cd))   # Cd > 0
        param_penalty += torch.mean(torch.relu(kTh - 2e-5))  # kTh < 2e-5 (reasonable upper bound)
        param_penalty += torch.mean(torch.relu(kTo - 5e-7))  # kTo < 5e-7
        
        # Motor dynamics constraints
        param_penalty += torch.mean(torch.relu(-tau2))  # tau2 > 0
        param_penalty += torch.mean(torch.relu(-kp))    # kp > 0
        param_penalty += torch.mean(torch.relu(-damp))  # damp > 0
        param_penalty += torch.mean(torch.relu(tau2 - 0.1))  # tau2 < 0.1s
        param_penalty += torch.mean(torch.relu(kp - 10.0))   # kp < 10.0
        
        # Calculate RMSE for reporting
        rmse_loss = torch.sqrt(mse_loss)
        
        # HONEST LOSS: No artificial weighting that reduces loss values
        total_loss = mse_loss + 0.05 * param_penalty
        
        # Store predicted trajectory and parameters for debugging
        self.predicted_z = predicted_z
        self.dxm = dxm
        self.dym = dym
        self.dzm = dzm
        self.kTh = kTh
        self.kTo = kTo
        self.rmse = rmse_loss
        
        return total_loss

# ----------------------------
# Data Handler
# ----------------------------
class HarData:
    def __init__(self, seq_len=16):
        print(f"Parsing for Patient File {numID}")
        # Prefer new extracted data in Dronepipeline/data; fall back to UCI HAR layout
        dp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        use_dp = os.path.exists(os.path.join(dp_dir, "xData.txt"))

        if use_dp:
            # why: use freshly extracted matrices; what: load and split 80/20 into train/test
            def L(name):
                return np.loadtxt(os.path.join(dp_dir, name))
            xX = L("xData.txt"); xY = L("yData.txt"); xZ = L("zData.txt")
            u1 = L("uMotor1.txt") if os.path.exists(os.path.join(dp_dir, "uMotor1.txt")) else np.zeros_like(xZ)
            u2 = L("uMotor2.txt") if os.path.exists(os.path.join(dp_dir, "uMotor2.txt")) else np.zeros_like(xZ)
            u3 = L("uMotor3.txt") if os.path.exists(os.path.join(dp_dir, "uMotor3.txt")) else np.zeros_like(xZ)
            u4 = L("uMotor4.txt") if os.path.exists(os.path.join(dp_dir, "uMotor4.txt")) else np.zeros_like(xZ)
            mx = L("maxMotor.txt") if os.path.exists(os.path.join(dp_dir, "maxMotor.txt")) else np.full_like(xZ, 2000.0)
            mn = L("minMotor.txt") if os.path.exists(os.path.join(dp_dir, "minMotor.txt")) else np.zeros_like(xZ)

            rows = xX.shape[0]
            split_idx = max(1, int(0.8 * rows))

            train_x_x, test_x_x = xX[:split_idx, :], xX[split_idx:, :]
            train_x_y, test_x_y = xY[:split_idx, :], xY[split_idx:, :]
            train_x_z, test_x_z = xZ[:split_idx, :], xZ[split_idx:, :]

            # actual position proxies (consistent with original): input minus 1
            train_y_x, test_y_x = train_x_x - 1, test_x_x - 1
            train_y_y, test_y_y = train_x_y - 1, test_x_y - 1
            train_y_z, test_y_z = train_x_z - 1, test_x_z - 1

            train_uMotor1, test_uMotor1 = u1[:split_idx, :], u1[split_idx:, :]
            train_uMotor2, test_uMotor2 = u2[:split_idx, :], u2[split_idx:, :]
            train_uMotor3, test_uMotor3 = u3[:split_idx, :], u3[split_idx:, :]
            train_uMotor4, test_uMotor4 = u4[:split_idx, :], u4[split_idx:, :]

            train_maxMotor, test_maxMotor = mx[:split_idx, :], mx[split_idx:, :]
            train_minMotor, test_minMotor = mn[:split_idx, :], mn[split_idx:, :]
        else:
            raise FileNotFoundError(f"Expected new data in {dp_dir}. Please run droneExtract.py and droneExtractAudio.py first.")

        # Get Nloop from data
        global Nloop
        Nloop = test_x_x.shape[1]  # Use actual data size (1701)
        print(f"Nloop {Nloop}")

        # Create sequences
        train_x_x, train_y_x = cut_in_sequences(train_x_x, train_y_x, seq_len)
        train_x_y, train_y_y = cut_in_sequences(train_x_y, train_y_y, seq_len)
        train_x_z, train_y_z = cut_in_sequences(train_x_z, train_y_z, seq_len)

        train_uMotor1, train_uMotor2 = cut_in_sequences(train_uMotor1, train_uMotor2, seq_len)
        train_uMotor3, train_uMotor4 = cut_in_sequences(train_uMotor3, train_uMotor4, seq_len)
        train_maxMotor, train_minMotor = cut_in_sequences(train_maxMotor, train_minMotor, seq_len)

        test_x_x, test_y_x = cut_in_sequences(test_x_x, test_y_x, seq_len, inc=8)
        test_x_y, test_y_y = cut_in_sequences(test_x_y, test_y_y, seq_len, inc=8)
        test_x_z, test_y_z = cut_in_sequences(test_x_z, test_y_z, seq_len, inc=8)

        test_uMotor1, test_uMotor2 = cut_in_sequences(test_uMotor1, test_uMotor2, seq_len, inc=8)
        test_uMotor3, test_uMotor4 = cut_in_sequences(test_uMotor3, test_uMotor4, seq_len, inc=8)
        test_maxMotor, test_minMotor = cut_in_sequences(test_maxMotor, test_minMotor, seq_len, inc=8)

        # Validation split
        valid_size = int(0.1 * train_x_x.shape[1])
        print(f"Validation split: {valid_size}, training split: {train_x_x.shape[1] - valid_size}")

        # Split data - matches TF exactly
        self.valid_x_x = torch.tensor(train_x_x[:, :valid_size], dtype=torch.float32)
        self.valid_x_y = torch.tensor(train_x_y[:, :valid_size], dtype=torch.float32)
        self.valid_x_z = torch.tensor(train_x_z[:, :valid_size], dtype=torch.float32)
        self.valid_y_x = torch.tensor(train_y_x[:, :valid_size], dtype=torch.float32)
        self.valid_y_y = torch.tensor(train_y_y[:, :valid_size], dtype=torch.float32)
        self.valid_y_z = torch.tensor(train_y_z[:, :valid_size], dtype=torch.float32)

        self.valid_uMotor1 = torch.tensor(train_uMotor1[:, :valid_size], dtype=torch.float32)
        self.valid_uMotor2 = torch.tensor(train_uMotor2[:, :valid_size], dtype=torch.float32)
        self.valid_uMotor3 = torch.tensor(train_uMotor3[:, :valid_size], dtype=torch.float32)
        self.valid_uMotor4 = torch.tensor(train_uMotor4[:, :valid_size], dtype=torch.float32)
        self.valid_maxMotor = torch.tensor(train_maxMotor[:, :valid_size], dtype=torch.float32)
        self.valid_minMotor = torch.tensor(train_minMotor[:, :valid_size], dtype=torch.float32)

        self.train_x_x = torch.tensor(train_x_x[:, valid_size:], dtype=torch.float32)
        self.train_x_y = torch.tensor(train_x_y[:, valid_size:], dtype=torch.float32)
        self.train_x_z = torch.tensor(train_x_z[:, valid_size:], dtype=torch.float32)
        self.train_y_x = torch.tensor(train_y_x[:, valid_size:], dtype=torch.float32)
        self.train_y_y = torch.tensor(train_y_y[:, valid_size:], dtype=torch.float32)
        self.train_y_z = torch.tensor(train_y_z[:, valid_size:], dtype=torch.float32)

        self.train_uMotor1 = torch.tensor(train_uMotor1[:, valid_size:], dtype=torch.float32)
        self.train_uMotor2 = torch.tensor(train_uMotor2[:, valid_size:], dtype=torch.float32)
        self.train_uMotor3 = torch.tensor(train_uMotor3[:, valid_size:], dtype=torch.float32)
        self.train_uMotor4 = torch.tensor(train_uMotor4[:, valid_size:], dtype=torch.float32)
        self.train_maxMotor = torch.tensor(train_maxMotor[:, valid_size:], dtype=torch.float32)
        self.train_minMotor = torch.tensor(train_minMotor[:, valid_size:], dtype=torch.float32)

        self.test_x_x = torch.tensor(test_x_x, dtype=torch.float32)
        self.test_x_y = torch.tensor(test_x_y, dtype=torch.float32)
        self.test_x_z = torch.tensor(test_x_z, dtype=torch.float32)
        self.test_y_x = torch.tensor(test_y_x, dtype=torch.float32)
        self.test_y_y = torch.tensor(test_y_y, dtype=torch.float32)
        self.test_y_z = torch.tensor(test_y_z, dtype=torch.float32)

        self.test_uMotor1 = torch.tensor(test_uMotor1, dtype=torch.float32)
        self.test_uMotor2 = torch.tensor(test_uMotor2, dtype=torch.float32)
        self.test_uMotor3 = torch.tensor(test_uMotor3, dtype=torch.float32)
        self.test_uMotor4 = torch.tensor(test_uMotor4, dtype=torch.float32)
        self.test_maxMotor = torch.tensor(test_maxMotor, dtype=torch.float32)
        self.test_minMotor = torch.tensor(test_minMotor, dtype=torch.float32)

        print(f"Total number of test sequences: {self.test_x_x.shape[1]}")

    def iterate_train(self, batch_size=32):
        total_seqs = self.train_x_x.shape[1]
        permutation = torch.randperm(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            indices = permutation[start:end]

            batch_x_x = self.train_x_x[:, indices]
            batch_x_y = self.train_x_y[:, indices]
            batch_x_z = self.train_x_z[:, indices]
            batch_y_x = self.train_y_x[:, indices]
            batch_y_y = self.train_y_y[:, indices]
            batch_y_z = self.train_y_z[:, indices]

            batch_uMotor1 = self.train_uMotor1[:, indices]
            batch_uMotor2 = self.train_uMotor2[:, indices]
            batch_uMotor3 = self.train_uMotor3[:, indices]
            batch_uMotor4 = self.train_uMotor4[:, indices]
            batch_maxMotor = self.train_maxMotor[:, indices]
            batch_minMotor = self.train_minMotor[:, indices]

            yield (batch_x_x, batch_x_y, batch_x_z, batch_y_x, batch_y_y, batch_y_z,
                  batch_uMotor1, batch_uMotor2, batch_uMotor3, batch_uMotor4,
                  batch_maxMotor, batch_minMotor)

# ----------------------------
# Neural Network Model Class
# ----------------------------
class HarModel(nn.Module):
    """
    Neural network model for drone parameter estimation.
    
    This class implements the LTC (Liquid Time-Constant) neural network that learns
    to predict drone physical parameters from trajectory data. The model takes
    sequences of drone trajectory data as input and outputs 12 physical parameters.
    
    Architecture:
    - Input: [T, B, Nloop] where T=timesteps, B=batch_size, Nloop=1701 features
    - Output: [T, B, 12] where 12 is the number of drone parameters
    - Uses LTC for sequence-to-sequence learning
    """
    
    def __init__(self, model_type, model_size, learning_rate=0.001):
        """
        Initialize the neural network model.
        
        Args:
            model_type: Type of model ("ltc", "lstm", etc.)
            model_size: Hidden layer size
            learning_rate: Learning rate for optimization
        """
        super().__init__()
        self.model_type = model_type
        self.model_size = model_size
        self.constrain_op = None

        # Input size is the number of features per timestep (1701 for drone trajectory)
        input_size = Nloop  # 1701 timesteps

        print("Beginning")

        if model_type == "lstm":
            self.rnn = nn.LSTM(input_size, model_size, batch_first=False)
        elif model_type.startswith("ltc"):
            # Using official LTC implementation from ncps library
            learning_rate = 0.005  # Reduced learning rate for better convergence
            
            # Create official LTC with optimized configuration
            self.wm = LTC(
                input_size=input_size,
                units=model_size,
                return_sequences=True,
                batch_first=False,  # Time-major format
                mixed_memory=False,  # No memory cell for simplicity
                ode_unfolds=8,  # Increased ODE solver steps for better accuracy
                epsilon=1e-10  # Improved numerical stability
            )
            self.rnn = self.wm
            # Official LTC handles parameter constraints internally
            self.constrain_op = None
        elif model_type == "node":
            self.rnn = nn.RNN(input_size, model_size, batch_first=False)
        elif model_type == "ctgru":
            self.rnn = nn.GRU(input_size, model_size, batch_first=False)
        elif model_type == "ctrnn":
            self.rnn = nn.RNN(input_size, model_size, batch_first=False)
        else:
            raise ValueError(f"Unknown model type '{model_type}'")

        self.dense = nn.Linear(model_size, 12)  # 12 drone parameters
        self.sigmoid = nn.Sigmoid()

        # Optimized Adam optimizer with better settings
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, 
                                   weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8)
        self.to(device)
        
        # Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, 
            threshold=1e-4, min_lr=1e-6, verbose=True
        )

        # Results / checkpoints (mirrors TF paths)
        self.result_file = os.path.join("results", "har", f"{model_type}_{model_size}.csv")
        os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
        if not os.path.isfile(self.result_file):
            with open(self.result_file, "w") as f:
                f.write("best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("torch_sessions", "har", f"{model_type}")
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

    def forward(self, x_z):
        """
        x_z: [T, B, Nloop] - full z data
        Returns y: [T, B, 12] - 12 drone parameters
        """
        if self.model_type.startswith("ltc"):
            # Official LTC returns (output, hidden_state) tuple
            out, _ = self.rnn(x_z)           # [T,B,H]
        else:
            # Other RNNs return (output, hidden_state) tuple
            out, _ = self.rnn(x_z)           # [T,B,H]
        
        T, B, H = out.shape
        y = self.sigmoid(self.dense(out.reshape(T*B, H))).reshape(T, B, 12)
        return y

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_path + ".pth")

    def restore(self):
        if os.path.isfile(self.checkpoint_path + ".pth"):
            self.load_state_dict(torch.load(self.checkpoint_path + ".pth", map_location=device))

    def compute_loss(self, y_pred, target_y_z, uMotor1, uMotor2, uMotor3, uMotor4, maxMotor, minMotor):
        """Build the loss object and call .forward()."""
        loss_fn = Custom_CE_Loss(target_y_z, y_pred, uMotor1, uMotor2, uMotor3, uMotor4, maxMotor, minMotor)
        return loss_fn.forward()

    def fit(self, gesture_data, epochs, verbose=True, log_period=50):
        best_valid_loss = 10000
        best_valid_stats = (0, 0, 0, 0, 0, 0)
        self.save()
        
        for e in range(epochs):
            if e % log_period == 0:
                # Test evaluation
                self.eval()
                with torch.no_grad():
                    test_y_pred = self.forward(gesture_data.test_x_z)
                    test_loss = self.compute_loss(test_y_pred, gesture_data.test_y_z, 
                                                gesture_data.test_uMotor1, gesture_data.test_uMotor2,
                                                gesture_data.test_uMotor3, gesture_data.test_uMotor4,
                                                gesture_data.test_maxMotor, gesture_data.test_minMotor)
                    
                    # Validation evaluation
                    valid_y_pred = self.forward(gesture_data.valid_x_z)
                    valid_loss = self.compute_loss(valid_y_pred, gesture_data.valid_y_z,
                                                 gesture_data.valid_uMotor1, gesture_data.valid_uMotor2,
                                                 gesture_data.valid_uMotor3, gesture_data.valid_uMotor4,
                                                 gesture_data.valid_maxMotor, gesture_data.valid_minMotor)
                    
                    if valid_loss < best_valid_loss and e > 0:
                        best_valid_loss = valid_loss
                        best_valid_stats = (e, 0, 0, valid_loss, 0, test_loss)
                        self.save()

            # Training
            self.train()
            losses = []
            
            batch_count = 0
            for batch_x_x, batch_x_y, batch_x_z, batch_y_x, batch_y_y, batch_y_z, \
                batch_uMotor1, batch_uMotor2, batch_uMotor3, batch_uMotor4, \
                batch_maxMotor, batch_minMotor in gesture_data.iterate_train(batch_size=1):
                
                batch_count += 1
                self.optimizer.zero_grad()
                
                # Forward pass
                y_pred = self.forward(batch_x_z)
                loss = self.compute_loss(y_pred, batch_y_z, batch_uMotor1, batch_uMotor2,
                                       batch_uMotor3, batch_uMotor4, batch_maxMotor, batch_minMotor)
                
                # Backward pass with gradient clipping
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                losses.append(loss.item())
                if batch_count == 1:  # Debug first batch
                    print(f"Debug - Batch shapes: x_z={batch_x_z.shape}, y_z={batch_y_z.shape}, loss={loss.item():.6f}")
            
            if batch_count == 0:
                print("Warning: No training batches found!")
            
            if verbose and e % log_period == 0:
                avg_loss = sum(losses) / len(losses) if losses else 0.0
                print(f"Epoch {e:03d}, train loss: {avg_loss:.4f}, valid loss: {valid_loss:.4f}, test loss: {test_loss:.4f}")
                
            # Learning rate scheduling
            self.scheduler.step(valid_loss)
        
        self.restore()
        best_epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss = best_valid_stats
        print(f"Best epoch {best_epoch:03d}, train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}, test loss: {test_loss:.4f}")
        
        # Save results
        with open(self.result_file, "a") as f:
            f.write(f"{best_epoch:03d}, {train_loss:.4f}, {train_acc:.4f}, {valid_loss:.4f}, {valid_acc:.4f}, {test_loss:.4f}, 0.0\n")

        # Save parameters to CSV
        file_path = 'outputV2GPUV3_torch_ltc.csv'
        columns = ['numID', 'test_loss', 'test_mard', 'rmse', 'mB', 'g', 'dxm', 'dym', 'dzm', 
                  'IBxx', 'IByy', 'IBzz', 'Cd', 'kTh', 'kTo', 'tau', 'kp', 'damp']
        
        # Extract final parameters for reporting
        with torch.no_grad():
            final_pred = self.forward(gesture_data.test_x_z)
            
            # Calculate RMSE from the final prediction
            loss_fn = Custom_CE_Loss(gesture_data.test_y_z, final_pred, 
                                   gesture_data.test_uMotor1, gesture_data.test_uMotor2, 
                                   gesture_data.test_uMotor3, gesture_data.test_uMotor4, 
                                   gesture_data.test_maxMotor, gesture_data.test_minMotor)
            loss_fn.forward()  # Run the simulation
            rmse_value = loss_fn.rmse.item() if hasattr(loss_fn.rmse, 'item') else float(loss_fn.rmse)
            
            maxChange = 95.0
            getp = lambda k: final_pred[:,:,k].mean()
            
            dxm = (1 + (0.5 - getp(0)) * maxChange / 100.0) * 0.16
            dym = (1 + (0.5 - getp(1)) * maxChange / 100.0) * 0.16
            dzm = (1 + (0.5 - getp(2)) * maxChange / 100.0) * 0.05
            IBxx = (1 + (0.5 - getp(3)) * maxChange / 100.0) * 0.0123
            IByy = (1 + (0.5 - getp(4)) * maxChange / 100.0) * 0.0123
            IBzz = (1 + (0.5 - getp(5)) * maxChange / 100.0) * 0.0123
            Cd = (1 + (0.5 - getp(6)) * maxChange / 100.0) * 0.1
            kTh = (1 + (0.5 - getp(7)) * maxChange / 100.0) * 1.076e-5
            kTo = (1 + (0.5 - getp(8)) * maxChange / 100.0) * 1.632e-7
            tau = (1 + (0.5 - getp(9)) * maxChange / 100.0) * 0.015
            kp = (1 + (0.5 - getp(10)) * maxChange / 100.0) * 1.0
            damp = (1 + (0.5 - getp(11)) * maxChange / 100.0) * 1.0
        
        # Ensure test_loss is a scalar
        test_loss_scalar = test_loss.item() if hasattr(test_loss, 'item') else float(test_loss)
        
        # RMSE is calculated above in the loss function
        
        values = [numID, test_loss_scalar, 0.0, rmse_value, 1.2, 9.81, dxm.item(), dym.item(), dzm.item(),
                 IBxx.item(), IByy.item(), IBzz.item(), Cd.item(), kTh.item(), kTo.item(),
                 tau.item(), kp.item(), damp.item()]
        
        file_exists = os.path.exists(file_path)
        with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(columns)
            writer.writerow(values)
        
        print(f"Results saved to {file_path}")

# ----------------------------
# THETA SIMULATION (full) - ported compactly from EMMA_drone_torch_ltc.py
# ----------------------------
class ThetaSimulator:
    def __init__(self, theta_coeffs):
        # Why: hold theta (physical parameters) for simulation; What: dict of dxm, dym, dzm, IBxx, IByy, IBzz, Cd, kTh, kTo, tau2, kp, damp
        self.theta_coeffs = theta_coeffs

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quaternion_rotate_vector(self, q, v):
        vq = np.array([0, v[0], v[1], v[2]])
        qc = np.array([q[0], -q[1], -q[2], -q[3]])
        return self.quaternion_multiply(self.quaternion_multiply(q, vq), qc)[1:]

    def simulate_trajectory(self, initial_state, control_inputs, dt=0.005, duration=None):
        if duration is None:
            duration = len(control_inputs) * dt
        n_steps = int(duration / dt)
        times = np.linspace(0, duration, n_steps)

        states = np.zeros((n_steps, 13))
        states[0] = initial_state

        dxm = self.theta_coeffs['dxm']; dym = self.theta_coeffs['dym']; dzm = self.theta_coeffs['dzm']
        IBxx = self.theta_coeffs['IBxx']; IByy = self.theta_coeffs['IByy']; IBzz = self.theta_coeffs['IBzz']
        Cd = self.theta_coeffs['Cd']; kTh = self.theta_coeffs['kTh']; kTo = self.theta_coeffs['kTo']
        mB = 1.2; g = 9.81

        for i in range(1, n_steps):
            x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz = states[i-1]
            idx = min(int(i * len(control_inputs) / n_steps), len(control_inputs) - 1)
            u1, u2, u3, u4 = control_inputs[idx]

            thrust = kTh * (u1**2 + u2**2 + u3**2 + u4**2)
            torque_x = kTo * (u2**2 - u4**2) * dym
            torque_y = kTo * (u1**2 - u3**2) * dxm
            torque_z = kTo * (u1**2 - u2**2 + u3**2 - u4**2)

            F_body = np.array([0.0, 0.0, thrust])
            q = np.array([qw, qx, qy, qz])
            F_world = self.quaternion_rotate_vector(q, F_body)
            F_world[2] -= mB * g
            vel = np.array([vx, vy, vz])
            F_world += -Cd * np.linalg.norm(vel) * vel

            vx += (F_world[0] / mB) * dt
            vy += (F_world[1] / mB) * dt
            vz += (F_world[2] / mB) * dt
            x += vx * dt; y += vy * dt; z += vz * dt

            wx += (torque_x / IBxx) * dt
            wy += (torque_y / IByy) * dt
            wz += (torque_z / IBzz) * dt

            states[i] = [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        return states, times

    def plot_trajectory(self, states, times, save_plot=True):
        # Why: static visualization (Z-up); What: 3D path + time plots with safe margins
        pos = states[:, :3]; vel = states[:, 7:10]
        pos_plot = pos.copy(); pos_plot[:, 2] = -pos_plot[:, 2]
        vel_plot = vel.copy(); vel_plot[:, 2] = -vel_plot[:, 2]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax1 = fig.add_subplot(2,2,1, projection='3d')
        ax1.plot(pos_plot[:,0], pos_plot[:,1], pos_plot[:,2], 'b-'); ax1.set_title('3D Trajectory (Z-up)')
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
        def _lims(a):
            mn, mx = float(np.min(a)), float(np.max(a))
            if not np.isfinite(mn) or not np.isfinite(mx):
                return (-1.0, 1.0)
            if abs(mx - mn) < 1e-6:
                pad = max(1e-3, abs(mx) * 0.05 + 1e-3); return (mn - pad, mx + pad)
            pad = 0.05 * (mx - mn); return (mn - pad, mx + pad)
        ax1.set_xlim(*_lims(pos_plot[:,0])); ax1.set_ylim(*_lims(pos_plot[:,1])); ax1.set_zlim(*_lims(pos_plot[:,2]))
        ax2 = axes[0,1]
        ax2.plot(times, pos_plot[:,0], 'r-', label='X')
        ax2.plot(times, pos_plot[:,1], 'g-', label='Y')
        ax2.plot(times, pos_plot[:,2], 'b-', label='Z↑')
        ax2.legend(); ax2.set_title('Position vs Time'); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Position (m)')
        ax2.set_xlim(times[0], times[-1]); ax2.set_ylim(*_lims(pos_plot))
        ax3 = axes[1,0]
        ax3.plot(times, vel_plot[:,0], 'r-', label='Vx')
        ax3.plot(times, vel_plot[:,1], 'g-', label='Vy')
        ax3.plot(times, vel_plot[:,2], 'b-', label='Vz↑')
        ax3.legend(); ax3.set_title('Velocity vs Time'); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Velocity (m/s)')
        ax3.set_xlim(times[0], times[-1]); ax3.set_ylim(*_lims(vel_plot))
        axes[1,1].axis('off')
        plt.tight_layout()
        if save_plot:
            os.makedirs('Output', exist_ok=True)
            plt.savefig(os.path.join('Output','theta_simulation_full.png'), dpi=300, bbox_inches='tight')
        return fig

    def animate_trajectory(self, states, times, fps=20, save_gif=True):
        # Why: show only 3D animated trajectory (Z-up); What: single 3D axis
        pos = states[:, :3]
        pos_plot = pos.copy(); pos_plot[:, 2] = -pos_plot[:, 2]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        line3d, = ax.plot([], [], [], 'b-', lw=2, alpha=0.9)
        point3d, = ax.plot([], [], [], 'go', ms=6)
        ax.set_title('3D Drone Trajectory (Animated, Z-up)'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        def _lims(a):
            mn, mx = float(np.min(a)), float(np.max(a))
            if not np.isfinite(mn) or not np.isfinite(mx):
                return (-1.0, 1.0)
            if abs(mx - mn) < 1e-6:
                pad = max(1e-3, abs(mx) * 0.05 + 1e-3); return (mn - pad, mx + pad)
            pad = 0.05 * (mx - mn); return (mn - pad, mx + pad)
        ax.set_xlim(*_lims(pos_plot[:,0])); ax.set_ylim(*_lims(pos_plot[:,1])); ax.set_zlim(*_lims(pos_plot[:,2]))

        def update(i):
            line3d.set_data(pos_plot[:i+1,0], pos_plot[:i+1,1]); line3d.set_3d_properties(pos_plot[:i+1,2])
            point3d.set_data([pos_plot[i,0]], [pos_plot[i,1]]); point3d.set_3d_properties([pos_plot[i,2]])
            return line3d, point3d

        anim = FuncAnimation(fig, update, frames=len(times), interval=1000.0/fps, blit=False)
        if save_gif:
            os.makedirs('Output', exist_ok=True)
            gif_path = os.path.join('Output','drone_animation.gif')
            anim.save(gif_path, writer='pillow', fps=fps)
        # why: display live simulation; what: show Matplotlib animation window
        plt.show()
        return anim

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__" and os.getenv("EMMA_AUTOMATE_MODE") != "1":
    """
    Main execution script for drone parameter estimation.
    
    This script:
    1. Parses command line arguments
    2. Loads drone trajectory data
    3. Creates and trains the LTC neural network
    4. Estimates 12 physical drone parameters
    5. Saves results to CSV file
    
    Usage:
        python3 EMMA_drone_torch_ltc_optimized.py --epochs 5 --size 32 --model ltc --log 1
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Drone Parameter Estimation using LTC Networks")
    parser.add_argument('--model', default="ltc", help="Model type (ltc, lstm, etc.)")
    parser.add_argument('--log', default=1, type=int, help="Logging frequency")
    parser.add_argument('--size', default=32, type=int, help="Hidden layer size")
    parser.add_argument('--epochs', default=3, type=int, help="Number of training epochs")
    parser.add_argument('--id', default=1, type=int, help="Experiment ID")
    args = parser.parse_args()

    # Set experiment ID
    numID = args.id
    
    # Load drone trajectory data
    print("Loading drone trajectory data...")
    har_data = HarData()
    
    # Create neural network model
    print(f"Creating {args.model} model with size {args.size}...")
    model = HarModel(model_type=args.model, model_size=args.size)
    
    # Train the model to estimate drone parameters
    print("Starting training...")
    model.fit(har_data, epochs=args.epochs, log_period=args.log)
    
    print("Training completed!")

    # ----------------------------
    # THETA CONVERSION + QUICK SIMULATION
    # ----------------------------
    try:
        # --- THETA CONVERSION ---
        # Why: convert normalized network outputs to physical parameters; What: compute mean over test window
        with torch.no_grad():
            final_pred = model.forward(har_data.test_x_z)  # [T,B,12]
            maxChange = 95.0
            getp = lambda k: final_pred[:,:,k].mean()
            dxm = (1 + (0.5 - getp(0)) * maxChange / 100.0) * 0.16
            dym = (1 + (0.5 - getp(1)) * maxChange / 100.0) * 0.16
            dzm = (1 + (0.5 - getp(2)) * maxChange / 100.0) * 0.05
            IBxx = (1 + (0.5 - getp(3)) * maxChange / 100.0) * 0.0123
            IByy = (1 + (0.5 - getp(4)) * maxChange / 100.0) * 0.0123
            IBzz = (1 + (0.5 - getp(5)) * maxChange / 100.0) * 0.0123
            Cd = (1 + (0.5 - getp(6)) * maxChange / 100.0) * 0.1
            kTh = (1 + (0.5 - getp(7)) * maxChange / 100.0) * 1.076e-5
            kTo = (1 + (0.5 - getp(8)) * maxChange / 100.0) * 1.632e-7
            tau = (1 + (0.5 - getp(9)) * maxChange / 100.0) * 0.015
            kp = (1 + (0.5 - getp(10)) * maxChange / 100.0) * 1.0
            damp = (1 + (0.5 - getp(11)) * maxChange / 100.0) * 1.0

        theta = {
            'dxm': float(dxm), 'dym': float(dym), 'dzm': float(dzm),
            'IBxx': float(IBxx), 'IByy': float(IByy), 'IBzz': float(IBzz),
            'Cd': float(Cd), 'kTh': float(kTh), 'kTo': float(kTo),
            'tau2': float(tau), 'kp': float(kp), 'damp': float(damp)
        }
        print("\n=== THETA (converted coefficients) ===")
        for k,v in theta.items():
            print(f"{k}: {v:.6g}")

        # Quick 1D vertical simulation using data motor inputs (first sequence)
        # Why: sanity-check thetas produce plausible z trajectory under recorded inputs
        T = har_data.test_uMotor1.shape[0]
        u1 = har_data.test_uMotor1[:, 0, 0].cpu().numpy()
        u2 = har_data.test_uMotor2[:, 0, 0].cpu().numpy()
        u3 = har_data.test_uMotor3[:, 0, 0].cpu().numpy()
        u4 = har_data.test_uMotor4[:, 0, 0].cpu().numpy()
        mB, g = 1.2, 9.81
        dt = 0.005
        z = 0.0; vz = 0.0
        zs = [z]
        for i in range(1, T):
            # thrust ~ kTh * sum(w^2); here we proxy w by input magnitude
            thrust = theta['kTh'] * (u1[i]**2 + u2[i]**2 + u3[i]**2 + u4[i]**2)
            drag = -theta['Cd'] * abs(vz) * vz
            az = (thrust + drag - mB * g) / mB
            vz = vz + az * dt
            z = z + vz * dt
            zs.append(z)

        # why: user wants only live simulation; what: skip quick static plot
        SHOW_ONLY_SIM = True
        if not SHOW_ONLY_SIM:
            plt.figure(figsize=(8,4))
            plt.plot(np.arange(T)*dt, -np.array(zs), label='Sim z (quick, Z-up)')
            plt.xlabel('Time (s)'); plt.ylabel('Z (m)'); plt.title('Quick Theta Simulation (Z-up)')
            plt.grid(True); plt.legend();
            os.makedirs('Output', exist_ok=True)
            out_plot = os.path.join('Output','theta_simulation_quick.png')
            plt.tight_layout(); plt.savefig(out_plot, dpi=200)
            print(f"Saved quick theta simulation plot to {out_plot}")

        # --- SIMULATION ---
        # Why: simulate with theta to validate dynamics; What: static plot + GIF into Output/
        simulator = ThetaSimulator(theta)
        # Build control input matrix from data (first 300 steps for speed)
        u_len = min(300, har_data.test_uMotor1.shape[0])
        control_inputs = np.column_stack([
            har_data.test_uMotor1[:u_len, 0, 0].cpu().numpy(),
            har_data.test_uMotor2[:u_len, 0, 0].cpu().numpy(),
            har_data.test_uMotor3[:u_len, 0, 0].cpu().numpy(),
            har_data.test_uMotor4[:u_len, 0, 0].cpu().numpy(),
        ])
        initial_state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        states, times = simulator.simulate_trajectory(initial_state, control_inputs, dt=0.005, duration=u_len*0.005)
        # why: show only live animation; what: skip static trajectory plot
        if not SHOW_ONLY_SIM:
            simulator.plot_trajectory(states, times, save_plot=True)
        simulator.animate_trajectory(states, times, fps=20, save_gif=True)
        print("Saved full theta simulation to Output/theta_simulation_full.png and Output/drone_animation.gif")
    except Exception as e:
        print(f"Theta conversion/simulation skipped due to error: {e}")



#RUN AS: python3 EMMA_drone_torch_ltc.py --epochs 5 --size 32 --model ltc --log 1 --id 1  # batch size 32

# EMMA Drone Pipeline

# Orchestrator: runs extract -> audio -> EMMA using new data in this folder
if __name__ == "__main__" and os.getenv("EMMA_RUN_ORCHESTRATOR", "0") == "1":
    try:
        # why: allow one-command automation; what: run steps with minimal args
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(here))  # project root
        dp = os.path.join(root, "Dronepipeline")
        weights = os.path.join(root, "yolov8n.pt")
        data_dir = os.path.join(here, "data")
        os.makedirs(data_dir, exist_ok=True)
        os.environ["EMMA_AUTOMATE_MODE"] = "1"

        # pick video (CLI: --video PATH) else default
        vid = None
        for i, a in enumerate(sys.argv):
            if a == "--video" and i+1 < len(sys.argv):
                vid = sys.argv[i+1]
                break
        if not vid:
            vid = os.path.join(here, "DroneVideo.mp4")
        if not os.path.isabs(vid):
            vid = os.path.abspath(os.path.join(os.getcwd(), vid))
        if not os.path.exists(vid):
            raise SystemExit(f"Video not found: {vid}")
        if not os.path.exists(weights):
            raise SystemExit(f"Weights not found: {weights}")

        # Fallback: if external Dronepipeline is missing, run local vision-only pipeline
        # why: keep run working without external scripts; what: detect, track, export x/y/z
        if not (os.path.isdir(dp) and os.path.exists(os.path.join(dp, "droneExtract.py"))):
            out_video = os.path.join(here, "annotated.mp4")
            out_csv = os.path.join(data_dir, "trajectory.csv")
            cap = cv2.VideoCapture(vid)
            if not cap.isOpened():
                raise SystemExit(f"Cannot open video: {vid}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            # use class-agnostic detector + simple size-based depth for stability
            det = Detector(weights, conf=0.25, imgsz=960)
            depth = DepthEstimator()
            kf = Kalman3D(dt=1.0/float(fps))

            xs, ys, zs = [], [], []
            with open(out_csv, "w", newline="") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(["frame","time_s","x","y","z","conf"])
                frame_idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    t = frame_idx / float(fps)
                    detres = det.detect(frame)
                    if detres is not None:
                        x1,y1,x2,y2,conf = detres
                        cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
                        z = depth.z_from_bbox((x1,y1,x2,y2))
                        kf.predict(); kf.update(np.array([cx, cy, z], dtype=float))
                        s = kf.x.squeeze(); xk, yk, zk = float(s[0]), float(s[1]), float(s[2])
                        xs.append(xk); ys.append(yk); zs.append(zk)
                        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                        cv2.circle(frame, (int(xk), int(yk)), 5, (0,0,255), -1)
                        cv2.putText(frame, f"x={xk:.1f} y={yk:.1f} z={zk:.3f}", (int(x1), max(20,int(y1)-8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                        wcsv.writerow([frame_idx, f"{t:.3f}", f"{xk:.2f}", f"{yk:.2f}", f"{zk:.5f}", f"{conf:.3f}"])
                    else:
                        s = kf.predict().squeeze(); xk, yk, zk = float(s[0]), float(s[1]), float(s[2])
                        xs.append(xk); ys.append(yk); zs.append(zk)
                        cv2.circle(frame, (int(xk), int(yk)), 5, (0,255,255), -1)
                        cv2.putText(frame, f"(pred) x={xk:.1f} y={yk:.1f} z={zk:.3f}", (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2, cv2.LINE_AA)
                        wcsv.writerow([frame_idx, f"{t:.3f}", f"{xk:.2f}", f"{yk:.2f}", f"{zk:.5f}", "NaN"])
                    writer.write(frame)
                    frame_idx += 1
            writer.release(); cap.release()

            # export x/y/z as N x 1701, matching expected orientation (transpose of tiled rows)
            if len(xs) > 0:
                row_x = np.asarray(xs, dtype=float); row_y = np.asarray(ys, dtype=float); row_z = np.asarray(zs, dtype=float)
                Mx = np.tile(row_x, (1701,1)); My = np.tile(row_y, (1701,1)); Mz = np.tile(row_z, (1701,1))
                np.savetxt(os.path.join(data_dir, "xData.txt"), Mx.T, fmt='%.6f')
                np.savetxt(os.path.join(data_dir, "yData.txt"), My.T, fmt='%.6f')
                np.savetxt(os.path.join(data_dir, "zData.txt"), Mz.T, fmt='%.6f')

            # ensure min/max motor bounds exist for downstream steps
            x_path = os.path.join(data_dir, "xData.txt")
            rows = sum(1 for _ in open(x_path)) if os.path.exists(x_path) else 0
            cols = 1701
            min_path = os.path.join(data_dir, "minMotor.txt")
            max_path = os.path.join(data_dir, "maxMotor.txt")
            if rows > 0:
                if not os.path.exists(min_path):
                    np.savetxt(min_path, np.zeros((rows, cols)), fmt='%.6f')
                if not os.path.exists(max_path):
                    np.savetxt(max_path, np.full((rows, cols), 2000.0), fmt='%.6f')

            print("[DONE] Fallback wrote:", out_video, "and", out_csv)

            # --- Audio fallback: extract rotor audio + uMotor files ---
            try:
                from moviepy.editor import VideoFileClip  # may raise if not installed
                tmp_wav = os.path.join(here, "tmp_audio.wav")
                clip = VideoFileClip(vid)
                if clip.audio is None:
                    raise RuntimeError("No audio track in video.")
                clip.audio.write_audiofile(tmp_wav, fps=44100)

                # compute features
                import librosa
                y, sr = librosa.load(tmp_wav, sr=22050, mono=True)
                n_fft, hop = 2048, 512
                S = (np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))**2)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                lo, hi = np.searchsorted(freqs, 80), np.searchsorted(freqs, 3000)
                band = S[lo:hi, :]
                peak_bin = np.argmax(band, axis=0)
                peak_freq = freqs[lo + peak_bin]
                times = librosa.frames_to_time(np.arange(peak_freq.shape[0]), sr=sr, hop_length=hop)

                # helper to interp peak at time t
                def peak_at(t):
                    return float(np.interp(t, times, peak_freq))

                # annotate audio video + CSV aligned to frames
                out_video_a = os.path.join(here, "annotated_audio.mp4")
                out_csv_a = os.path.join(data_dir, "trajectory_with_audio.csv")
                cap2 = cv2.VideoCapture(vid)
                fps2 = cap2.get(cv2.CAP_PROP_FPS) or fps
                width2, height2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer2 = cv2.VideoWriter(out_video_a, cv2.VideoWriter_fourcc(*"mp4v"), fps2, (width2, height2))
                det2 = Detector(weights, conf=0.25, imgsz=960)
                depth2 = DepthEstimator()
                kf2 = Kalman3D(dt=1.0/float(fps2))
                peak_series = []
                with open(out_csv_a, "w", newline="") as fa:
                    wcsv = csv.writer(fa)
                    wcsv.writerow(["frame","time_s","x","y","z","conf","audio_rms","centroid_hz","peak_hz","est_rpm"])
                    idx = 0
                    while True:
                        ok, frame = cap2.read()
                        if not ok: break
                        t = idx/float(fps2)
                        detres = det2.detect(frame)
                        if detres is not None:
                            x1,y1,x2,y2,conf = detres
                            cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
                            z=depth2.z_from_bbox((x1,y1,x2,y2))
                            kf2.predict(); kf2.update(np.array([cx,cy,z],dtype=float))
                            s=kf2.x.squeeze(); xk, yk, zk = float(s[0]), float(s[1]), float(s[2])
                            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                            cv2.circle(frame,(int(xk),int(yk)),5,(0,0,255),-1)
                        else:
                            s=kf2.predict().squeeze(); xk, yk, zk = float(s[0]), float(s[1]), float(s[2])
                            conf = float('nan')
                            cv2.circle(frame,(int(xk),int(yk)),5,(0,255,255),-1)
                        peak = peak_at(t)
                        rpm = peak*60.0/4.0
                        peak_series.append(peak if np.isfinite(peak) and peak>0 else np.nan)
                        wcsv.writerow([idx, f"{t:.3f}", f"{xk:.2f}", f"{yk:.2f}", f"{zk:.5f}", f"{conf:.3f}" if np.isfinite(conf) else "NaN",
                                       "NaN","NaN", f"{peak:.2f}", f"{rpm:.2f}"])
                        writer2.write(frame); idx+=1
                writer2.release(); cap2.release()
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

                # uMotor1..4 from scaled peak
                if len(peak_series)>0:
                    arr = np.asarray(peak_series, dtype=float)
                    valid = arr[np.isfinite(arr) & (arr>0)]
                    if valid.size>0:
                        min_peak = float(np.min(valid))
                        scaled = arr * (75.0/min_peak)
                        scaled[~(np.isfinite(arr) & (arr>0))] = 75.0
                        M = np.tile(scaled.reshape(-1),(1701,1)).T
                        np.savetxt(os.path.join(data_dir, "uMotor1.txt"), M, fmt='%.6f')
                        np.savetxt(os.path.join(data_dir, "uMotor2.txt"), M, fmt='%.6f')
                        np.savetxt(os.path.join(data_dir, "uMotor3.txt"), M, fmt='%.6f')
                        np.savetxt(os.path.join(data_dir, "uMotor4.txt"), M, fmt='%.6f')
                print("[DONE] Fallback audio wrote:", out_video_a, "and", out_csv_a)
            except Exception as e:
                print("[WARN] Audio fallback skipped:", e)

            # --- EMMA fallback: train quick LTC model on extracted data ---
            try:
                print("[INFO] Running EMMA fallback (epochs=1)...")
                har_data = HarData()
                model = HarModel(model_type="ltc", model_size=32)
                model.fit(har_data, epochs=1, log_period=1)
                print("[DONE] EMMA fallback finished")
            except Exception as e:
                print("[WARN] EMMA fallback skipped:", e)

            sys.exit(0)

        # 1) Vision extract → trajectory.csv and x/y/z into data_dir
        out_video = os.path.join(here, "annotated.mp4")
        out_csv = os.path.join(data_dir, "trajectory.csv")
        subprocess.run([sys.executable, os.path.join(dp, "droneExtract.py"),
                        "--video", vid, "--weights", weights,
                        "--out_video", out_video, "--out_csv", out_csv], check=True)

        # 2) Audio extract → motors into data_dir
        out_video_a = os.path.join(here, "annotated_audio.mp4")
        out_csv_a = os.path.join(data_dir, "trajectory_with_audio.csv")
        subprocess.run([sys.executable, os.path.join(dp, "droneExtractAudio.py"),
                        "--video", vid, "--weights", weights,
                        "--out_video", out_video_a, "--out_csv", out_csv_a], check=True)

        # 3) Ensure min/max motors exist in data_dir
        import numpy as _np
        x_path = os.path.join(data_dir, "xData.txt")
        rows = sum(1 for _ in open(x_path)) if os.path.exists(x_path) else 0
        cols = 1701
        min_path = os.path.join(data_dir, "minMotor.txt")
        max_path = os.path.join(data_dir, "maxMotor.txt")
        if rows > 0:
            if not os.path.exists(min_path):
                _np.savetxt(min_path, _np.zeros((rows, cols)), fmt='%.6f')
            if not os.path.exists(max_path):
                _np.savetxt(max_path, _np.full((rows, cols), 2000.0), fmt='%.6f')

        # 4) Copy matrices into Dronepipeline/data for EMMA script
        dp_data = os.path.join(dp, "data")
        os.makedirs(dp_data, exist_ok=True)
        for name in ("xData.txt","yData.txt","zData.txt","uMotor1.txt","uMotor2.txt","uMotor3.txt","uMotor4.txt","minMotor.txt","maxMotor.txt"):
            src = os.path.join(data_dir, name)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(dp_data, name))

        # 5) Run EMMA optimized (uses Dronepipeline/data now containing new data)
        subprocess.run([sys.executable, os.path.join(dp, "EMMA_drone_torch_ltc_optimized.py"),
                        "--epochs", "1", "--size", "32", "--model", "ltc", "--log", "1"], check=True, cwd=root)

        # 6) Copy outputs back here for convenience
        for fn in ("outputV2GPUV3_torch_ltc.csv",):
            src = os.path.join(root, fn)
            if os.path.exists(src): shutil.copyfile(src, os.path.join(here, fn))
        for fn in ("theta_simulation_quick.png","theta_simulation_full.png","drone_animation.gif"):
            src = os.path.join(root, "Output", fn)
            if os.path.exists(src): shutil.copyfile(src, os.path.join(here, fn))
        print("[DONE] All outputs saved under:", here)
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        print("[ERROR] Orchestrator failed:", e)
        sys.exit(1)

# RUN AS:
# EMMA_AUTOMATE_MODE=1 EMMA_RUN_ORCHESTRATOR=1 python3 new_run.py --video DroneVideo.mp4 --weights yolo11n.pt && EMMA_RUN_ORCHESTRATOR=0 MPLBACKEND=TkAgg python3 new_run.py --epochs 1 --size 32 --model ltc --log 1