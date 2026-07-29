"""
Smart Traffic Management System (YOLO11 version) - 30-SECOND RESET & HISTORY EDITION
-------------------------------------------------------------------------------------------
Features:
  1. MANUAL LINE SELECTION -> Mouse se 2 points pick karke boundary set hoti hai.
  2. IN / OUT DIRECTION -> Automatic normal-based direction mapping.
  3. BOX-TOUCH COUNTING -> Bounding box intersection se zero-lag detection.
  4. 30-SECOND RESET -> Har 30 sec baad current count RESET ho kar 0 ho jata hai.
  5. WINDOW HISTORY & FINAL REPORT -> Har 30 sec ka result save hota hai aur video ke end 
     mein HAR FRAME/WINDOW ka detailed summary ek saath print hota hai.
"""

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# 1. CONFIGURATION
# =========================================================

class Config:
    MODEL_PATH = "yolo11n.pt"

    # --- YOLO11 inference params ---
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    INFER_SIZE = 640

    # --- COCO class ids (id -> readable name) ---
    VEHICLE_CLASS_IDS = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    # --- Tracker config ---
    TRACKER = "bytetrack.yaml"
    
    # --- 30 Seconds Window Config ---
    INTERVAL_SECONDS = 30


# =========================================================
# 2. GEOMETRY HELPERS
# =========================================================

def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d):
    return (_ccw(a, c, d) != _ccw(b, c, d)) and (_ccw(a, b, c) != _ccw(a, b, d))


def _point_in_rect(pt, x1, y1, x2, y2):
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


def line_intersects_box(p1, p2, x1, y1, x2, y2):
    if _point_in_rect(p1, x1, y1, x2, y2) or _point_in_rect(p2, x1, y1, x2, y2):
        return True

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    for c, d in edges:
        if _segments_intersect(p1, p2, c, d):
            return True
    return False


# =========================================================
# 3. LINE SELECTOR
# =========================================================

class LineSelector:
    def __init__(self):
        self.window_name = "Select LINE Boundary: Left-Click 2 points | R=Reset | ENTER=Confirm | Q=Quit"
        self.points = []
        self.frame = None
        self.display_frame = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 2:
            self.points.append((x, y))
            self._redraw()

    def _redraw(self):
        self.display_frame = self.frame.copy()
        for pt in self.points:
            cv2.circle(self.display_frame, pt, 6, (0, 0, 255), -1)
        if len(self.points) == 2:
            cv2.line(self.display_frame, self.points[0], self.points[1], (0, 255, 255), 3)
        cv2.imshow(self.window_name, self.display_frame)

    def _pick_points(self, frame):
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.points = []

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.imshow(self.window_name, self.display_frame)

        print("\n[LINE SELECTION] Frame par LEFT CLICK karke 2 points banao.")
        print("[LINE SELECTION] Confirm karne ke liye ENTER dabao.")
        print("[LINE SELECTION] Galti ho jaaye to 'r' dabao reset karne ke liye.\n")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                if len(self.points) == 2:
                    break
                else:
                    print("Exactly 2 points select karo (abhi tak: %d)!" % len(self.points))
            elif key == ord('r'):
                self.points = []
                self._redraw()
            elif key == ord('q'):
                cv2.destroyWindow(self.window_name)
                raise SystemExit("Line selection cancelled by user.")

        cv2.destroyWindow(self.window_name)
        p1, p2 = self.points[0], self.points[1]
        print(f"[LINE SELECTION] Line confirm ho gayi: {p1} -> {p2}\n")
        return p1, p2

    def select(self, frame):
        p1, p2 = self._pick_points(frame)
        pos_label, neg_label = "IN", "OUT"
        return p1, p2, pos_label, neg_label


# =========================================================
# 4. BOX-TOUCH VEHICLE COUNTER WITH RESET & HISTORY
# =========================================================

class BoxTouchCounter:
    def __init__(self, p1, p2, vehicle_class_ids: dict, pos_label: str, neg_label: str):
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.p1_t = tuple(self.p1)
        self.p2_t = tuple(self.p2)
        self.vehicle_class_ids = vehicle_class_ids

        line_vec = self.p2 - self.p1
        self.normal = np.array([-line_vec[1], line_vec[0]])

        self.dir_pos = pos_label
        self.dir_neg = neg_label

        self.prev_centroid = {}
        
        # Har current 30s window ke liye ID tracker aur counter (Reset hone par clearance ke liye)
        self.current_window_ids = set()
        self.current_class_counts = {name: 0 for name in vehicle_class_ids.values()}
        self.current_direction_counts = {self.dir_pos: 0, self.dir_neg: 0}
        
        # History store karne ke liye (End output ke liye)
        self.history = []

    def update(self, track_id, class_id, x1, y1, x2, y2, cx, cy):
        result = None

        # Agar vehicle iss current 30s window mein count nahi hua hai
        if track_id not in self.current_window_ids and class_id in self.vehicle_class_ids:
            touching = line_intersects_box(self.p1_t, self.p2_t, x1, y1, x2, y2)

            if touching:
                prev = self.prev_centroid.get(track_id)
                if prev is not None:
                    disp = np.array([cx - prev[0], cy - prev[1]])
                    proj = float(np.dot(disp, self.normal))
                else:
                    vec = np.array([cx, cy]) - self.p1
                    proj = float(np.dot(vec, self.normal))

                direction = self.dir_pos if proj >= 0 else self.dir_neg
                class_name = self.vehicle_class_ids[class_id]

                self.current_window_ids.add(track_id)
                self.current_class_counts[class_name] += 1
                self.current_direction_counts[direction] += 1
                
                result = (class_name, direction)

        self.prev_centroid[track_id] = (cx, cy)
        return result

    def reset_and_save_window(self, window_idx: int, time_label: str):
        """30 seconds poore hone par purana record save karta hai aur counts ko 0 karta hai."""
        total_veh = sum(self.current_class_counts.values())
        
        # Traffic Density Classification
        if total_veh < 10:
            density = "LOW"
        elif total_veh <= 25:
            density = "MEDIUM"
        else:
            density = "HIGH"

        # Record save karo history list mein
        record = {
            "window_idx": window_idx,
            "time_label": time_label,
            "total": total_veh,
            "density": density,
            "class_counts": self.current_class_counts.copy(),
            "direction_counts": self.current_direction_counts.copy()
        }
        self.history.append(record)

        # COUNT RESET TO ZERO (Agli 30s Window ke liye)
        self.current_window_ids.clear()
        self.current_class_counts = {name: 0 for name in self.vehicle_class_ids.values()}
        self.current_direction_counts = {self.dir_pos: 0, self.dir_neg: 0}
        
        return record

    def current_total(self):
        return sum(self.current_class_counts.values())


# =========================================================
# 5. DRAWING HELPERS
# =========================================================

CLASS_COLORS = {
    "car": (0, 255, 0),
    "bus": (255, 200, 0),
    "truck": (0, 165, 255),
    "motorcycle": (255, 0, 255),
    "bicycle": (177, 33, 123),
}


def draw_line(frame, counter: BoxTouchCounter):
    p1 = tuple(counter.p1.astype(int))
    p2 = tuple(counter.p2.astype(int))
    cv2.line(frame, p1, p2, (0, 255, 255), 3)

    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    norm_unit = counter.normal / (np.linalg.norm(counter.normal) + 1e-6)
    arrow_end = (int(mid[0] + norm_unit[0] * 40), int(mid[1] + norm_unit[1] * 40))
    cv2.arrowedLine(frame, mid, arrow_end, (255, 0, 0), 2, tipLength=0.3)
    cv2.putText(frame, counter.dir_pos, arrow_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame


def draw_detection(frame, x1, y1, x2, y2, track_id, class_name, conf, counted):
    color = CLASS_COLORS.get(class_name, (200, 200, 200))
    if counted:
        color = (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id} {class_name.upper()} {conf:.2f}"
    cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, color, -1)
    return frame


def draw_counts_panel(frame, counter: BoxTouchCounter, window_idx: int, sec_remaining: float):
    last_record = counter.history[-1] if len(counter.history) > 0 else None
    
    lines = [
        f"--- CURRENT WINDOW (#{window_idx}) ---",
        f"30s Count (Resets in {sec_remaining:.1f}s): {counter.current_total()}",
        f"IN : {counter.current_direction_counts.get('IN', 0)} | OUT: {counter.current_direction_counts.get('OUT', 0)}",
    ]

    if last_record:
        lines.append("--- PREVIOUS 30s SUMMARY ---")
        lines.append(f"Window #{last_record['window_idx']} Total: {last_record['total']} ({last_record['density']} Traffic)")

    h = 35 + 22 * len(lines)
    cv2.rectangle(frame, (10, 10), (430, 10 + h), (0, 0, 0), -1)
    y = 30
    for line in lines:
        if "CURRENT" in line or "PREVIOUS" in line:
            color = (0, 255, 255)
        elif "Traffic" in line:
            color = (0, 0, 255) if "HIGH" in line else (0, 255, 0)
        else:
            color = (255, 255, 255)

        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        y += 22
    return frame


# =========================================================
# 6. VIDEO SOURCE INPUT HELPER
# =========================================================

def get_video_source():
    print("=" * 60)
    print(" SMART TRAFFIC MANAGEMENT SYSTEM ")
    print("=" * 60)
    print(" Examples:")
    print("   - Mobile IP Webcam: http://192.168.1.15:8080/video")
    print("   - Laptop Webcam   : 0")
    print("   - Video File      : demo.mp4")
    print("=" * 60)

    user_input = input("\nInput enter karo: ").strip().strip('"').strip("'")

    if user_input == "0":
        return 0

    if user_input.startswith("http://") and not user_input.endswith("/video"):
        if not user_input.endswith("/"):
            user_input += "/"
        user_input += "video"

    if user_input.lower().startswith(("rtsp://", "http://", "https://", "rtmp://")):
        return user_input

    if not os.path.isfile(user_input):
        raise FileNotFoundError(f"Source file ya link invalid hai: {user_input}")

    return user_input


# =========================================================
# 7. MAIN PIPELINE
# =========================================================

def main():
    cfg = Config()
    model = YOLO(cfg.MODEL_PATH)

    video_source = get_video_source()
    is_live_stream = isinstance(video_source, str) and video_source.lower().startswith(
        ("rtsp://", "http://", "https://", "rtmp://")
    )

    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG if is_live_stream else cv2.CAP_ANY)

    if is_live_stream:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise IOError(f"Video source connect nahi ho saka: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback assumption for live feeds

    frames_per_window = int(fps * cfg.INTERVAL_SECONDS)

    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Stream/video se pehla frame nahi mila.")

    selector = LineSelector()
    p1, p2, pos_label, neg_label = selector.select(first_frame)

    counter = BoxTouchCounter(p1, p2, cfg.VEHICLE_CLASS_IDS, pos_label, neg_label)

    if isinstance(video_source, str) and not is_live_stream:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    class_ids_list = list(cfg.VEHICLE_CLASS_IDS.keys())
    cv2.namedWindow("Smart Traffic Management (30s Reset)", cv2.WINDOW_NORMAL)

    frame_counter = 0
    window_index = 1
    window_start_time = time.time()

    while True:
        if is_live_stream:
            cap.grab()
            ret, frame = cap.retrieve()
        else:
            ret, frame = cap.read()

        if not ret:
            # Agar last window adhuri reh gayi thi aur video khatam ho gaya, use bhi save kar lo
            if counter.current_total() > 0:
                time_lbl = f"Window {window_index} (End of Video)"
                counter.reset_and_save_window(window_index, time_lbl)
            print("\nEnd of video stream.")
            break

        frame_counter += 1

        # Check 30-Second Interval Expiry
        if is_live_stream:
            elapsed_time = time.time() - window_start_time
            is_window_over = elapsed_time >= cfg.INTERVAL_SECONDS
            sec_remaining = max(0.0, cfg.INTERVAL_SECONDS - elapsed_time)
        else:
            is_window_over = (frame_counter % frames_per_window == 0)
            sec_remaining = ((frames_per_window - (frame_counter % frames_per_window)) / fps)

        # HAR 30 SECONDS PAR COUNT RESET AUR RECORD SAVE HOT A HAI
        if is_window_over:
            start_sec = (window_index - 1) * 30
            end_sec = window_index * 30
            time_lbl = f"{start_sec}s - {end_sec}s"

            saved_record = counter.reset_and_save_window(window_index, time_lbl)

            print(f"\n==================================================")
            print(f" [30s WINDOW #{saved_record['window_idx']} COMPLETED ({time_lbl})]")
            print(f" Vehicles in this 30s : {saved_record['total']}")
            print(f" Traffic Density      : {saved_record['density']}")
            print(f" Vehicle Breakdown    : {saved_record['class_counts']}")
            print(f" Direction Breakdown  : {saved_record['direction_counts']}")
            print(f" --> COUNT RESET TO 0 FOR NEXT WINDOW <--")
            print(f"==================================================\n")

            window_index += 1
            window_start_time = time.time()

        results = model.track(
            frame,
            conf=cfg.CONF_THRESHOLD,
            iou=cfg.IOU_THRESHOLD,
            imgsz=cfg.INFER_SIZE,
            classes=class_ids_list,
            tracker=cfg.TRACKER,
            persist=True,
            verbose=False
        )

        result = results[0]
        frame = draw_line(frame, counter)

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = cfg.VEHICLE_CLASS_IDS.get(int(class_id), "unknown")

                touch_result = counter.update(int(track_id), int(class_id), x1, y1, x2, y2, cx, cy)
                if touch_result:
                    c_name, direction = touch_result
                    print(f"[WINDOW #{window_index}] ID {track_id} -> {c_name.upper()} crossed line | Dir: {direction}")

                already_counted = int(track_id) in counter.current_window_ids
                frame = draw_detection(frame, x1, y1, x2, y2, track_id, class_name, conf, already_counted)

        frame = draw_counts_panel(frame, counter, window_index, sec_remaining)

        cv2.imshow("Smart Traffic Management (30s Reset)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # =========================================================
    # ALL 30-SECOND WINDOWS FINAL COMBINED REPORT AT THE END
    # =========================================================
    print("\n" + "=" * 65)
    print("        FINAL REPORT: ALL 30-SECOND WINDOWS SUMMARY")
    print("=" * 65)
    
    if len(counter.history) == 0:
        print("Koi bhi 30-second window complete nahi hui.")
    else:
        grand_total = 0
        print(f"{'WINDOW':<12} | {'TIME FRAME':<14} | {'TOTAL VEHICLES':<15} | {'TRAFFIC DENSITY':<15}")
        print("-" * 65)
        
        for rec in counter.history:
            grand_total += rec['total']
            print(f"Window #{rec['window_idx']:<5} | {rec['time_label']:<14} | {rec['total']:<15} | {rec['density']:<15}")
            print(f"  └─ Vehicles  : {rec['class_counts']}")
            print(f"  └─ Directions: {rec['direction_counts']}")
            print("-" * 65)
            
        print(f"\nGRAND TOTAL VEHICLES ACROSS ALL WINDOWS: {grand_total}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()