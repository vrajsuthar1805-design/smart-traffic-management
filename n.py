"""
Smart Traffic Management System (YOLO11 version) - LINE Boundary Crossing Edition
-------------------------------------------------------------------------------------
Kya naya hai (updates from ROI version):
  1. Ab ROI POLYGON nahi, ek SINGLE LINE boundary hai (2 points click karke banao).
  2. YOLO11 built-in tracking (ByteTrack) se har vehicle ko UNIQUE ID milta hai.
  3. Jab vehicle ka CENTER OF MASS (centroid) line ko CROSS karta hai (ek side se
     doosre side), TABHI count hota hai.
  4. DIRECTION bhi automatically detect hoti hai -> vehicle "aa raha hai" ya
     "ja raha hai", based on line ke orientation par (LEFT_TO_RIGHT / RIGHT_TO_LEFT
     ya TOP_TO_BOTTOM / BOTTOM_TO_TOP).
  5. Ek vehicle sirf EK HI BAAR count hota hai (repeat nahi hoga), chahe wo
     line ke paas baar-baar hile-dule ya thoda side-to-side move kare.
  6. Terminal mein REAL-TIME print -> kaunsa vehicle, kis direction mein cross
     hua, aur ab tak ka running total (per class + per direction).

Requirements:
    pip install ultralytics opencv-python numpy

Model file:
    yolo11n.pt  -> pehli baar chalane par internet hone par khud download ho jaayegi
"""

import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# 1. CONFIGURATION
# =========================================================

class Config:
    MODEL_PATH = "yolo11n.pt"

    # --- Video source: apni video ka path yahan daalo, ya webcam ke liye 0 ---
    VIDEO_PATH = "Traffic Control CCTV.mp4"

    # --- YOLO11 inference params ---
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    INFER_SIZE = 640

    # --- COCO class ids jo humein chahiye (id -> readable name) ---
    VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    # --- Tracker config jo ultralytics ke saath already aata hai ---
    TRACKER = "bytetrack.yaml"


# =========================================================
# 2. LINE SELECTOR (mouse click based - exactly 2 points)
# =========================================================

class LineSelector:
    """
    Pehle frame par user ko mouse se ek LINE (2 points) banane deta hai.
    Left click -> point add (max 2 points)
    'r'        -> reset
    ENTER      -> confirm (exactly 2 points chahiye)
    'q'        -> cancel
    """

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

    def select(self, frame):
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.points = []

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.imshow(self.window_name, self.display_frame)

        print("\n[LINE SELECTION] Frame par LEFT CLICK karke 2 points banao (line ke dono ends).")
        print("[LINE SELECTION] Confirm karne ke liye ENTER dabao (exactly 2 points chahiye).")
        print("[LINE SELECTION] Galti ho jaaye to 'r' dabao reset karne ke liye. Cancel ke liye 'q'.\n")

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


# =========================================================
# 3. LINE-CROSSING VEHICLE COUNTER (centroid + tracking based)
# =========================================================

class LineCrossCounter:
    """
    Har track_id ka pichla 'side' (line ke relative) yaad rakhta hai.
    Jab side badalta hai (crossing hoti hai), vehicle ko count karta hai
    aur uski direction bhi decide karta hai. Ek baar count hone ke baad
    wahi track_id dobara kabhi count nahi hota.
    """

    def __init__(self, p1, p2, vehicle_class_ids: dict):
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.vehicle_class_ids = vehicle_class_ids

        # --- Normal vector nikalo (line ko 90 degree rotate karke) ---
        line_vec = self.p2 - self.p1
        self.normal = np.array([-line_vec[1], line_vec[0]])  # (nx, ny)

        # --- Normal ke dominant axis ke hisaab se human-friendly direction labels ---
        nx, ny = self.normal
        if abs(nx) >= abs(ny):
            if nx > 0:
                self.dir_pos, self.dir_neg = "LEFT_TO_RIGHT", "RIGHT_TO_LEFT"
            else:
                self.dir_pos, self.dir_neg = "RIGHT_TO_LEFT", "LEFT_TO_RIGHT"
        else:
            if ny > 0:
                self.dir_pos, self.dir_neg = "TOP_TO_BOTTOM", "BOTTOM_TO_TOP"
            else:
                self.dir_pos, self.dir_neg = "BOTTOM_TO_TOP", "TOP_TO_BOTTOM"

        self.prev_side = {}          # track_id -> -1 / +1 (last known side)
        self.counted_ids = set()     # track_ids jo already count ho chuke hain (never re-count)

        self.class_counts = {name: 0 for name in vehicle_class_ids.values()}
        self.direction_counts = {self.dir_pos: 0, self.dir_neg: 0}

    def _side(self, point):
        """Returns +1, -1, or 0 depending on which side of the line the point is on."""
        vec = np.array(point, dtype=np.float64) - self.p1
        d = np.dot(vec, self.normal)
        if d > 0:
            return 1
        elif d < 0:
            return -1
        return 0

    def update(self, track_id, class_id, cx, cy):
        """
        Returns (class_name, direction) if a NEW crossing just happened this frame,
        else None.
        """
        current_side = self._side((cx, cy))
        prev_side = self.prev_side.get(track_id)
        self.prev_side[track_id] = current_side if current_side != 0 else prev_side

        if track_id in self.counted_ids:
            return None
        if class_id not in self.vehicle_class_ids:
            return None
        if prev_side is None or current_side == 0 or prev_side == 0:
            return None
        if prev_side == current_side:
            return None

        # --- Crossing detected! ---
        self.counted_ids.add(track_id)
        class_name = self.vehicle_class_ids[class_id]
        direction = self.dir_pos if (prev_side < 0 and current_side > 0) else self.dir_neg

        self.class_counts[class_name] += 1
        self.direction_counts[direction] += 1
        return class_name, direction

    def total(self):
        return sum(self.class_counts.values())


# =========================================================
# 4. DRAWING HELPERS
# =========================================================

CLASS_COLORS = {
    "car": (0, 255, 0),
    "bus": (255, 200, 0),
    "truck": (0, 165, 255),
    "motorcycle": (255, 0, 255),
}


def draw_line(frame, counter: LineCrossCounter):
    p1 = tuple(counter.p1.astype(int))
    p2 = tuple(counter.p2.astype(int))
    cv2.line(frame, p1, p2, (0, 255, 255), 3)

    # --- Normal direction arrow (dikhata hai kaunsi side "dir_pos" hai) ---
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    norm_unit = counter.normal / (np.linalg.norm(counter.normal) + 1e-6)
    arrow_end = (int(mid[0] + norm_unit[0] * 40), int(mid[1] + norm_unit[1] * 40))
    cv2.arrowedLine(frame, mid, arrow_end, (255, 0, 0), 2, tipLength=0.3)
    cv2.putText(frame, counter.dir_pos, arrow_end, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    return frame


def draw_detection(frame, x1, y1, x2, y2, track_id, class_name, conf, crossed):
    color = CLASS_COLORS.get(class_name, (200, 200, 200))
    if crossed:
        color = (0, 0, 255)  # already-counted vehicles highlighted in red

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id} {class_name.upper()} {conf:.2f}"
    cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, color, -1)
    return frame


def draw_counts_panel(frame, counter: LineCrossCounter):
    lines = [f"Total Crossed: {counter.total()}"]
    for name, count in counter.class_counts.items():
        lines.append(f"{name.capitalize()}: {count}")
    lines.append(f"{counter.dir_pos}: {counter.direction_counts[counter.dir_pos]}")
    lines.append(f"{counter.dir_neg}: {counter.direction_counts[counter.dir_neg]}")

    h = 35 + 25 * len(lines)
    cv2.rectangle(frame, (10, 10), (360, 10 + h), (0, 0, 0), -1)
    y = 35
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1 if i else 2)
        y += 25
    return frame


# =========================================================
# 5. MAIN PIPELINE
# =========================================================

def main():
    cfg = Config()
    model = YOLO(cfg.MODEL_PATH)

    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {cfg.VIDEO_PATH}")

    # --- Pehla frame padho aur usi par LINE select karwao ---
    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Could not read the first frame from the video.")

    selector = LineSelector()
    p1, p2 = selector.select(first_frame)

    counter = LineCrossCounter(p1, p2, cfg.VEHICLE_CLASS_IDS)
    print(f"[INFO] Direction labels based on line orientation: "
          f"'{counter.dir_pos}' vs '{counter.dir_neg}'\n")

    # --- Video ko wapas shuru se chalao (kyunki pehla frame already consume ho chuka hai) ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    class_ids_list = list(cfg.VEHICLE_CLASS_IDS.keys())

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video stream.")
            break

        # --- YOLO11 tracking: detection + unique ID assignment ek hi call mein ---
        results = model.track(
            frame,
            conf=cfg.CONF_THRESHOLD,
            iou=cfg.IOU_THRESHOLD,
            imgsz=cfg.INFER_SIZE,
            classes=class_ids_list,
            tracker=cfg.TRACKER,
            persist=True,
            verbose=False
        )  # type: ignore

        result = results[0]  # type: ignore
        frame = draw_line(frame, counter)

        if result.boxes is not None and result.boxes.id is not None:  # type: ignore
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)       # type: ignore
            track_ids = result.boxes.id.cpu().numpy().astype(int)      # type: ignore
            class_ids = result.boxes.cls.cpu().numpy().astype(int)     # type: ignore
            confs = result.boxes.conf.cpu().numpy()                    # type: ignore

            for (x1, y1, x2, y2), track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = cfg.VEHICLE_CLASS_IDS.get(int(class_id), "unknown")

                crossing_result = counter.update(int(track_id), int(class_id), cx, cy)
                if crossing_result:
                    c_name, direction = crossing_result
                    print(f"[CROSS] ID {track_id} -> {c_name.upper()} crossed line "
                          f"| Direction: {direction} "
                          f"| Class totals: {counter.class_counts} "
                          f"| Direction totals: {counter.direction_counts} "
                          f"| Total: {counter.total()}")

                already_counted = int(track_id) in counter.counted_ids
                frame = draw_detection(frame, x1, y1, x2, y2, track_id, class_name, conf, already_counted)

        frame = draw_counts_panel(frame, counter)

        cv2.imshow("Smart Traffic Management (YOLO11) - Line Crossing Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n===== FINAL COUNTS =====")
    for name, count in counter.class_counts.items():
        print(f"{name.capitalize()}: {count}")
    print(f"{counter.dir_pos}: {counter.direction_counts[counter.dir_pos]}")
    print(f"{counter.dir_neg}: {counter.direction_counts[counter.dir_neg]}")
    print(f"Total Vehicles: {counter.total()}")
    print("=========================")


if __name__ == "__main__":
    main()