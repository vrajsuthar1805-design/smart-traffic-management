"""
Smart Traffic Management System (YOLO11 version) - MANUAL LINE + IN/OUT + BOX-TOUCH Edition
-------------------------------------------------------------------------------------------
Kya hai is version mein:

  1. MANUAL LINE SELECTION -> Auto-detection hata diya gaya hai. Ab pehle
     frame par tum khud mouse se LINE banaoge (2 points click karke).

  2. IN / OUT CONCEPT (auto-assigned) -> Koi extra key-press ya condition
     nahi. Line banate hi ek side automatically "IN" aur doosri "OUT" set ho
     jaati hai (line ke normal ki positive side = IN, opposite = OUT).

  3. BOX-TOUCH COUNTING (FIXED, jaisa tha waisa hi hai) -> Vehicle ka
     CENTROID line cross kare ye zaroori nahi. Jaise hi vehicle ka
     BOUNDING BOX line ko TOUCH/INTERSECT karta hai, turant count ho jaata
     hai. Ek track_id sirf EK HI BAAR count hota hai.

  4. USER VIDEO INPUT -> Program start hote hi terminal mein poochega ki
     konsi video file use karni hai. Webcam ke liye sirf '0' likh do.

  5. Terminal mein REAL-TIME print -> kaunsa vehicle, IN hua ya OUT, aur
     running total (per class + per direction).

Requirements:
    pip install ultralytics opencv-python numpy

Model file:
    yolo11n.pt  -> pehli baar chalane par internet hone par khud download ho jaayegi
"""

import os
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

    # --- COCO class ids jo humein chahiye (id -> readable name) ---
    VEHICLE_CLASS_IDS = {1 :"bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    # --- Tracker config jo ultralytics ke saath already aata hai ---
    TRACKER = "bytetrack.yaml"


# =========================================================
# 2. GEOMETRY HELPERS (line-vs-box intersection)
# =========================================================

def _ccw(a, b, c):
    """Counter-clockwise test — teen points ka orientation batata hai."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d):
    """Kya segment AB aur segment CD ek doosre ko cross karte hain?"""
    return (_ccw(a, c, d) != _ccw(b, c, d)) and (_ccw(a, b, c) != _ccw(a, b, d))


def _point_in_rect(pt, x1, y1, x2, y2):
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


def line_intersects_box(p1, p2, x1, y1, x2, y2):
    """
    Line segment (p1 -> p2) box (x1,y1,x2,y2) ko touch/intersect karta hai ya nahi.
    - Agar line ka koi endpoint box ke andar hai -> True
    - Agar line box ke kisi bhi edge ko cross karti hai -> True
    """
    if _point_in_rect(p1, x1, y1, x2, y2) or _point_in_rect(p2, x1, y1, x2, y2):
        return True

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    for c, d in edges:
        if _segments_intersect(p1, p2, c, d):
            return True
    return False


# =========================================================
# 3. LINE SELECTOR (mouse click based - exactly 2 points + IN/OUT labeling)
# =========================================================

class LineSelector:
    """
    Step 1: Pehle frame par user ko mouse se ek LINE (2 points) banane deta hai.
        Left click -> point add (max 2 points)
        'r'        -> reset
        ENTER      -> confirm (exactly 2 points chahiye)
        'q'        -> cancel

    Step 2: Line confirm hone ke baad direction labels automatically "IN" aur
        "OUT" assign ho jaate hain - koi extra key-press nahi karni. Line ke
        normal ki positive side "IN" aur opposite side "OUT" maani jaati hai.
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

    def _pick_points(self, frame):
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

    def select(self, frame):
        p1, p2 = self._pick_points(frame)
        # Direction labels ab automatically assign ho rahi hain - koi key-press
        # nahi karni. Line ke normal ki "positive" side ko "IN" aur opposite
        # side ko "OUT" maan liya jaata hai.
        pos_label, neg_label = "IN", "OUT"
        return p1, p2, pos_label, neg_label


# =========================================================
# 4. BOX-TOUCH VEHICLE COUNTER (bounding box line ko touch kare to count)
# =========================================================

class BoxTouchCounter:
    """
    Vehicle ka CENTROID line cross kare ye zaroori nahi. Jaise hi vehicle ka
    BOUNDING BOX line ko touch/intersect karta hai, turant count ho jaata hai
    (per track_id sirf ek baar). Direction "IN" / "OUT" labels user ne khud
    diye the line selection ke waqt.
    """

    def __init__(self, p1, p2, vehicle_class_ids: dict, pos_label: str, neg_label: str):
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.p1_t = tuple(self.p1)
        self.p2_t = tuple(self.p2)
        self.vehicle_class_ids = vehicle_class_ids

        line_vec = self.p2 - self.p1
        self.normal = np.array([-line_vec[1], line_vec[0]])

        # User-defined labels (IN / OUT), positive normal side = pos_label
        self.dir_pos = pos_label
        self.dir_neg = neg_label

        self.prev_centroid = {}     # track_id -> (cx, cy) pichle frame ka
        self.counted_ids = set()    # track_ids jo already count ho chuke hain

        self.class_counts = {name: 0 for name in vehicle_class_ids.values()}
        self.direction_counts = {self.dir_pos: 0, self.dir_neg: 0}

    def update(self, track_id, class_id, x1, y1, x2, y2, cx, cy):
        """
        Returns (class_name, direction) agar is frame mein NAYA touch/count
        hua ho, warna None.
        """
        result = None

        if track_id not in self.counted_ids and class_id in self.vehicle_class_ids:
            touching = line_intersects_box(self.p1_t, self.p2_t, x1, y1, x2, y2)

            if touching:
                prev = self.prev_centroid.get(track_id)
                if prev is not None:
                    disp = np.array([cx - prev[0], cy - prev[1]])
                    proj = float(np.dot(disp, self.normal))
                else:
                    # Pehla hi frame hai jisme track dikha aur turant touch bhi ho gaya
                    # -> centroid ki line ke relative side se direction guess karo
                    vec = np.array([cx, cy]) - self.p1
                    proj = float(np.dot(vec, self.normal))

                direction = self.dir_pos if proj >= 0 else self.dir_neg
                class_name = self.vehicle_class_ids[class_id]

                self.counted_ids.add(track_id)
                self.class_counts[class_name] += 1
                self.direction_counts[direction] += 1
                result = (class_name, direction)

        self.prev_centroid[track_id] = (cx, cy)
        return result

    def total(self):
        return sum(self.class_counts.values())


# =========================================================
# 5. DRAWING HELPERS
# =========================================================

CLASS_COLORS = {
    "car": (0, 255, 0),
    "bus": (255, 200, 0),
    "truck": (0, 165, 255),
    "motorcycle": (255, 0, 255),
    "bicycle" : (177, 33, 123),
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
        color = (0, 0, 255)  # already-counted vehicles highlighted in red

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id} {class_name.upper()} {conf:.2f}"
    cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, color, -1)
    return frame


def draw_counts_panel(frame, counter: BoxTouchCounter):
    lines = [f"Total Counted: {counter.total()}"]
    for name, count in counter.class_counts.items():
        lines.append(f"{name.capitalize()}: {count}")
    lines.append(f"IN : {counter.direction_counts.get('IN', 0)}")
    lines.append(f"OUT: {counter.direction_counts.get('OUT', 0)}")

    h = 35 + 25 * len(lines)
    cv2.rectangle(frame, (10, 10), (360, 10 + h), (0, 0, 0), -1)
    y = 35
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1 if i else 2)
        y += 25
    return frame


# =========================================================
# 6. VIDEO INPUT HELPER
# =========================================================

def get_video_source():
    """
    User se video file ka path (ya webcam ke liye '0') terminal mein poochta hai.
    """
    print("=" * 60)
    print(" SMART TRAFFIC MANAGEMENT SYSTEM ")
    print("=" * 60)
    user_input = input(
        "Video file ka full path daalo (webcam ke liye sirf 0 likho): "
    ).strip().strip('"').strip("'")

    if user_input == "0":
        return 0

    if not os.path.isfile(user_input):
        raise FileNotFoundError(f"Ye video file nahi mili: {user_input}")

    return user_input


# =========================================================
# 7. MAIN PIPELINE
# =========================================================

def main():
    cfg = Config()
    model = YOLO(cfg.MODEL_PATH)

    video_source = get_video_source()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {video_source}")

    # --- Pehla frame padho aur usi par LINE + IN/OUT label select karwao ---
    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Could not read the first frame from the video.")

    selector = LineSelector()
    p1, p2, pos_label, neg_label = selector.select(first_frame)

    counter = BoxTouchCounter(p1, p2, cfg.VEHICLE_CLASS_IDS, pos_label, neg_label)
    print(f"[INFO] Direction labels: '{counter.dir_pos}' (arrow side) vs '{counter.dir_neg}' (opposite side)\n")

    # --- Video ko wapas shuru se chalao (pehla frame already consume ho chuka hai) ---
    if isinstance(video_source, str):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    class_ids_list = list(cfg.VEHICLE_CLASS_IDS.keys())

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video stream.")
            break

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

                touch_result = counter.update(int(track_id), int(class_id), x1, y1, x2, y2, cx, cy)
                if touch_result:
                    c_name, direction = touch_result
                    print(f"[COUNTED] ID {track_id} -> {c_name.upper()} box touched line "
                          f"| Direction: {direction} "
                          f"| Class totals: {counter.class_counts} "
                          f"| Direction totals: {counter.direction_counts} "
                          f"| Total: {counter.total()}")

                already_counted = int(track_id) in counter.counted_ids
                frame = draw_detection(frame, x1, y1, x2, y2, track_id, class_name, conf, already_counted)

        frame = draw_counts_panel(frame, counter)

        cv2.imshow("Smart Traffic Management (YOLO11) - Manual Line + IN/OUT + Box-Touch Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n===== FINAL COUNTS =====")
    for name, count in counter.class_counts.items():
        print(f"{name.capitalize()}: {count}")
    print(f"IN : {counter.direction_counts.get('IN', 0)}")
    print(f"OUT: {counter.direction_counts.get('OUT', 0)}")
    print(f"Total Vehicles: {counter.total()}")
    print("=========================")


if __name__ == "__main__":
    main()