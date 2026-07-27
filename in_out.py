"""
Smart Traffic Management System (YOLO11 version) - LINE CROSSING Edition
---------------------------------------------------------------------------
Kya naya hai (updates over ROI-polygon version):
  1. Ab boundary ek POLYGON nahi, ek SINGLE LINE hai (2 points se banti hai)
  2. User pehle frame par mouse se sirf 2 CLICK karega -> line ban jayegi
  3. YOLO11 tracking (ByteTrack) se har vehicle ko unique ID milta hai
  4. Har vehicle ke CENTER OF MASS (centroid) ka current frame vs previous frame
     ka "side" (line ke left/right ya upar/neeche) compare hota hai:
        - agar side change hua -> vehicle ne line CROSS kiya
        - cross ki DIRECTION se pata chalta hai vehicle "IN" (aa raha hai)
          ya "OUT" (ja raha hai) - jaisa bhi define karo
  5. Terminal mein REAL-TIME print hota hai -> kis vehicle ne kis direction
     mein line cross ki, aur ab tak IN / OUT / per-class running totals

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
    VIDEO_PATH = "traffic_video_1.mp4"

    # --- YOLO11 inference params ---
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    INFER_SIZE = 640

    # --- COCO class ids jo humein chahiye (id -> readable name) ---
    VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    # --- Tracker config jo ultralytics ke saath already aata hai ---
    TRACKER = "bytetrack.yaml"

    # --- Line ke aas-paas kitne pixels tak crossing ko valid maana jaaye ---
    # (line segment ke bounding box ko itne pixels se extend karta hai, taaki
    #  bahut door ka "infinite line" crossing galti se count na ho)
    LINE_MARGIN = 60


# =========================================================
# 2. LINE SELECTOR (mouse click based - sirf 2 points)
# =========================================================

class LineSelector:
    """
    Pehle frame par user ko mouse se ek LINE (2 points) define karne deta hai.
    Left click -> point add (max 2 points)
    'r'        -> reset
    ENTER      -> confirm (exactly 2 points chahiye)
    'q'        -> cancel
    """

    def __init__(self):
        self.window_name = "Select Boundary LINE: Left-Click 2 points | R=Reset | ENTER=Confirm | Q=Quit"
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

        print("\n[LINE SELECTION] Frame par LEFT CLICK karke EXACTLY 2 points banao (line ke 2 ends).")
        print("[LINE SELECTION] Confirm karne ke liye ENTER dabao.")
        print("[LINE SELECTION] Galti ho jaaye to 'r' dabao reset karne ke liye. Cancel ke liye 'q'.\n")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                if len(self.points) == 2:
                    break
                else:
                    print("Exactly 2 points select karo (line ke dono ends)!")
            elif key == ord('r'):
                self.points = []
                self._redraw()
            elif key == ord('q'):
                cv2.destroyWindow(self.window_name)
                raise SystemExit("Line selection cancelled by user.")

        cv2.destroyWindow(self.window_name)
        line_start, line_end = self.points[0], self.points[1]
        print(f"[LINE SELECTION] Line confirm ho gayi: {line_start} -> {line_end}\n")
        return line_start, line_end


# =========================================================
# 3. LINE-CROSSING VEHICLE COUNTER (direction aware)
# =========================================================

class LineCrossCounter:
    """
    Har track_id ke liye pichle frame ka "side" (line ke ek taraf ya doosri
    taraf) yaad rakhta hai. Jab side change hoti hai -> matlab vehicle ne
    line cross ki.

    side == +1 -> line ke ek taraf
    side == -1 -> line ke doosri taraf

    +1 -> -1 crossing ko "IN"  maana hai
    -1 -> +1 crossing ko "OUT" maana hai
    (agar tumhari video mein ulta lage to bas IN/OUT labels neeche swap kar dena)
    """

    def __init__(self, line_start, line_end, vehicle_class_ids: dict, margin: int = 60):
        self.line_start = line_start
        self.line_end = line_end
        self.vehicle_class_ids = vehicle_class_ids
        self.margin = margin

        self.track_last_side = {}   # track_id -> last known side (+1/-1)
        self.counted_crossings = set()  # (track_id, frame_side_transition) - avoid duplicate same-frame counts

        self.counts_in = {name: 0 for name in vehicle_class_ids.values()}
        self.counts_out = {name: 0 for name in vehicle_class_ids.values()}

    def _side_of_line(self, px, py):
        ax, ay = self.line_start
        bx, by = self.line_end
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross > 0:
            return 1
        elif cross < 0:
            return -1
        return 0

    def _near_line(self, px, py):
        ax, ay = self.line_start
        bx, by = self.line_end
        min_x, max_x = min(ax, bx) - self.margin, max(ax, bx) + self.margin
        min_y, max_y = min(ay, by) - self.margin, max(ay, by) + self.margin
        return min_x <= px <= max_x and min_y <= py <= max_y

    def update(self, track_id, class_id, cx, cy):
        """
        Returns (direction, class_name) if a crossing was just detected this
        frame, else None. direction is "IN" or "OUT".
        """
        if class_id not in self.vehicle_class_ids:
            return None

        current_side = self._side_of_line(cx, cy)
        prev_side = self.track_last_side.get(track_id)

        direction = None
        if (prev_side is not None and current_side != 0 and prev_side != 0
                and current_side != prev_side and self._near_line(cx, cy)):

            class_name = self.vehicle_class_ids[class_id]
            if prev_side == 1 and current_side == -1:
                direction = "IN"
                self.counts_in[class_name] += 1
            elif prev_side == -1 and current_side == 1:
                direction = "OUT"
                self.counts_out[class_name] += 1

        if current_side != 0:
            self.track_last_side[track_id] = current_side

        if direction:
            return direction, self.vehicle_class_ids[class_id]
        return None

    def total_in(self):
        return sum(self.counts_in.values())

    def total_out(self):
        return sum(self.counts_out.values())


# =========================================================
# 4. DRAWING HELPERS
# =========================================================

CLASS_COLORS = {
    "car": (0, 255, 0),
    "bus": (255, 200, 0),
    "truck": (0, 165, 255),
    "motorcycle": (255, 0, 255),
}


def draw_line(frame, line_start, line_end):
    cv2.line(frame, line_start, line_end, (0, 255, 255), 3)
    cv2.circle(frame, line_start, 6, (0, 0, 255), -1)
    cv2.circle(frame, line_end, 6, (0, 0, 255), -1)
    return frame


def draw_detection(frame, x1, y1, x2, y2, track_id, class_name, conf):
    color = CLASS_COLORS.get(class_name, (200, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id} {class_name.upper()} {conf:.2f}"
    cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, color, -1)
    return frame


def draw_counts_panel(frame, counter: LineCrossCounter):
    names = list(counter.vehicle_class_ids.values())
    h = 70 + 25 * len(names)
    cv2.rectangle(frame, (10, 10), (360, 10 + h), (0, 0, 0), -1)

    cv2.putText(frame, f"IN: {counter.total_in()}   OUT: {counter.total_out()}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    y = 65
    for name in names:
        cv2.putText(frame,
                    f"{name.capitalize()} -> IN:{counter.counts_in[name]}  OUT:{counter.counts_out[name]}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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
    line_start, line_end = selector.select(first_frame)

    counter = LineCrossCounter(line_start, line_end, cfg.VEHICLE_CLASS_IDS, margin=cfg.LINE_MARGIN)

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
        frame = draw_line(frame, line_start, line_end)

        if result.boxes is not None and result.boxes.id is not None:  # type: ignore
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)       # type: ignore
            track_ids = result.boxes.id.cpu().numpy().astype(int)      # type: ignore
            class_ids = result.boxes.cls.cpu().numpy().astype(int)     # type: ignore
            confs = result.boxes.conf.cpu().numpy()                    # type: ignore

            for (x1, y1, x2, y2), track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = cfg.VEHICLE_CLASS_IDS.get(int(class_id), "unknown")

                # --- Line crossing check (direction aware, unique per crossing) ---
                crossing = counter.update(int(track_id), int(class_id), cx, cy)
                if crossing:
                    direction, c_name = crossing
                    print(f"[CROSS] Vehicle ID {track_id} -> {c_name.upper()} went {direction} | "
                          f"IN totals: {counter.counts_in} | OUT totals: {counter.counts_out} | "
                          f"Total IN:{counter.total_in()} OUT:{counter.total_out()}")

                frame = draw_detection(frame, x1, y1, x2, y2, track_id, class_name, conf)

        frame = draw_counts_panel(frame, counter)

        cv2.imshow("Smart Traffic Management (YOLO11) - Line Crossing Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n===== FINAL COUNTS =====")
    for name in cfg.VEHICLE_CLASS_IDS.values():
        print(f"{name.capitalize()} -> IN: {counter.counts_in[name]}   OUT: {counter.counts_out[name]}")
    print(f"TOTAL IN: {counter.total_in()}   TOTAL OUT: {counter.total_out()}")
    print("=========================")


if __name__ == "__main__":
    main()