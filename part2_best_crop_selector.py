import cv2
import easyocr
import os
import re
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from yolov5 import load
from my_utils.plate_enhancer import enhance_with_realesrgan

# -----------------------------
# CONFIGURATION
# -----------------------------
VIDEO_PATH = 'video/input_video.mp4'
VEHICLE_CLASSES = [2, 3, 5, 7]
MIN_FRAMES_PER_TRACK = 3

SAVE_DIR = 'outputs'
FRAME_DIR = os.path.join(SAVE_DIR, 'best_frames')
VEHICLE_DIR = os.path.join(SAVE_DIR, 'vehicle_crops')
PLATE_DIR = os.path.join(SAVE_DIR, 'plate_crops')
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(VEHICLE_DIR, exist_ok=True)
os.makedirs(PLATE_DIR, exist_ok=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
vehicle_model = YOLO('yolov5n.pt')
plate_model = load('weights/best.pt', device='cpu')
plate_model.conf = 0.25
plate_model.iou = 0.45

ocr_reader = easyocr.Reader(['en'], gpu=False)
tracker = DeepSort(max_age=15)

# -----------------------------
# HELPERS
# -----------------------------
def get_sharpness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def crop_with_padding(img, x1, y1, x2, y2, pad=30):
    h, w = img.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img[y1:y2, x1:x2]

def extract_text_from_plate(plate_crop):
    try:
        results = ocr_reader.readtext(plate_crop)
        for (_, text, conf) in results:
            digits = re.findall(r'\d', text)
            if len(digits) >= 4 and conf >= 0.3:
                return ''.join(digits), conf
    except Exception as e:
        print(f"[OCR ERROR]: {e}")
    return 'unreadable', 0.0

# -----------------------------
# TRACK + BUFFER VEHICLES
# -----------------------------
track_buffers = defaultdict(list)
cap = cv2.VideoCapture(VIDEO_PATH)
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = vehicle_model(frame)[0]
    vehicle_dets = []
    for box in detections.boxes:
        cls = int(box.cls[0].item())
        if cls in VEHICLE_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            vehicle_dets.append(([x1, y1, x2 - x1, y2 - y1], 0.99, 'vehicle'))

    tracks = tracker.update_tracks(vehicle_dets, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        box = track.to_ltrb()
        x1, y1, x2, y2 = map(int, box)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size > 0:
            track_buffers[track_id].append((frame_index, frame.copy(), crop))

    frame_index += 1

cap.release()

# -----------------------------
# FINAL SUMMARY REPORT
# -----------------------------
final_results = []

for track_id, frames in track_buffers.items():
    if len(frames) < MIN_FRAMES_PER_TRACK:
        continue

    best_frame = None
    best_score = -1
    best_index = -1

    for idx, (frame_num, full_frame, crop) in enumerate(frames):
        score = get_sharpness_score(crop)
        if score > best_score:
            best_score = score
            best_frame = full_frame
            best_index = frame_num

    # Save best frame
    best_frame_path = f"{FRAME_DIR}/vehicle_{int(track_id):03d}_frame_{int(best_index):05d}.jpg"
    try:
        cv2.imwrite(best_frame_path, best_frame)
    except Exception as e:
        print(f"[Save Error] Best frame for ID {track_id}: {e}")

    plate_results = plate_model(best_frame, size=640)
    plate_preds = plate_results.pred[0]
    vehicle_area = best_frame.shape[0] * best_frame.shape[1]
    vehicle_width = best_frame.shape[1]

    plate_saved = False
    for pred in plate_preds:
        try:
            px1, py1, px2, py2, conf, _ = pred.tolist()
            px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
            plate_w, plate_h = px2 - px1, py2 - py1
            plate_area = plate_w * plate_h
            aspect_ratio = plate_w / plate_h if plate_h != 0 else 0

            if plate_area / vehicle_area > 0.05 or not (1.2 < aspect_ratio < 7.0):
                continue
            if not (0.2 * vehicle_width <= (px1 + px2) / 2 <= 0.8 * vehicle_width):
                continue

            plate_crop = crop_with_padding(best_frame, px1, py1, px2, py2, pad=30)
            if plate_crop.size > 0:
                plate_path = f"{PLATE_DIR}/plate_{int(track_id):03d}_frame_{int(best_index):05d}.jpg"
                cv2.imwrite(plate_path, plate_crop)
                plate_saved = True
        except Exception as e:
            print(f"[Detection Error] Track {track_id} Plate: {e}")

    # Save vehicle crop
    for (_, _, crop_frame) in frames:
        if crop_frame.size > 0:
            try:
                cv2.imwrite(f"{VEHICLE_DIR}/vehicle_{int(track_id):03d}.jpg", crop_frame)
            except Exception as e:
                print(f"[Save Error] Vehicle crop for ID {track_id}: {e}")
            break

    text, conf = ('unreadable', 0.0)
    if plate_saved:
        try:
            text, conf = extract_text_from_plate(plate_crop)
        except Exception as e:
            print(f"[OCR Error]: {e}")

    try:
        final_results.append((int(track_id), int(best_index), str(text)))
    except Exception as e:
        print(f"[Append Error] Invalid result entry → {e}")

# -----------------------------
# SAFE SUMMARY OUTPUT
# -----------------------------
print("\n✅ Summary:")
for item in final_results:
    try:
        tid, fno, plate = item
        print(f"Vehicle ID: {tid:03d} | Frame: {fno:05d} | Plate Numbers: {plate}")
    except Exception as e:
        print(f"❌ Error formatting result {item}: {e}")
