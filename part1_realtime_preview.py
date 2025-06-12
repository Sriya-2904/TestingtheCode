import cv2
import os
import yolov5
from ultralytics import YOLO
import easyocr

# ----------- Setup Models -------------
device = 'cpu'
vehicle_model = YOLO("yolov5n.pt").to(device)
plate_model = yolov5.load('weights/best.pt', device=device)
plate_model.conf = 0.25
plate_model.iou = 0.45
plate_model.agnostic = False
plate_model.multi_label = False
plate_model.max_det = 10

reader = easyocr.Reader(['en'], gpu=False)
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorbike, bus, truck

# ----------- Setup Input ----------
cap = cv2.VideoCapture("video/input_video.mp4")
frame_id = 0
cv2.namedWindow("Vehicle and Number Plate Detection", cv2.WINDOW_NORMAL)

# ----------- Start Processing -----------
latest_detected_texts = []

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ End of video or unreadable frame.")
        break

    frame_id += 1
    if frame_id % 5 != 0:
        continue

    output_frame = frame.copy()
    vehicle_results = vehicle_model(frame)[0]

    for box in vehicle_results.boxes:
        cls = int(box.cls[0].item())
        if cls in VEHICLE_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            pad = 15
            crop_y1 = max(0, y1 - pad)
            crop_y2 = min(frame.shape[0], y2 + pad)
            crop_x1 = max(0, x1 - pad)
            crop_x2 = min(frame.shape[1], x2 + pad)
            vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            if vehicle_crop.size == 0:
                continue

            # Plate detection
            plate_results = plate_model(vehicle_crop, size=640)
            predictions = plate_results.pred[0]
            vehicle_area = vehicle_crop.shape[0] * vehicle_crop.shape[1]
            vehicle_width = vehicle_crop.shape[1]

            for plate_idx, pred in enumerate(predictions):
                px1, py1, px2, py2, conf, plate_cls = pred.tolist()
                px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
                plate_w = px2 - px1
                plate_h = py2 - py1
                plate_area = plate_w * plate_h
                aspect_ratio = plate_w / plate_h if plate_h != 0 else 0

                if plate_area / vehicle_area > 0.05:
                    continue
                if aspect_ratio < 1.2 or aspect_ratio > 7.0:
                    continue

                plate_x_center = (px1 + px2) / 2
                if not (0.2 * vehicle_width <= plate_x_center <= 0.8 * vehicle_width):
                    continue

                gx1 = crop_x1 + px1
                gy1 = crop_y1 + py1
                gx2 = crop_x1 + px2
                gy2 = crop_y1 + py2

                # Draw plate box
                cv2.rectangle(output_frame, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
                cv2.putText(output_frame, "Plate", (gx1, gy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Draw vehicle box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = vehicle_model.names.get(cls, f"Class {cls}")
            cv2.putText(output_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    display_frame = cv2.resize(output_frame, (960, 540))
    window_title = " | ".join(latest_detected_texts)
    if window_title.strip():
        cv2.setWindowTitle("Vehicle and Number Plate Detection", f"Detected Plates: {window_title}")

    cv2.imshow("Vehicle and Number Plate Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        print("🛑 ESC pressed. Exiting.")
        break

cap.release()
cv2.destroyAllWindows()
print("🎬 Video processing complete.")
