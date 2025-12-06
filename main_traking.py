import cv2
from ultralytics import YOLO


from object_traker import ObjectTracker

def run_real_time_tracking():
  
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Eroare: Nu s-a putut deschide camera!")
        return

    # 2. Inițializare Tracker-ul Multi-Obiect
    tracker = ObjectTracker(iou_threshold=0.5, max_age=5)

    print("Sistem de Tracking Incarcat. Apasa 'q' pentru a iesi.")

    while True:
        # Citeste un nou cadru
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Detecția YOLO
        results = model(frame, verbose=False) 

        yolo_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extrage BBox-ul (colțuri)
                x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]

               
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width // 2
                center_y = y1 + height // 2

                # Formatul necesar: (x_center, y_center, width, height)
                yolo_detections.append((center_x, center_y, width, height))

        # 4. Actualizarea Tracker-ului cu Filtre Kalman
        tracked_objects_data = tracker.update_tracks(yolo_detections)

       
        for obj_id, x1, y1, x2, y2 in tracked_objects_data:
            # Desenează Bounding Box-ul CU COORDONATELE CORECTATE DE KALMAN
            color = (255, 0, 0) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("YOLOv8 + Kalman Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_real_time_tracking()
