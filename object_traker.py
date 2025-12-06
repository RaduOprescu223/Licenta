from collections import OrderedDict
import numpy as np
import cv2
from Filtrul_Kalman import KalmanFilterTracker 
class ObjectTracker:
    """
    Gestionează mai multe filtre Kalman, atribuie ID-uri și asociază detecțiile YOLO
    cu obiectele urmărite folosind metrica IOU.
    """

    def __init__(self, iou_threshold=0.5, max_age=5):
        
        self.tracked_objects = OrderedDict()

        self.next_object_id = 0

        self.iou_threshold = iou_threshold

        self.max_age = max_age

        # Contorul pentru cât timp a lipsit fiecare obiect
        self.missing_counts = OrderedDict()

        # Stocăm ultima casetă de delimitare (BBox) pentru fiecare obiect
        # (necesar pentru a calcula dimensiunea BBox-ului prezis)
        self._last_bboxes = {}

    def update_tracks(self, yolo_detections):
        """
      
        """

        # 1. PREDICTEAZĂ pozițiile noi pentru obiectele urmărite
        predicted_bboxes = []  # Va stoca: (obj_id, [x1, y1, x2, y2] prezis)
        for obj_id, tracker in self.tracked_objects.items():
            # Predicția se face pe punctul central (px, py)
            pred_center_x, pred_center_y = tracker.predict()

            # Folosim ultima dimensiune cunoscută a BBox-ului
            last_bbox_center_x, last_bbox_center_y, pred_width, pred_height = self._get_last_bbox(obj_id)

            # Format: [x1, y1, x2, y2]
            pred_bbox = self._center_to_corners(pred_center_x, pred_center_y, pred_width, pred_height)
            predicted_bboxes.append((obj_id, pred_bbox))

        # Formatul detecțiilor YOLO pentru IOU: [x1, y1, x2, y2]
        current_detections_corners = [self._center_to_corners(d[0], d[1], d[2], d[3]) for d in yolo_detections]

        # 2. ASOCIEREA DATELOR
        # Găsim cea mai bună potrivire între predicții (Kalman) și măsurători (YOLO)
        matches, unmatched_yolo_indices = self._associate_detections(predicted_bboxes, current_detections_corners)

        # 3. ACTUALIZAREA TRACKERELOR EXISTENTE
        used_yolo_indices = set()

        # matches: Listă de (obj_id, yolo_idx) pentru potrivirile găsite
        for obj_id, yolo_idx in matches:
            if yolo_idx >= 0:
                tracker = self.tracked_objects[obj_id]
                detection = yolo_detections[yolo_idx]

                # Coordonatele centrului YOLO sunt măsurătoarea (measurement)
                yolo_center = (detection[0], detection[1])

                # Aplicarea CORRECȚIEI (Update) a filtrului Kalman
                tracker.correct(yolo_center)

                # Actualizează dimensiunea BBox-ului și contorul de lipsă
                self._update_last_bbox(obj_id, detection)
                self.missing_counts[obj_id] = 0
                used_yolo_indices.add(yolo_idx)

        # 4. GESTIONAREA OBIECTELOR NOI
        for i, detection in enumerate(yolo_detections):
            if i in unmatched_yolo_indices:
                # O nouă detecție YOLO care nu a fost asociată
                self._register_new_object(detection)

        # 5. GESTIONAREA OBIECTELOR PIERDUTE (Lost/Aged)
        objects_to_remove = []
        for obj_id in list(self.tracked_objects.keys()):
            # Dacă obiectul nu a fost actualizat (nu a fost în lista de matches)
            if obj_id not in [m[0] for m in matches if m[1] >= 0]:
                self.missing_counts[obj_id] += 1

                # Dacă obiectul a lipsit prea mult, îl ștergem
                if self.missing_counts[obj_id] > self.max_age:
                    objects_to_remove.append(obj_id)

        # Curățarea trackere-lor șterse
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
            del self.missing_counts[obj_id]
            if obj_id in self._last_bboxes:
                del self._last_bboxes[obj_id]

        # Returnează lista finală de rezultate urmărite
        return self._get_tracked_results()

    # --- Metode Ajutătoare pentru Asocierea Datelor ---

    def _associate_detections(self, predicted_bboxes, current_detections_corners):
        """
        Calculează matricea IOU și găsește cele mai bune potriviri (Potrivire Greedy).
        """
        num_preds = len(predicted_bboxes)
        num_dets = len(current_detections_corners)

        if num_preds == 0 or num_dets == 0:
            return [], list(range(num_dets))

        # Matricea IOU (rânduri = predicții, coloane = detecții)
        iou_matrix = np.zeros((num_preds, num_dets))
        for i, (_, pred_bbox) in enumerate(predicted_bboxes):
            for j, det_bbox in enumerate(current_detections_corners):
                iou_matrix[i, j] = self._calculate_iou(pred_bbox, det_bbox)

        # Potrivire Greedy
        matches = []
        used_preds = set()
        used_dets = set()

        # Iterăm pe detecțiile cu cel mai mare IOU
        while np.max(iou_matrix) >= self.iou_threshold:
            # Găsim indexul celui mai mare IOU rămas
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            pred_idx, det_idx = max_iou_idx

            # Verificăm dacă sunt deja folosite
            if pred_idx not in used_preds and det_idx not in used_dets:
                obj_id, _ = predicted_bboxes[pred_idx]
                matches.append((obj_id, det_idx))
                used_preds.add(pred_idx)
                used_dets.add(det_idx)

            # Marcăm elementul ca deja procesat (cu o valoare sub prag)
            iou_matrix[pred_idx, det_idx] = -1

            # Identificăm predicțiile neasociate
        unmatched_preds = [i for i in range(num_preds) if i not in used_preds]
        for pred_idx in unmatched_preds:
            obj_id, _ = predicted_bboxes[pred_idx]
            matches.append((obj_id, -1))  # -1 indică că nu s-a găsit o detecție pentru acest obiect

        # Identificăm detecțiile neasociate (obiecte noi)
        unmatched_yolo_indices = [i for i in range(num_dets) if i not in used_dets]

        return matches, unmatched_yolo_indices

    def _calculate_iou(self, boxA, boxB):
        """ Calculează Intersection Over Union (IOU) între două BBox-uri [x1, y1, x2, y2]. """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Aria intersecției
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Ariile celor două BBox-uri
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # IOU = Intersecție / (AriaA + AriaB - Intersecție)
        if (boxAArea + boxBArea - interArea) == 0:
            return 0

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _register_new_object(self, detection):
        """ Creează un nou Kalman Filter Tracker pentru o detecție nouă. """
        center_x, center_y = detection[0], detection[1]

        new_tracker = KalmanFilterTracker((center_x, center_y))

        new_id = self.next_object_id
        self.tracked_objects[new_id] = new_tracker
        self.missing_counts[new_id] = 0
        self.next_object_id += 1

        self._update_last_bbox(new_id, detection)

        # --- Utilitare pentru formatul BBox ---

    def _center_to_corners(self, center_x, center_y, width, height):
        """ Convertește (centru_x, centru_y, lățime, înălțime) la (x1, y1, x2, y2). """
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        return [x1, y1, x2, y2]

    def _update_last_bbox(self, obj_id, detection):
        """ Stochează ultima detecție cunoscută a BBox-ului (centru, lățime, înălțime). """
        # detection: (x_center, y_center, width, height)
        self._last_bboxes[obj_id] = detection

    def _get_last_bbox(self, obj_id):
        """ Returnează ultima BBox-ul stocată sau o valoare implicită. """
        # Format returnat: (x_center, y_center, width, height)
        return self._last_bboxes.get(obj_id, (0, 0, 50, 50))

    def _get_tracked_results(self):
        """
        Returnează rezultatele finale, folosind poziția corectată de Kalman.
        Format: [(ID, x1, y1, x2, y2), ...]
        """
        results = []
        for obj_id, tracker in self.tracked_objects.items():
            # Poziția corectată/actualizată (cea mai bună estimare Kalman)
            corrected_center_x = tracker.kf.statePost[0, 0]
            corrected_center_y = tracker.kf.statePost[1, 0]

            # Folosim dimensiunea BBox-ului cunoscută
            _, _, width, height = self._get_last_bbox(obj_id)

            # Convertim înapoi la formatul colțuri [x1, y1, x2, y2] pentru afișare
            bbox_corners = self._center_to_corners(corrected_center_x, corrected_center_y, width, height)

            results.append((obj_id, bbox_corners[0], bbox_corners[1], bbox_corners[2], bbox_corners[3]))

        return results
