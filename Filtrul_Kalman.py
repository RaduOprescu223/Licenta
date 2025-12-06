import numpy as np
import cv2

class KalmanFilterTracker:
    """
    Gestionează un singur filtru Kalman pentru urmărirea unui obiect 2D (poziție și viteză).
    """
    def __init__(self, initial_bbox_center):
        # 1. Definiția Stării (State Vector): [px, py, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)

        # 2. Matricea de Tranziție (A): Model de Viteză Constantă
        # dt = 1 (interval de timp între cadre)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)

        # 3. Matricea de Măsurare (H): Măsurăm doar poziția [px, py]
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)

        # 4. Covarianța Zgomotului Procesului (Q) - Ajustează încrederea în mișcare
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32) * 0.1

        # 5. Covarianța Zgomotului de Măsurare (R) - Ajustează încrederea în YOLO
        self.kf.measurementNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 5

        # 6. Inițializarea Stării (folosind poziția detectată de YOLO)
        self.kf.statePost = np.array([[initial_bbox_center[0]],
                                      [initial_bbox_center[1]],
                                      [0],
                                      [0]], np.float32)


    def predict(self):
        """ Calculează următoarea stare prezisă. """
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])

    def correct(self, measurement):
        """ Actualizează starea filtrului cu o nouă măsurătoare de la YOLO. """
        # measurement: tuple/list (px_masurat, py_masurat)
        measurement_vector = np.array([[np.float32(measurement[0])],
                                       [np.float32(measurement[1])]])

        # Aplicarea corecției (Update)
        corrected = self.kf.correct(measurement_vector)
        return int(corrected[0]), int(corrected[1])