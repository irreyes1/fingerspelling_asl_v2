import cv2
import numpy as np
import mediapipe as mp
import time
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MP_HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

def landmarks_to_vec63(hand_landmarks):
    vec = []
    for lm in hand_landmarks:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec, dtype=np.float32)

def main():
    # RUTA ROBUSTA (independiente de dónde ejecutes python)
    here = Path(__file__).resolve().parent          # .../src
    root = here.parent                               # .../fingerspelling_asl
    model_path = root / "artifacts" / "models" / "hand_landmarker.task"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No encuentro el modelo:\n{model_path}\n\n"
            "Descárgalo como 'hand_landmarker.task' y ponlo en:\n"
            f"{root / 'artifacts' / 'models'}"
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam (prueba índice 1 o permisos)")

    print("Webcam OK. Pulsa ESC para salir.")

    t0 = time.perf_counter()
    last = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.perf_counter()
        dt = now - last
        last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)  # suavizado

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int((time.perf_counter() - t0) * 1000)  # timestamp monótono
        result = detector.detect_for_video(mp_image, ts_ms)

        h, w = frame.shape[:2]

        if result.hand_landmarks:
            # Si quieres vec63 de la primera mano:
            vec63 = landmarks_to_vec63(result.hand_landmarks[0])

            # Dibuja todas las manos
            for i, hand in enumerate(result.hand_landmarks):
                # Puntos
                pts = []
                for lm in hand:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    pts.append((x, y))
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # Conexiones
                for a, b in MP_HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

                # Handedness (si está disponible)
                label = None
                if result.handedness and len(result.handedness) > i and len(result.handedness[i]) > 0:
                    label = result.handedness[i][0].category_name  # "Left"/"Right"
                if label:
                    cv2.putText(frame, f"{label}", (pts[0][0], pts[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"vec63: {vec63.shape} (min={vec63.min():.2f} max={vec63.max():.2f})", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Fingerspelling Webcam (MediaPipe Tasks)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
