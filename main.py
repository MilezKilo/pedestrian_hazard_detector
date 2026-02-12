# ----------------------- ИМПОРТ БИБЛИОТЕК -----------------------
import os

os.environ["TORCH_HOME"] = r"D:\Institute\Diploma_v3\models"

import cv2
import time
import threading

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

# ----------------------- КОНСТАНТЫ -----------------------
VIDEO_DIR = "data/video_clips/"

COCO = {
    1: "person",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

SCORE_THRESH = 0.8
EVERY = 4
MAX_STALE_SEC = 1.0 # Сколько секунд можно рисовать "старые" детекции, если инференс не успевает

# ----------------------- МЕТОДЫ -----------------------
def predict(model, frame_bgr, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = to_tensor(frame_rgb).to(device)

    with torch.inference_mode():
        out = model([x])[0]  # Словарь: boxes, labels, scores

    detections = []
    boxes = out["boxes"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()

    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        if sc < SCORE_THRESH:
            continue
        if int(lab) not in COCO:
            continue
        detections.append((COCO[int(lab)], float(sc), (int(x1), int(y1), int(x2), int(y2))))
    return detections


def draw(frame, det):
    label, score, (x1, y1, x2, y2) = det
    color = (0, 255, 0) if label == "person" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{label}:{score:.2f}", (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _infer_loop(model, device, shared, lock, stop_event):
    """Поток, который делает инференс, когда главный поток положил новый кадр."""
    last_seen_req_id = -1

    while not stop_event.is_set():
        time.sleep(0.001)

        with lock:
            req_id = shared.get("req_id", -1)
            frame = shared.get("req_frame", None)

        # нет нового запроса
        if req_id == last_seen_req_id or frame is None:
            continue

        # делаем инференс
        dets = predict(model, frame, device)
        ts = time.monotonic()

        with lock:
            # если dets пустой, всё равно обновляем (так честнее)
            shared["detections"] = dets
            shared["detections_ts"] = ts
            last_seen_req_id = req_id



def main(path):
    # Проверка на доступность GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ускоряет torch/свертки при фиксированных размерах
    torch.backends.cudnn.benchmark = True

    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    model.to(device).eval()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Не могу открыть видео")

    # -------- синхронизация показа по FPS, чтобы видео не "ускорялось" --------
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = int(1000 / fps) if fps and fps > 1 else 16
    print("fps:", fps, "delay_ms:", delay_ms)

    # -------- общие данные между потоками --------
    shared = {
        "req_id": -1,
        "req_frame": None,
        "detections": [],
        "detections_ts": 0.0,
    }
    lock = threading.Lock()
    stop_event = threading.Event()

    t = threading.Thread(target=_infer_loop, args=(model, device, shared, lock, stop_event), daemon=True)
    t.start()

    frame_idx = 0

    # FPS отображения (чтобы видеть, что всё плавно)
    show_t0 = time.monotonic()
    show_cnt = 0
    show_fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        # каждые EVERY кадров отправляем запрос на инференс в поток
        if frame_idx % EVERY == 0:
            with lock:
                shared["req_id"] = frame_idx
                shared["req_frame"] = frame.copy()  # важно: копия, иначе кадр поменяется

        # берём последние детекции (не блокируя инференс)
        now = time.monotonic()
        with lock:
            detections = list(shared.get("detections", []))
            det_ts = float(shared.get("detections_ts", 0.0))

        # рисуем только если детекции не слишком старые (иначе будет "залипание")
        if detections and (now - det_ts) <= MAX_STALE_SEC:
            for d in detections:
                draw(frame, d)

        # считаем fps отображения
        show_cnt += 1
        if now - show_t0 >= 1.0:
            show_fps = show_cnt / (now - show_t0)
            show_cnt = 0
            show_t0 = now

        cv2.putText(frame, f"show_fps: {show_fps:.1f} | EVERY: {EVERY}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Detections", frame)

        # важное: delay_ms держит видео в реальном темпе (не ускоряется)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == 27:
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


# ----------------------- ЗАПУСК -----------------------
if __name__ == "__main__":
    vid_name = input('Введите название клипа (пример: vid_001_0000 / vid_002_0002): ')
    clip_path = VIDEO_DIR + f'{vid_name}.mp4'
    main(path=clip_path)



# # ----------------------- ИМПОРТ БИБЛИОТЕК -----------------------
# import os
#
# os.environ["TORCH_HOME"] = r"D:\Institute\Diploma_v3\models"
#
# import cv2
# import torch
# from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
# from torchvision.transforms.functional import to_tensor
#
# # ----------------------- КОНСТАНТЫ -----------------------
# VIDEO_DIR = "data/video_clips/"
#
# COCO = {
#     1: "person",
#     3: "car",
#     4: "motorcycle",
#     6: "bus",
#     8: "truck",
# }
#
# SCORE_THRESH = 0.5
# EVERY = 4

# ----------------------- МЕТОДЫ -----------------------
# def predict(model, frame_bgr, device):
#     frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#
#     x = to_tensor(frame_rgb).to(device)
#
#     with torch.no_grad():
#         out = model([x])[0]  # Словарь: boxes, labels, scores
#
#     detections = []
#     boxes = out["boxes"].detach().cpu().numpy()
#     labels = out["labels"].detach().cpu().numpy()
#     scores = out["scores"].detach().cpu().numpy()
#
#     for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
#         if sc < SCORE_THRESH:
#             continue
#         if int(lab) not in COCO:
#             continue
#         detections.append((COCO[int(lab)], float(sc), (int(x1), int(y1), int(x2), int(y2))))
#     return detections
#
#
# def draw(frame, det):
#     label, score, (x1, y1, x2, y2) = det
#     color = (0, 255, 0) if label == "person" else (0, 0, 255)
#     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#     cv2.putText(frame, f"{label}:{score:.2f}", (x1, max(20, y1 - 7)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#
# def main(path):
#     # Проверка на доступность GPU
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("device:", device)
#
#     model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
#     model.to(device).eval()
#
#     cap = cv2.VideoCapture(path)
#
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     delay_ms = int(1000 / fps) if fps and fps > 1 else 16
#     print("fps:", fps, "delay_ms:", delay_ms)
#
#     if not cap.isOpened():
#         raise RuntimeError("Не могу открыть видео")
#
#     detections = []
#     frame_idx = 0
#
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#
#         frame_idx += 1
#         if frame_idx % EVERY == 0:
#             detections = predict(model, frame, device)
#
#         for d in detections:
#             draw(frame, d)
#
#         cv2.imshow("Detections", frame)
#         if (cv2.waitKey(delay_ms) & 0xFF) == 27:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# # ----------------------- ЗАПУСК -----------------------
# if __name__ == "__main__":
#     vid_name = input('Введите название клипа (пример: vid_001_0000 / vid_002_0002): ')
#     clip_path = VIDEO_DIR + f'{vid_name}.mp4'
#     main(path=clip_path)