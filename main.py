# ----------------------- ИМПОРТ БИБЛИОТЕК ПУТЕЙ И ОС -----------------------
import os
from utils.paths import MODELS_DIR, ROIS_DIR, CLIPS_DIR, paths_check

os.environ["TORCH_HOME"] = str(MODELS_DIR) # Дефолтная директория для моделей


# ----------------------- ИМПОРТ ОСТАЛЬНЫХ БИБЛИОТЕК -----------------------
import cv2
import time
import threading

from roi_handler.roi_loader import load_roi
from utils.drawing_handler import draw_polygons

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

# # ----------------------- КОНСТАНТЫ -----------------------
COCO = {
    1: "person",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

SCORE_THRESH = 0.8
EVERY = 4
MAX_STALE_SEC = 1.0


# ----------------------- ФУНКЦИИ -----------------------
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


# Функция проверки путей (Вынес за мейн функцию для расчистки)




def main(roi_dir, clips_dir, clip_id):
    # ---------------------------------- МОДЕЛИ И НАСТРОЙКА ----------------------------------
    # Настройка девайса
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ускоряет torch/свертки при фиксированных размерах
    torch.backends.cudnn.benchmark = True

    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    model.to(device).eval()

    # ---------------------------------- ПРОВЕРКА ПУТЕЙ ----------------------------------
    clip_path, roi_path = paths_check(roi_dir=roi_dir, clips_dir=clips_dir, clip_id=clip_id)

    # ---------------------------------- ЛОКАЛЬНЫЕ ПЕРЕМЕННЫЕ ----------------------------------
    show_roi = False # Флаг показа ROI
    pause = False # Флаг паузы
    frame = None
    frame_idx = 0 # Индекс кадра
    frozen = None


    # Если существует, то создаем переменные с координатами пешеходного перехода (crosswalk) и дороги (risk)
    roi_id = load_roi(roi_json_path=roi_path)
    crosswalk = roi_id["crosswalk"]
    risk = roi_id["risk"]

    # Общие данные между потоками
    shared = {
        "req_id": -1,
        "req_frame": None,
        "detections": [],
        "detections_ts": 0.0,
        "detections_id": -1
    }
    lock = threading.Lock()
    stop_event = threading.Event()

    # ---------------------------------- РАБОТА С ВИДЕО ----------------------------------
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError("Не могу открыть видео")

    # Синхронизация показа по FPS, чтобы видео не "ускорялось"
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = int(1000 / fps) if fps and fps > 1 else 16
    print("fps:", fps, "delay_ms:", delay_ms)

    t = threading.Thread(target=_infer_loop, args=(model, device, shared, lock, stop_event), daemon=True)
    t.start()

    # Цикл в котором происходит показ видео
    while True:
        if not pause:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # каждые EVERY кадров отправляем запрос на инференс в поток
            if frame_idx % EVERY == 0:
                with lock:
                    shared["req_id"] = frame_idx
                    shared["req_frame"] = frame.copy()  # важно: копия, иначе кадр поменяется
            base = frame

        else:
            # На паузе показываем "замороженный" кадр
            if frozen is None:
                frozen = frame.copy()
            base = frozen

        # Копируем исходный кадр для отрисовки
        vis = base.copy()

        # Берём последние детекции (не блокируя инференс)
        now = time.monotonic()
        with lock:
            detections = list(shared.get("detections", []))
            det_ts = float(shared.get("detections_ts", 0.0))

        # рисуем только если детекции не слишком старые (иначе будет "залипание")
        if detections and (pause or (now - det_ts) <= MAX_STALE_SEC):
            for d in detections:
                draw(vis, d)

        # Флаг рисовать ли ROI
        if show_roi:
            draw_polygons(frame=vis, crosswalk_poly=crosswalk, risk_poly=risk)

        cv2.imshow("Detections", vis)

        # Клавиши управления. q - выход, r - отрисовка ROI, space - пауза
        key = cv2.waitKey(30 if pause else delay_ms) & 0xFF
        if key == ord('q'):
            break

        if key == ord('r'):
            show_roi = not show_roi

        if key == ord(' '):
            pause = not pause
            if not pause:
                frozen = None


    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


# ----------------------- ЗАПУСК -----------------------
if __name__ == "__main__":
    val = input('Введите название клипа (пример: vid_001_0000 / vid_002_0002): ')
    main(roi_dir=ROIS_DIR, clips_dir=CLIPS_DIR, clip_id=val)


# DEPRICATED
# считаем fps отображения
# show_cnt += 1
# if now - show_t0 >= 1.0:
#     show_fps = show_cnt / (now - show_t0)
#     show_cnt = 0
#     show_t0 = now
# cv2.putText(vis, f"show_fps: {show_fps:.1f} | EVERY: {EVERY}",(20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
