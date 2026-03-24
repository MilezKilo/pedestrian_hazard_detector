# ---------------------------------- ИМПОРТ БИБЛИОТЕК ПУТЕЙ И ОС ----------------------------------
import os
from utils.paths import (
    MODELS_DIR,
    ROIS_DIR,
    CLIPS_DIR,
    LOGS_DIR,

    roi_check,
    clip_check,
    logger_check)

os.environ["TORCH_HOME"] = str(MODELS_DIR)

# ---------------------------------- ОСТАЛЬНЫЕ БИБЛИОТЕКИ ----------------------------------
import time
import threading

import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from roi_handler.roi_loader import load_roi
from roi_handler.roi_event import (
    SimpleIoUTracker,
    annotate_zones,
    update_track_zone_state,
    compute_frame_events,
)
from logger.objects_logger import DetectionCsvLogger
from utils.drawing_handler import show_roi_polygons, draw_tracked, draw_events
from detector.detection_methods import infer_loop

# ---------------------------------- КОНСТАНТЫ ----------------------------------
COCO = {
    1: "person",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

SCORE_THRESH = 0.6
EVERY = 4 # Инференс каждые N кадров
MAX_STALE_SEC = 1.0 # Максимальное время жизни старых детекций


# ---------------------------------- MAIN ----------------------------------
def main(roi_dir, clips_dir, clip_name,
         logging_enable: bool= True, show_vid: bool = False, model=None, device=None):

    # ---------------------------------- МОДЕЛЬ И НАСТРОЙКА ----------------------------------
    # Если хотим пройтись только по 1 видео, загружаем модель в мейн функцию
    # Если хотим пройтись по всем видео, то передаем через run_all
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.benchmark = True

        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        model.to(device).eval()


    # ---------------------------------- ПРОВЕРКА ПУТЕЙ ----------------------------------
    # КЛИПЫ
    clip_path = clip_check(clips_dir=clips_dir, clip_name=clip_name)

    # ROI
    roi_path = roi_check(roi_dir=roi_dir, return_roi_path=True, clip_name=clip_name)
    roi_data = load_roi(roi_json_path=roi_path)

    # ЛОГИ
    if logging_enable:
        objects_logs_path = logger_check(logger_dir=LOGS_DIR, logger_type='object')
        events_logs_path = logger_check(logger_dir=LOGS_DIR, logger_type='event')

    # РАЗДЕЛЕНИЕ ROI НА ЗЕБРУ И ДОРОГУ
    crosswalks = roi_data["crosswalks"]
    risks = roi_data["risks"]

    # ---------------------------------- ОБЩЕЕ СОСТОЯНИЕ ПОТОКОВ ----------------------------------
    shared = {
        "req_id": -1,
        "req_frame": None,
        "detections": [],
        "detections_ts": 0.0,
        "detections_id": -1,
    }
    lock = threading.Lock()
    stop_event = threading.Event()

    t = threading.Thread(
        target=infer_loop,
        args=(model, device, shared, lock, stop_event, COCO, SCORE_THRESH),
        daemon=True,
    )
    t.start()

    # ---------------------------------- РАБОТА С ВИДЕО ----------------------------------
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        stop_event.set()
        raise RuntimeError(f"Не могу открыть видео: {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = int(1000 / fps) if fps and fps > 1 else 16

    show_roi = False
    pause = False
    frozen = None
    frame_idx = 0
    frame = None
    last_events = None  # Последние события — показываем пока не устарели

    # ---------------------------------- ТРЕКЕР И ЛОГГЕР ----------------------------------
    tracker = SimpleIoUTracker(iou_thresh=0.50, max_age=25)
    if logging_enable:
        logger = DetectionCsvLogger(
                                objs_path=objects_logs_path / f"{clip_name}_objs_dets.csv",
                                evt_path=events_logs_path / f"{clip_name}_events_dets.csv",
                                fps = fps
                                )

    # ---------------------------------- ГЛАВНЫЙ ЦИКЛ ----------------------------------
    try:
        while True:
            if not pause:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                # Каждые EVERY кадров — отправляем на инференс
                if frame_idx % EVERY == 0:
                    with lock:
                        shared["req_id"]    = frame_idx
                        shared["req_frame"] = frame.copy()

                base   = frame
                frozen = None
            else:
                if frozen is None and frame is not None:
                    frozen = frame.copy()
                base = frozen if frozen is not None else frame

            if base is None:
                break

            vis = base.copy()
            now = time.monotonic()

            with lock:
                detections = list(shared.get("detections", []))
                det_ts     = float(shared.get("detections_ts", 0.0))

            # ---------------------------------- ТРЕКИНГ И СОБЫТИЯ ----------------------------------
            if detections and (pause or (now - det_ts) <= MAX_STALE_SEC):

                # Обновляем трекер — получаем TrackedDetection для каждого объекта
                tracked = tracker.update(detections, frame_idx)

                # Проставляем зоны каждому объекту
                annotate_zones(tracked, crosswalks, risks)

                # Обновляем prev_in_risk / prev_in_crosswalk в треках
                update_track_zone_state(tracker, tracked)

                # Считаем события кадра
                last_events = compute_frame_events(frame_idx, tracked)
                if logging_enable:
                    logger.log(tracked, last_events)

                # Рисуем только объекты, попавшие в ROI
                for td in tracked:
                    if td.in_crosswalk or td.in_risk:
                        draw_tracked(vis, td)
            else:
                last_events = None

            # ---------------------------------- ОТРИСОВКА ----------------------------------
            if show_vid:
                if show_roi:
                    show_roi_polygons(frame=vis, crosswalks=crosswalks, risks=risks)

                draw_events(vis, last_events)

                cv2.putText(vis, "ESC/Q - exit | R - ROI | SPACE - pause",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Road Safety Detection", vis)

                key = cv2.waitKey(0 if pause else delay_ms) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("r"), ord("R")):
                    show_roi = not show_roi
                if key == ord(" "):
                    pause = not pause
    finally:
        stop_event.set()
        if logging_enable:
            logger.close()
        cap.release()
        cv2.destroyAllWindows()


def run_all(clips_dir):
    """
    Запускает детекцию на всех клипах в папке.
    Модель загружается один раз и переиспользуется.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").to(device).eval()
    print(f'Модель загружена, используется {device}')

    clips = sorted(clips_dir.glob("vid_*.mp4"))
    print(f"Найдено клипов: {len(clips)}")

    for clip_path in clips:
        clip_name = clip_path.stem
        print(f"\n→ Обработка: {clip_name}")
        try:
            main(
                roi_dir=ROIS_DIR,
                clips_dir=clips_dir,
                clip_name=clip_name,
                logging_enable=True,
                show_vid=False,
                model=model,
                device=device)
        except Exception as e:
            print(f"Ошибка: {e}")
            continue
        print(f"Инференс закончен")


if __name__ == "__main__":
    print('''Выберите типа запуска (1 или 2):
1 - Запустить все без вывода видео
2 - Запустить один клип
    ''')
    run_type = input('>> ')

    try:
        run_type = int(run_type)

        if run_type == 1:
            run_all(clips_dir=CLIPS_DIR)
        elif run_type == 2:
            clip_name = input("Введите название клипа (пример: vid_001_0000): ").strip()
            main(
                roi_dir=ROIS_DIR,
                clips_dir=CLIPS_DIR,
                clip_name=clip_name,
                logging_enable=False,
                show_vid=True,
                model=None)
        else:
            print('Введено что то кроме 1 или 2. Попробуйте еще раз')
    except ValueError:
        raise ValueError('Ошибка преобразования в int')
