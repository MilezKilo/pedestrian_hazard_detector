# ----------------------- ИМПОРТ БИБЛИОТЕК -----------------------
import cv2
import time
from dataclasses import dataclass

from torch import inference_mode
from torchvision.transforms.functional import to_tensor


# ----------------------- СТРУКТУРА ДЕТЕКЦИИ -----------------------
@dataclass
class Detection:
    """Единый формат детекции для всего проекта"""
    label: str
    score: float
    bbox_xyxy: tuple  # (x1, y1, x2, y2)


# ----------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -----------------------
# DEPRICATED - трекер и annotate_zones делают это сами
def det_anchor_bottom_center(det: Detection):
    """
    :param det: Найденный объект
    :return: точка координат низа и центра
    """
    x1, y1, x2, y2 = det.bbox_xyxy
    return (x1 + x2) // 2, y2


# DEPRICATED - используется только в det_in_roi
def point_in_any_polygon(polygons, x, y):
    """
    :param polygons: Полигоны для проверки
    :param x: Координата по горизонтали
    :param y: Координата по вертикали
    :return: Попадает ли точка по x и y в полигоны
    """
    for poly in polygons:
        if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
            return True
    return False


# DEPRICATED - заменена на annotate_zones из roi_event.py
def det_in_roi(det: Detection, crosswalks, risks):
    """
    Проверка попадает ли объект в одну из областей интересов (Фильтр)

    :param det: Найденные объекты
    :param crosswalks: Пешеходные переходы
    :param risks: Зона дорог
    :return: Возвращает результат функции point_in_any_polygon (True/False)
    """
    ax, ay = det_anchor_bottom_center(det)
    return point_in_any_polygon(crosswalks, ax, ay) or point_in_any_polygon(risks, ax, ay)


def predict(model, frame, device, score: float, classes: dict) -> list[Detection]:
    """
    Инференс модели на одном кадре

    :param model: Сама модель для детекции
    :param frame: Кадр для проверки
    :param device: CUDA/CPU
    :param score: Порог уверенности для проверки
    :param classes: Классы для проверки
    :return: Список найденных объектов
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = to_tensor(frame_rgb).to(device)

    with inference_mode():
        out = model([x])[0]

    detections = []
    boxes  = out["boxes"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()

    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        if sc < score:
            continue
        lab = int(lab)
        if lab not in classes:
            continue
        detections.append(Detection(
            label=classes[lab],
            score=float(sc),
            bbox_xyxy=(int(x1), int(y1), int(x2), int(y2))
        ))

    return detections


def infer_loop(model, device, shared, lock, stop_event, coco, score: float = 0.7):
    """
    Поток инференса: ждёт новый кадр, делает детекцию, кладёт результат в shared.

    :param model: Сама модель для детекции
    :param device: CUDA/CPU
    :param shared: Словарь для обмена данными между потоками
    :param lock: threading.Lock — защищает shared от одновременного доступа двух потоков (race condition)
    :param stop_event: Флаг остановки потока
    :param coco: Словарь для фильтрации классов COCO
    :param score: Порог уверенности для проверки
    """
    last_seen_req_id = -1

    while not stop_event.is_set():
        time.sleep(0.001)

        with lock:
            req_id = shared.get("req_id", -1)
            frame  = shared.get("req_frame", None)

        if req_id == last_seen_req_id or frame is None:
            continue

        dets = predict(model, frame, device, score, coco)
        ts   = time.monotonic()

        with lock:
            shared["detections"]    = dets
            shared["detections_ts"] = ts
            shared["detections_id"] = req_id
            last_seen_req_id        = req_id



# OLD CODE (DEPRICATED)
# # ----------------------- ИМПОРТ БИБЛИОТЕК ПУТЕЙ И ОС -----------------------
# import cv2
# import time
#
# from torch import inference_mode
# from torchvision.transforms.functional import to_tensor
#
# # ----------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -----------------------
# def det_anchor_bottom_center(det):
#     """
#     :param det: Найденный объект
#     :return: точка координат низа и середины
#     """
#     # det: (label, score, (x1,y1,x2,y2))
#     x1, y1, x2, y2 = det[2]
#     return (x1 + x2) // 2, y2
#
#
# def point_in_any_polygon(polygons, x, y):
#     # polygons: list[np.ndarray (N,1,2)]
#     for poly in polygons:
#         if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
#             return True
#     return False
#
#
# def det_in_roi(det, crosswalks, risks):
#     ax, ay = det_anchor_bottom_center(det)
#     in_crosswalk = point_in_any_polygon(crosswalks, ax, ay)
#     in_risk = point_in_any_polygon(risks, ax, ay)
#     return in_crosswalk or in_risk
#
#
# def predict(model, frame_bgr, device, score, classes):
#     frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#     x = to_tensor(frame_rgb).to(device)
#
#     with inference_mode():
#         out = model([x])[0]  # boxes, labels, scores
#
#     detections = []
#     boxes = out["boxes"].detach().cpu().numpy()
#     labels = out["labels"].detach().cpu().numpy()
#     scores = out["scores"].detach().cpu().numpy()
#
#     for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
#         if sc < score:
#             continue
#         lab = int(lab)
#         if lab not in classes:
#             continue
#
#         detections.append((classes[lab], float(sc), (int(x1), int(y1), int(x2), int(y2))))
#
#     return detections
#
#
# def infer_loop(model, device, shared, lock, stop_event, coco, score: float = 0.7):
#     """Поток, который делает инференс, когда главный поток положил новый кадр."""
#     last_seen_req_id = -1
#
#     while not stop_event.is_set():
#         time.sleep(0.001)
#
#         with lock:
#             req_id = shared.get("req_id", -1)
#             frame = shared.get("req_frame", None)
#
#         if req_id == last_seen_req_id or frame is None:
#             continue
#
#         dets = predict(model, frame, device, score, coco)
#         ts = time.monotonic()
#
#         with lock:
#             shared["detections"] = dets
#             shared["detections_ts"] = ts
#             shared["detections_id"] = req_id
#             last_seen_req_id = req_id
