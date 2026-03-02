# ----------------------- ИМПОРТ БИБЛИОТЕК ПУТЕЙ И ОС -----------------------
import cv2
import time

from torch import inference_mode
from torchvision.transforms.functional import to_tensor

# ----------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -----------------------
def det_anchor_bottom_center(det):
    """
    :param det: Найденный объект
    :return: точка координат низа и середины
    """
    # det: (label, score, (x1,y1,x2,y2))
    x1, y1, x2, y2 = det[2]
    return (x1 + x2) // 2, y2


def point_in_any_polygon(polygons, x, y):
    # polygons: list[np.ndarray (N,1,2)]
    for poly in polygons:
        if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
            return True
    return False


def det_in_roi(det, crosswalks, risks):
    ax, ay = det_anchor_bottom_center(det)
    in_crosswalk = point_in_any_polygon(crosswalks, ax, ay)
    in_risk = point_in_any_polygon(risks, ax, ay)
    return in_crosswalk or in_risk


def predict(model, frame_bgr, device, score, classes):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = to_tensor(frame_rgb).to(device)

    with inference_mode():
        out = model([x])[0]  # boxes, labels, scores

    detections = []
    boxes = out["boxes"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()

    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        if sc < score:
            continue
        lab = int(lab)
        if lab not in classes:
            continue

        detections.append((classes[lab], float(sc), (int(x1), int(y1), int(x2), int(y2))))

    return detections


def infer_loop(model, device, shared, lock, stop_event, coco, score: float = 0.7):
    """Поток, который делает инференс, когда главный поток положил новый кадр."""
    last_seen_req_id = -1

    while not stop_event.is_set():
        time.sleep(0.001)

        with lock:
            req_id = shared.get("req_id", -1)
            frame = shared.get("req_frame", None)

        if req_id == last_seen_req_id or frame is None:
            continue

        dets = predict(model, frame, device, score, coco)
        ts = time.monotonic()

        with lock:
            shared["detections"] = dets
            shared["detections_ts"] = ts
            shared["detections_id"] = req_id
            last_seen_req_id = req_id


# def _infer_loop(model, device, shared, lock, stop_event):
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
#         dets = predict(model, frame, device)
#         ts = time.monotonic()
#
#         with lock:
#             shared["detections"] = dets
#             shared["detections_ts"] = ts
#             shared["detections_id"] = req_id
#             last_seen_req_id = req_id