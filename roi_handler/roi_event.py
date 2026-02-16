from roi_handler.roi_loader import point_in_polygon

# Список классов, который детектится как "Транспорт"
VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}

# Метчинг только внутри категорий
def category_of_label(label):
    if label == "person":
        return "person"
    if label in VEHICLE_LABELS:
        return "vehicle"
    return label

# Определение где находится объект относительно ROI
def anchor_point_xyxy(bbox_xyxy, mode="bottom_center"):
    x1, y1, x2, y2 = bbox_xyxy
    if mode == "center":
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    return int((x1 + x2) / 2), int(y2)

# Считает IoU двух ББ
def iou_xyxy(a, b):
    """
    :param a: Первый BB
    :param b: Второй BB
    :return: Возвращает число между 1 и 0 (То насколько 2 ББ перекрываются)
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = float(inter_w * inter_h)

    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


#
class TrackState:
    def __init__(self, track_id, category, bbox_xyxy, last_seen_frame):
        self.track_id = track_id
        self.category = category
        self.bbox_xyxy = bbox_xyxy
        self.last_seen_frame = last_seen_frame
        self.prev_in_risk = False
        self.prev_in_crosswalk = False


class TrackedDetection:
    def __init__(self, track_id, det, category, anchor_xy):
        self.track_id = track_id
        self.det = det
        self.category = category
        self.anchor_xy = anchor_xy

        self.in_crosswalk = False
        self.in_risk = False
        self.zone = "none"  # crosswalk | road | none

        self.entered_risk = False
        self.entered_crosswalk = False


class FrameEvents:
    def __init__(
        self,
        frame_idx,
        ped_on_crosswalk,
        ped_on_road,
        ped_entered_road_ids,
        vehicle_present,
        danger_same_zone,
        danger_ped_crosswalk_vehicle_risk,
    ):
        self.frame_idx = frame_idx
        self.ped_on_crosswalk = ped_on_crosswalk
        self.ped_on_road = ped_on_road
        self.ped_entered_road_ids = ped_entered_road_ids
        self.vehicle_present = vehicle_present
        self.danger_same_zone = danger_same_zone
        self.danger_ped_crosswalk_vehicle_risk = danger_ped_crosswalk_vehicle_risk


class SimpleIoUTracker:
    def __init__(self, iou_thresh=0.30, max_age=25):
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self._next_id = 1
        self._tracks = {}

    @property
    def tracks(self):
        return self._tracks

    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        return tid

    def _purge_stale(self, frame_idx):
        stale = [tid for tid, st in self._tracks.items() if (frame_idx - st.last_seen_frame) > self.max_age]
        for tid in stale:
            self._tracks.pop(tid, None)

    def update(self, detections, frame_idx):
        self._purge_stale(frame_idx)

        det_by_cat = {}
        for i, d in enumerate(detections):
            cat = category_of_label(d.label)
            det_by_cat.setdefault(cat, []).append((i, d))

        assigned_det_to_track = {}

        for cat, det_items in det_by_cat.items():
            det_indices = [i for i, _ in det_items]
            cat_tracks = [st for st in self._tracks.values() if st.category == cat]

            pairs = []
            for det_idx, det in det_items:
                for st in cat_tracks:
                    pairs.append((iou_xyxy(det.bbox_xyxy, st.bbox_xyxy), det_idx, st.track_id))

            pairs.sort(key=lambda x: x[0], reverse=True)

            used_dets = set()
            used_tracks = set()

            for score, det_idx, tid in pairs:
                if score < self.iou_thresh:
                    break
                if det_idx in used_dets or tid in used_tracks:
                    continue
                assigned_det_to_track[det_idx] = tid
                used_dets.add(det_idx)
                used_tracks.add(tid)

            for det_idx in det_indices:
                if det_idx in assigned_det_to_track:
                    continue
                tid = self._new_id()
                d = detections[det_idx]
                self._tracks[tid] = TrackState(
                    track_id=tid,
                    category=cat,
                    bbox_xyxy=d.bbox_xyxy,
                    last_seen_frame=frame_idx,
                )
                assigned_det_to_track[det_idx] = tid

        for det_idx, tid in assigned_det_to_track.items():
            d = detections[det_idx]
            st = self._tracks.get(tid)
            if st is None:
                continue
            st.bbox_xyxy = d.bbox_xyxy
            st.last_seen_frame = frame_idx

        out = []
        for i, d in enumerate(detections):
            tid = assigned_det_to_track.get(i)
            if tid is None:
                tid = self._new_id()
                self._tracks[tid] = TrackState(tid, category_of_label(d.label), d.bbox_xyxy, frame_idx)

            out.append(
                TrackedDetection(
                    track_id=tid,
                    det=d,
                    category=category_of_label(d.label),
                    anchor_xy=anchor_point_xyxy(d.bbox_xyxy, mode="bottom_center"),
                )
            )
        return out


def annotate_zones(tracked, crosswalk_poly, risk_poly):
    for t in tracked:
        x, y = t.anchor_xy
        t.in_crosswalk = point_in_polygon(crosswalk_poly, x, y)
        t.in_risk = point_in_polygon(risk_poly, x, y)

        if t.in_crosswalk:
            t.zone = "crosswalk"
        elif t.in_risk:
            t.zone = "road"
        else:
            t.zone = "none"


def update_track_zone_state(tracker, tracked):
    for t in tracked:
        st = tracker.tracks.get(t.track_id)
        if st is None:
            continue

        t.entered_risk = (not st.prev_in_risk) and t.in_risk
        t.entered_crosswalk = (not st.prev_in_crosswalk) and t.in_crosswalk

        st.prev_in_risk = t.in_risk
        st.prev_in_crosswalk = t.in_crosswalk


def compute_frame_events(frame_idx, tracked):
    peds = [t for t in tracked if t.category == "person"]
    vehs = [t for t in tracked if t.category == "vehicle"]

    ped_on_crosswalk = any(t.in_crosswalk for t in peds)
    ped_on_road = any((t.in_risk and not t.in_crosswalk) for t in peds)
    ped_entered_road_ids = [t.track_id for t in peds if t.entered_risk]

    vehicle_present = (len(vehs) > 0)

    danger_same_zone = (
        (any(t.in_risk for t in peds) and any(t.in_risk for t in vehs))
        or (any(t.in_crosswalk for t in peds) and any(t.in_crosswalk for t in vehs))
    )

    danger_ped_crosswalk_vehicle_risk = (any(t.in_crosswalk for t in peds) and any(t.in_risk for t in vehs))

    return FrameEvents(
        frame_idx=frame_idx,
        ped_on_crosswalk=ped_on_crosswalk,
        ped_on_road=ped_on_road,
        ped_entered_road_ids=ped_entered_road_ids,
        vehicle_present=vehicle_present,
        danger_same_zone=danger_same_zone,
        danger_ped_crosswalk_vehicle_risk=danger_ped_crosswalk_vehicle_risk,
    )