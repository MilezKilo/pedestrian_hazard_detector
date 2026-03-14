# ----------------------- БИБЛИОТЕКИ -----------------------
import csv
from pathlib import Path

# ----------------------- ЛОГГЕР ОБЪЕКТОВ -----------------------
class DetectionCsvLogger:
    DET_FIELDS = ["frame_idx", "timestamp_sec", "track_id", "label", "score", "zone", "x1", "y1", "x2", "y2"]
    EVT_FIELDS = ["frame_idx", "timestamp_sec", "ped_on_crosswalk", "ped_on_road", "ped_entered_road", "vehicle_present", "danger_same_zone"]

    def __init__(self, objs_path: Path, evt_path: Path, fps: float = 25):
        # det_path = path.parent / (path.stem + "_detections.csv")
        # evt_path = path.parent / (path.stem + "_events.csv")
        self.fps = fps if fps and fps > 1 else 25.0

        self._obj_file = open(objs_path, "w", newline="", encoding="utf-8")
        self._evt_file = open(evt_path, "w", newline="", encoding="utf-8")

        self._obj_writer = csv.DictWriter(self._obj_file, fieldnames=self.DET_FIELDS)
        self._evt_writer = csv.DictWriter(self._evt_file, fieldnames=self.EVT_FIELDS)

        self._obj_writer.writeheader()
        self._evt_writer.writeheader()

    def log(self, tracked, events):
        if events is None:
            return
        ts = round(events.frame_idx / self.fps, 2)

        # Объекты
        for td in tracked:
            if not (td.in_crosswalk or td.in_risk):
                continue
            x1, y1, x2, y2 = td.det.bbox_xyxy
            self._obj_writer.writerow({
                "frame_idx": events.frame_idx,
                "timestamp_sec": ts,
                "track_id":  td.track_id,
                "label":     td.det.label,
                "score":     round(td.det.score, 3),
                "zone":      td.zone,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

        # События — одна строка на кадр
        self._evt_writer.writerow({
            "frame_idx":        events.frame_idx,
            "timestamp_sec": ts,
            "ped_on_crosswalk": events.ped_on_crosswalk,
            "ped_on_road":      events.ped_on_road,
            "ped_entered_road": bool(events.ped_entered_road_ids),
            "vehicle_present":  events.vehicle_present,
            "danger_same_zone": events.danger_same_zone,
        })

    def close(self):
        self._obj_file.close()
        self._evt_file.close()