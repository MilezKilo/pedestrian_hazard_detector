# ----------------------- ИМПОРТ БИБЛИОТЕК -----------------------
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor


COCO = {
    1: "person",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

ALLOWED_IDS = set(COCO.keys())


class Detection:
    def __init__(self, class_id, label, score, bbox_xyxy):
        self.class_id = int(class_id)
        self.label = str(label)
        self.score = float(score)
        self.bbox_xyxy = tuple(int(v) for v in bbox_xyxy)


class TorchDetector:
    def __init__(self, score_thresh=0.5, device=None):
        self.score_thresh = float(score_thresh)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.benchmark = True

        self.model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        self.model.to(self.device).eval()

    def predict(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = to_tensor(frame_rgb).to(self.device)

        with torch.inference_mode():
            out = self.model([x])[0]  # boxes, labels, scores

        boxes = out["boxes"]
        labels = out["labels"]
        scores = out["scores"]

        keep = scores >= self.score_thresh
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        if boxes.numel() == 0:
            return []

        allowed = torch.tensor(list(ALLOWED_IDS), device=self.device)
        mask = (labels[:, None] == allowed[None, :]).any(dim=1)

        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        if boxes.numel() == 0:
            return []

        boxes = boxes.to("cpu").int().numpy()
        labels = labels.to("cpu").numpy()
        scores = scores.to("cpu").numpy()

        dets = []
        for (x1, y1, x2, y2), cid, sc in zip(boxes, labels, scores):
            cid = int(cid)
            label = COCO.get(cid, str(cid))
            dets.append(Detection(cid, label, float(sc), (x1, y1, x2, y2)))
        return dets


