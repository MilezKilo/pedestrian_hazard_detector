# ----------------------- ИМПОРТ БИБЛИОТЕК -----------------------
from pathlib import Path

# Корневая директория проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Директория с данными (Клипы, видео, ROI)
DATA_DIR = PROJECT_ROOT / "data"
CLIPS_DIR = DATA_DIR / "video_clips"
RAW_VID_DIR = DATA_DIR / 'video_raw'
ROIS_DIR = DATA_DIR / "rois"

# Директория с моделями НН
MODELS_DIR = PROJECT_ROOT / "models"


# Возвращает клип для отображения в cv2
def clip_path(clip_id):
    return CLIPS_DIR / f"{clip_id}.mp4"

# Возвращает ROI в формате json
def roi_path(clip_id):
    return ROIS_DIR / f"{clip_id}.json"