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
MODELS_DIR = PROJECT_ROOT / "detector" / 'models'


def paths_check(roi_dir, clips_dir, clip_id):
    if not clips_dir.exists():
        raise FileNotFoundError(f"Директория с клипами не найдена: {clips_dir}")

    # Получаем путь к клипу
    clip_path = str(clips_dir / clip_id) + '.mp4'

    # Проверка существует ли клип
    if not Path(clip_path).is_file():
        raise FileNotFoundError(f'Клип не найден {clip_path}')

    # Проверка существует ли папка с ROI
    if not roi_dir.exists():
        raise FileNotFoundError(f"Директория с ROI не найдена: {roi_dir}")

    # Если существует, то создаем переменную с ROI
    roi_id = clip_id[:-5]
    roi_path = str(roi_dir / roi_id) + '.json'

    # Проверка существует ли ROI
    if not Path(roi_path).is_file():
        raise FileNotFoundError(f"ROI не найден {roi_path}")

    return clip_path, roi_path