# ----------------------- ИМПОРТ БИБЛИОТЕК -----------------------
from pathlib import Path
import re
import platform

# ----------------------- КОНСТАНТЫ -----------------------
# Корневая директория проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Директория с данными (Клипы, видео, ROI, логи)
DATA_DIR = PROJECT_ROOT / "data"
CLIPS_DIR = DATA_DIR / "video_clips"
RAW_VID_DIR = DATA_DIR / 'video_raw'
ROIS_DIR = DATA_DIR / "rois"
LOGS_DIR = DATA_DIR / "csv_logs"
EVENTS_DIR = LOGS_DIR / 'events_logs'
OBJECTS_DIR = LOGS_DIR / 'objects_logs'
GT_DIR = LOGS_DIR / 'ground_truth'
METRICS_DIR = LOGS_DIR / 'metrics'

# Директория с моделями НН
MODELS_DIR = PROJECT_ROOT / "detector" / 'models'

# Шрифты
FONT_PATH = 'C:/Windows/Fonts/arial.ttf' if platform.system() == 'Windows' else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

# Регулярное выражение (т.е. шаблон поиска)
CLIP_SUFFIX_RE = re.compile(r"_\d{4}$")


def dirs_maker():
    """
    При первичной настройке создает отсутствующие папки.

    """
    for directory in [CLIPS_DIR,
                      RAW_VID_DIR,
                      ROIS_DIR,
                      LOGS_DIR,
                      EVENTS_DIR,
                      GT_DIR,
                      METRICS_DIR,
                      OBJECTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def roi_from_clip(clip_name):
    """
    Ищет и убирает лишние символы

    _  — символ подчёркивания

    \\d — цифра (0–9)

    {4} — ровно 4 цифры

    $  — конец строки

    Пример:

      vid_001_0000 -> vid_001

      vid_001      -> vid_001
    """
    return CLIP_SUFFIX_RE.sub("", str(clip_name).strip())


def roi_check(roi_dir: Path, return_roi_path: bool = False, clip_name: str = None):
    """
    Проверка пути к директории с ROI

    :param roi_dir: Сама директория
    :param return_roi_path: Флаг, возвращать ли JSON файл, или path директории
    :param clip_name: Название клипа, по которому будет возвращаться JSON с данными ROI
    :return: Либо path, либо JSON файл с данными ROI
    """
    if not roi_dir.exists():
        raise FileNotFoundError(f"Директория с ROI не найдена: {roi_dir}")

    # Если True, то возвращает JSON файл, содержащий ROI
    if return_roi_path:
        if clip_name is None:
            raise ValueError('Введено пустое название.')

        # Удаляем суффикс вида _0000 в конце, чтобы получить roi_id
        roi_id = roi_from_clip(clip_name=clip_name)
        roi_path = roi_dir / f'{roi_id}.json'

        # Если файл по пути не существует, выходим с ошибкой из функции, если он есть, возвращаем файл
        if not roi_path.is_file():
            raise FileNotFoundError(f"JSON файл с ROI не найден {roi_path}")

        return roi_path
    # Если False, то возвращает директорию
    return roi_dir


def clip_check(clips_dir: Path, clip_name: str):
    """
    Проверка пути к директории с клипами

    :param clips_dir: Сама директория
    :param clip_name: Имя клипа
    :return: Возвращает клип
    """
    if not clips_dir.exists():
        raise FileNotFoundError(f"Директория с клипами не найдена: {clips_dir}")

    # Получаем путь к клипу
    clip_path = clips_dir / f'{clip_name}.mp4'

    # Проверка существует ли клип
    if not clip_path.is_file():
        raise FileNotFoundError(f'Клип не найден {clip_path}')

    return clip_path # path type


def logger_check(logger_dir: Path, logger_type: str='object'):
    """
    Проверка путей к директории с логами

    :param logger_dir: Сама директория
    :param logger_type: Тип логгера (Объект/Эвент)
    """
    if not logger_dir.exists():
        raise FileNotFoundError(f"Директория с логами не найдена: {logger_dir}")

    logs = None

    if logger_type == 'object':
        logs = logger_dir / 'objects_logs'
    elif logger_type == 'event':
        logs = logger_dir / 'events_logs'

    if logs is None:
        raise ValueError(f"Неизвестный тип логгера: {logger_type}")

    logs.mkdir(parents=True, exist_ok=True)
    return logs


# Запускать при первичной настройке.
if __name__ == '__main__':
    dirs_maker()