# -------------------- БИБЛИОТЕКИ --------------------
import csv
import cv2
from pathlib import Path
from utils.paths import CLIPS_DIR, GT_DIR, FONT_PATH
from utils.drawing_handler import put_text_cyrillic



# -------------------- КОНСТАНТЫ --------------------
# Поля для csv файла
GT_FIELDS = ["clip_name", "start_sec", "end_sec"]

# Цвета
RED    = (0, 0, 255)
GREEN  = (0, 200, 0)
WHITE  = (255, 255, 255)
YELLOW = (0, 215, 255)


# -------------------- ЗАГРУЗКА РАЗМЕТКИ --------------------
def load_existing(clip_name: str) -> list[dict]:
    """
    Загрузка существующих разметок

    :param clip_name: Имя клипа
    :return: Список разметок по клипу
    """
    gt_path = GT_DIR / f"{clip_name}_gt.csv"
    if not gt_path.exists():
        return []
    with open(gt_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# -------------------- СОХРАНЕНИЕ РАЗМЕТКИ --------------------
def save_clip(clip_name: str, episodes: list[dict]):
    """
    Сохраняет разметку по клипу

    :param clip_name: Имя клипа
    :param episodes: Временные эпизоды с эвентом
    """
    GT_DIR.mkdir(parents=True, exist_ok=True)
    gt_path = GT_DIR / f"{clip_name}_gt.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GT_FIELDS)
        writer.writeheader()
        writer.writerows(episodes)


# -------------------- ОТОБРАЖЕНИЕ UI --------------------
def draw_ui(frame, clip_name, timestamp, is_danger, danger_start, episodes):
    """
    Рисует интерфейс поверх кадра.

    :param frame: Кадр для отрисовки
    :param clip_name: Имя клипа
    :param timestamp: Временной промежуток
    :param is_danger: Флаг опасности (Эвент)
    :param danger_start: Начало эвента
    :param episodes: Количество эпизодов с эвентами в клипе/видео
    """
    h, w = frame.shape[:2]

    # Статус записи
    if is_danger and danger_start is not None:
        duration = round(timestamp - danger_start, 1)
        status = f"[ЗАПИСЬ] {danger_start:.1f}s -> {timestamp:.1f}s  ({duration}s)"
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 180), -1)
        put_text_cyrillic(
            frame=frame,
            text=status,
            pos=(10, 15),
            font_path=FONT_PATH,
            font_size=22,
            color=WHITE)
    else:
        put_text_cyrillic(
            frame=frame,
            text="Нажми D чтобы обозначить опасный эпизод",
            pos=(10, 35),
            font_path=FONT_PATH,
            font_size=22,
            color=GREEN)

    # Клип и время
    put_text_cyrillic(
        frame=frame,
        text=f"{clip_name}  |  {timestamp:.2f} sec",
        pos=(10, h - 45),
        font_path=FONT_PATH,
        font_size=22,
        color=WHITE)

    # Сколько эпизодов размечено
    put_text_cyrillic(
        frame=frame,
        text=f"Эпизодов: {len(episodes)}  |  D=эпизод  SPACE=пауза  R=начало  Q=дальше",
        pos=(10, h - 15),
        font_path=FONT_PATH,
        font_size=22,
        color=YELLOW)


# -------------------- ОТОБРАЖЕНИЕ ВРЕМЕНИ ЭПИЗОДА --------------------
def draw_timeline(frame, episodes, danger_start, timestamp, total_sec):
    """Рисует полоску прогресса с отмеченными эпизодами."""
    h, w = frame.shape[:2]
    bar_y = h - 8

    cv2.rectangle(frame, (0, bar_y - 4), (w, bar_y + 4), (50, 50, 50), -1)

    for ep in episodes:
        x1 = int(float(ep["start_sec"]) / total_sec * w)
        x2 = int(float(ep["end_sec"]) / total_sec * w)
        cv2.rectangle(frame, (x1, bar_y - 4), (x2, bar_y + 4), RED, -1)

    if danger_start is not None:
        x1 = int(danger_start / total_sec * w)
        x2 = int(timestamp / total_sec * w)
        cv2.rectangle(frame, (x1, bar_y - 4), (x2, bar_y + 4), (0, 0, 140), -1)

    cx = int(timestamp / total_sec * w)
    cv2.circle(frame, (cx, bar_y), 6, WHITE, -1)


# ----------------------- РАЗМЕТКА ОДНОГО КЛИПА -----------------------
def annotate_clip(clip_path: Path) -> list[dict]:
    clip_name = clip_path.stem
    episodes = load_existing(clip_name)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"Не могу открыть: {clip_path}")
        return episodes

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps

    is_danger = False
    danger_start = None
    pause = False
    frame = None

    print(f"  Длина: {total_sec:.1f}s  |  FPS: {fps:.1f}  |  Уже размечено: {len(episodes)} эп.")

    try:
        while True:
            if not pause:
                ok, frame = cap.read()
                if not ok:
                    break

            if frame is None:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = round(frame_idx / fps, 2)

            vis = frame.copy()
            draw_timeline(vis, episodes, danger_start, timestamp, total_sec)
            draw_ui(vis, clip_name, timestamp, is_danger, danger_start, episodes)

            cv2.imshow("Annotator", vis)
            key = cv2.waitKey(1 if not pause else 0) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                if is_danger and danger_start is not None:
                    episodes.append({
                        "clip_name": clip_name,
                        "start_sec": round(danger_start, 2),
                        "end_sec": round(timestamp, 2),
                    })
                    print(f"  + эпизод {danger_start:.1f}s – {timestamp:.1f}s (закрыт автоматически)")
                    is_danger = False
                    danger_start = None
                break

            elif key in (ord("d"), ord("D")):
                if not is_danger:
                    is_danger = True
                    danger_start = timestamp
                    print(f"  ● начало: {timestamp:.2f}s")
                else:
                    is_danger = False
                    episodes.append({
                        "clip_name": clip_name,
                        "start_sec": round(danger_start, 2),
                        "end_sec": round(timestamp, 2),
                    })
                    print(f"  ✓ эпизод {danger_start:.1f}s – {timestamp:.1f}s  "
                          f"({round(timestamp - danger_start, 1)}s)")
                    danger_start = None

            elif key == ord(" "):
                pause = not pause

            elif key in (ord("r"), ord("R")):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                is_danger = False
                danger_start = None
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return episodes


# -------------------- МЕЙН ФУНКЦИЯ --------------------
def main():
    # ЗАГРУЗКА КЛИПОВ
    clips = sorted(CLIPS_DIR.glob("vid_*.mp4"))
    if not clips:
        print(f"Клипы не найдены в {CLIPS_DIR}")
        return

    # ОТОБРАЖЕНИЕ И ВЫБОР КЛИПОВ
    print("Доступные клипы:")
    for i, c in enumerate(clips):
        already = len(load_existing(c.stem))
        marker = f"  ({already} эп.)" if already else ""
        print(f"  {i + 1:>3}. {c.stem}{marker}")

    raw = input("\nВведите номера клипов через запятую (пример: 1,5,12): ").strip()

    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]
        selected = [clips[i] for i in indices if 0 <= i < len(clips)]
    except ValueError:
        print("Неверный ввод")
        return

    # ЕСЛИ НЕ ВВЕДЕНЫ НОМЕРА КЛИПОВ, ТО ВЫХОДИМ ИЗ ФУНКЦИИ
    if not selected:
        print("Ничего не выбрано")
        return

    print(f"\nВыбрано клипов: {len(selected)}")
    print("Управление: D=эпизод  SPACE=пауза  R=сначала  Q=следующий клип\n")

    for clip_path in selected:
        clip_name = clip_path.stem
        print(f"→ {clip_name}")

        episodes = annotate_clip(clip_path)

        if episodes:
            save_clip(clip_name, episodes)
            print(f"Сохранено: {GT_DIR / f'{clip_name}_gt.csv'}  ({len(episodes)} эп.)")
        else:
            print(f"Эпизодов нет, файл не создан")

    print(f"\nГотово. Файлы в: {GT_DIR}")


if __name__ == "__main__":
    main()
