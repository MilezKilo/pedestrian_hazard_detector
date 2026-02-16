# -------------------- БИБЛИОТЕКИ --------------------
import json
from pathlib import Path
import cv2
import numpy as np
from utils.paths import ROIS_DIR, CLIPS_DIR


# -------------------- ФУНКЦИИ РИСОВАНИЯ И СОХРАНЕНИЯ REGION OF INTEREST (ROI) --------------------
# Рисование линии (Если точек 2 и больше рисует линию)
def draw_polygon(img, pts, color, closed=False, thickness=2):
    """
    :param img: Изображение для рисования
    :param pts: Набор точек для создания региона
    :param color: Цвет рисования
    :param closed: Закрыт ли регион? (Если точки 3 и больше, последняя точка соединяется с первой)
    :param thickness: Толщина линии
    :return: Линию, или точку
    """
    if len(pts) >= 2:
        cv2.polylines(img, [pts], isClosed=closed, color=color, thickness=thickness)
    for (x, y) in pts:
        cv2.circle(img, (x, y), 5, color, -1)


# Сохраняет ROI
def save_rois(out_path: Path, clip_id: str, frame_w: int, frame_h: int, crosswalk, risk):
    """
    :param out_path: Путь сохранения
    :param clip_id: ID клипа
    :param frame_w: Ширина скрина из клипа
    :param frame_h: Высота скрина из клипа
    :param crosswalk: Регион пешеходного перехода
    :param risk: Регион дороги
    """
    ROIS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "clip_id": clip_id,
        "frame_size": [frame_w, frame_h],
        "polygons": {
            "crosswalk": crosswalk,
            "risk": risk
        }
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {out_path}")


# При каждом действии перерисовывает в кадре точки/линии связанные с ROI
def redraw_frame(frame, help_text, mode, crosswalk_pts, risk_pts):
    """
    :param frame: Кадр для рисования
    :param help_text: Текст помощи
    :param mode: Текущий режим рисования
    :param crosswalk_pts: Точки региона пешеходного перехода
    :param risk_pts: Точки региона дороги (Опасная зона)
    """
    img = frame.copy()

    cv2.rectangle(img, (5, 5), (520, 175), (0, 0, 0), -1)

    # Добавляет текст с режимом (Зебра/Дорога)
    cv2.putText(img, f"Mode: {mode}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Добавляет текст для помощи юзерам
    y = 70
    for t in help_text:
        cv2.putText(img, t, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
        y += 24

    if len(crosswalk_pts) > 0:
        pts = np.array(crosswalk_pts, dtype=np.int32)
        draw_polygon(img, pts, (0, 255, 0), closed=True)

    if len(risk_pts) > 0:
        pts = np.array(risk_pts, dtype=np.int32)
        draw_polygon(img, pts, (0, 0, 255), closed=True)

    cv2.imshow("ROI Tool", img)
    return img


# Рисование и сохранение ROI
def drawing():
    ROIS_DIR.mkdir(parents=True, exist_ok=True)

    clip_name = input(f"Название клипа, например vid_001_0000: ").strip()
    clip_path = CLIPS_DIR / f"{clip_name}.mp4"
    print(clip_path)

    # Если файл не найден выходим из цикла
    if not clip_path.exists():
        print("Файл не найден:", clip_path)
        return

    # Получаем id клипа и директорию для сохранения json данных
    clip_id = clip_path.stem
    out_path = ROIS_DIR / f"{clip_id}.json"

    # Захватываем видео и читаем первый кадр
    cap = cv2.VideoCapture(str(clip_path))
    ok, frame = cap.read()
    cap.release()

    # Если первый кадр не прочитан, выходим из цикла
    if not ok:
        print("Ошибка чтения кадра")
        return

    h, w = frame.shape[:2]

    # Точки координат интересов
    crosswalk_pts = []
    risk_pts = []

    # Текущий режим (Зебра/дорога)
    mode = "crosswalk"  # active polygon

    # Помощь с использованием
    help_text = [
        "LMB: add point",
        "BACKSPACE: undo last point",
        "TAB: switch polygon type (crosswalk/risk)",
        "S: save JSON",
        "ESC/Q: exit without saving"
    ]

    shown = redraw_frame(frame, help_text, mode, crosswalk_pts, risk_pts)

    def on_mouse(event, x, y, flags, param):
        nonlocal crosswalk_pts, risk_pts, shown
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == "crosswalk":
                crosswalk_pts.append([int(x), int(y)])
            else:
                risk_pts.append([int(x), int(y)])
            shown = redraw_frame(frame, help_text, mode, crosswalk_pts, risk_pts)

    cv2.namedWindow("ROI Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Tool", on_mouse)

    print("\n--- ROI Tool ---")
    print("Сначала накликай точки для CROSSWALK (Пешеходный переход), потом TAB -> RISK (дорога).")
    print("Сохраняй клавишей S.\n")

    # main loop
    while True:
        key = cv2.waitKey(0)

        # ESC or q (Кнопка выхода без сохранения)
        if key in [27, ord('q'), ord('Q')]:
            print("Выход без сохранения.")
            break

        # TAB (Кнопка смены режима)
        if key == 9:
            mode = "risk" if mode == "crosswalk" else "crosswalk"
            shown = redraw_frame(frame, help_text, mode, crosswalk_pts, risk_pts)
            continue

        # BACKSPACE Удаляет последнюю точку из списка
        if key in [8, 127]:
            if mode == "crosswalk" and crosswalk_pts:
                crosswalk_pts.pop()
            if mode == "risk" and risk_pts:
                risk_pts.pop()
            shown = redraw_frame(frame, help_text, mode, crosswalk_pts, risk_pts)
            continue

        # S Кнопка сохранения
        if key in [ord('s'), ord('S')]:
            if len(crosswalk_pts) < 3 or len(risk_pts) < 3:
                print("Нужно минимум 3 точки для каждого полигона (crosswalk и risk).")
                continue

            save_rois(out_path, clip_id, w, h, crosswalk_pts, risk_pts)
            break

    cv2.destroyAllWindows()


# -------------------- ЗАПУСК --------------------
if __name__ == "__main__":
    drawing()