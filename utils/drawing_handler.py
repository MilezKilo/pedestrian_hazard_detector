# ----------------------- ИМПОРТ БИБЛИОТЕК -----------------------
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from utils.paths import FONT_PATH

# ----------------------- КОНСТАНТЫ ЦВЕТОВ -----------------------
GREEN_COLOR  = (0, 255, 0)
RED_COLOR    = (0, 0, 255)
ORANGE_COLOR = (0, 165, 255)


# ----------------------- ОТОБРАЖЕНИЕ КИРИЛЛИЦЫ НА ЭКРАНЕ -----------------------
def put_text_cyrillic(frame, text, pos, font_path, font_size, color):
    """
    Отрисовка текста с поддержкой кириллицы через Pillow.

    :param frame: Кадр OpenCV (BGR)
    :param text: Текст для отрисовки
    :param pos: Позиция (x, y)
    :param font_path: Путь к .ttf шрифту
    :param font_size: Размер шрифта
    :param color: Цвет BGR
    """
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    # Pillow использует RGB, а не BGR
    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ----------------------- ОТРИСОВКА ROI -----------------------
def show_roi_polygons(frame, crosswalks, risks):
    """
    Функция отрисовки областей интересов

    :param frame: Кадр для отрисовки
    :param crosswalks: Области пешеходных пероеходов
    :param risks: Области опасных зон
    """
    for poly in crosswalks:
        cv2.polylines(frame, [poly], True, GREEN_COLOR, 2)
    for poly in risks:
        cv2.polylines(frame, [poly], True, RED_COLOR, 2)


# ----------------------- ОТРИСОВКА ДЕТЕКЦИИ -----------------------
def draw_tracked(frame, tracked_det):
    """
    Рисует bbox, track_id и зону для TrackedDetection.
    Цвет зависит от зоны: crosswalk=оранжевый, road=красный, none=обычный.

    :param frame: Кадр для отрисовки
    :param tracked_det: Обнаруженный объект
    """
    det = tracked_det.det
    x1, y1, x2, y2 = det.bbox_xyxy

    if tracked_det.zone == "crosswalk":
        color = ORANGE_COLOR
    elif tracked_det.zone == "road":
        color = RED_COLOR
    else:
        color = GREEN_COLOR if tracked_det.category == "person" else RED_COLOR

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"#{tracked_det.track_id} {det.label}:{det.score:.2f}",
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Якорная точка
    ax, ay = tracked_det.anchor_xy
    cv2.circle(frame, (ax, ay), 4, color, -1)


# ----------------------- ОТРИСОВКА СОБЫТИЙ -----------------------
def draw_events(frame, events):
    """
    Отрисовывает плашку событий в правом верхнем углу кадра.
    Обычные события — оранжевым снизу, опасные — красным сверху.

    :param frame: Кадр для отрисовки
    :param events: Текущие активные события
    """
    if events is None:
        return

    normal = []
    danger = []

    # -------- Обычные события --------
    if events.ped_on_crosswalk:
        normal.append("Пешеход на переходе")
    if events.ped_on_road:
        normal.append("Пешеход на дороге")
    if events.vehicle_present:
        normal.append("Транспорт обнаружен")

    # -------- Опасные события --------
    if events.danger_same_zone:
        danger.append("ОПАСНОСТЬ: Пешеход и транспорт на дороге")

    if not normal and not danger:
        return

    padding = 8
    line_h = 28

    w = frame.shape[1]
    all_lines = danger + normal  # опасные сверху
    all_colors = [RED_COLOR] * len(danger) + [ORANGE_COLOR] * len(normal)

    box_w = 550
    box_h = line_h * len(all_lines) + padding

    x0 = w - box_w - 10
    y0 = 10

    # Разделяем фон: красный для опасных, тёмный для обычных
    overlay = frame.copy()
    if danger:
        danger_h = line_h * len(danger) + padding // 2
        cv2.rectangle(overlay, (x0, y0+20), (x0 + box_w, y0 + danger_h+20), (0, 0, 60), -1)
        cv2.rectangle(overlay, (x0, y0 + danger_h+20), (x0 + box_w, y0 + box_h+20), (60, 60, 60), -1)
    else:
        cv2.rectangle(overlay, (x0, y0+20), (x0 + box_w, y0 + box_h+20), (60, 60, 60), -1)

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (text, color) in enumerate(zip(all_lines, all_colors)):
        ty = y0 + padding + line_h * i + 18
        put_text_cyrillic(frame, text, (x0 + padding, ty), FONT_PATH, 22, color)
