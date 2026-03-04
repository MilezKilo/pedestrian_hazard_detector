# ----------------------- ИМПОРТ БИБЛИОТЕК -----------------------
import cv2


# ----------------------- КОНСТАНТЫ ЦВЕТОВ -----------------------
GREEN_COLOR  = (0, 255, 0)
RED_COLOR    = (0, 0, 255)
ORANGE_COLOR = (0, 165, 255)
WHITE_COLOR  = (255, 255, 255) # НЕ ИСПОЛЬЗУЕТСЯ НА ДАННЫЙ МОМЕНТ


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

    Подсвечивает опасные ситуации.

    :param frame: Кадр для отрисовки
    :param events Текущие активные события
    """
    if events is None:
        return

    # ----------------------- ЛОКАЛЬНЫЕ КОНСТАНТЫ -----------------------
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    padding    = 8
    line_h     = 28

    w = frame.shape[1]
    lines = []

    # ----------------------- ОПИСАНИЕ СОБЫТИЙ НА ЭКРАНЕ -----------------------
    if events.ped_on_crosswalk:
        lines.append(("PED ON CROSSWALK", ORANGE_COLOR))
    if events.ped_on_road:
        lines.append(("PED ON ROAD", ORANGE_COLOR))
    if events.ped_entered_road_ids:
        ids = ",".join(str(i) for i in events.ped_entered_road_ids)
        lines.append((f"PED ENTERED ROAD: #{ids}", ORANGE_COLOR))
    if events.vehicle_present:
        lines.append(("VEHICLE IN ZONE", RED_COLOR))
    if events.danger_same_zone:
        lines.append(("!! DANGER: PED+VEH SAME ZONE !!", RED_COLOR))
    if events.danger_ped_crosswalk_vehicle_risk:
        lines.append(("!! DANGER: PED CROSSWALK + VEH ROAD !!", RED_COLOR))

    if not lines:
        return

    box_w = max(cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t, _ in lines) + padding * 2
    box_h = line_h * len(lines) + padding

    x0 = w - box_w - 10
    y0 = 10

    # Полупрозрачный фон
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (text, color) in enumerate(lines):
        ty = y0 + padding + line_h * i + 18
        cv2.putText(frame, text, (x0 + padding, ty), font, font_scale, color, thickness)



# OLD CODE (DEPRICATED)
# # ----------------------- ИМПОРТ БИБЛИОТЕК -----------------------
# import cv2
#
# # ----------------------- ФУНКЦИИ -----------------------
# def show_roi_polygons(frame, crosswalks, risks):
#     """
#     Функция отрисовки областей интересов
#
#     :param frame: Кадр для отрисовки
#     :param crosswalks: Области пешеходных пероеходов
#     :param risks: Области опасных зон
#     """
#     # crosswalks и risks — списки полигонов (каждый полигон: (N,1,2))
#     for poly in crosswalks:
#         cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
#
#     for poly in risks:
#         cv2.polylines(frame, [poly], True, (0, 0, 255), 2)
#
#
# def draw(frame, det):
#     """
#     Рисует на кадре 1 детекцию
#
#     :param frame: Кадр для отрисовки
#     :param det: Обнаруженный объект
#     """
#     label, score, (x1, y1, x2, y2) = det
#     color = (0, 255, 0) if label == "person" else (0, 0, 255)
#     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#     cv2.putText(frame, f"{label}:{score:.2f}", (x1, max(20, y1 - 7)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# def draw(frame, det):
#     """
#     Рисует на кадре 1 детекцию
#
#     :param frame: Кадр для отрисовки
#     :param det: Обнаруженный объект
#     """
#     # Поддержка обоих форматов
#     if hasattr(det, "label"):
#         label, score, (x1, y1, x2, y2) = det.label, det.score, det.bbox_xyxy
#     else:
#         label, score, (x1, y1, x2, y2) = det
#
#     color = COLOR_PERSON if label == "person" else COLOR_VEHICLE
#     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#     cv2.putText(frame, f"{label}:{score:.2f}", (x1, max(20, y1 - 7)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

