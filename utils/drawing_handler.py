# ----------------------- ИМПОРТ БИБЛИОТЕК -----------------------
import cv2

# ----------------------- ФУНКЦИИ -----------------------
def show_roi_polygons(frame, crosswalks, risks):
    """
    Функция отрисовки областей интересов

    :param frame: Кадр для отрисовки
    :param crosswalks: Области пешеходных пероеходов
    :param risks: Области опасных зон
    """
    # crosswalks и risks — списки полигонов (каждый полигон: (N,1,2))
    for poly in crosswalks:
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)

    for poly in risks:
        cv2.polylines(frame, [poly], True, (0, 0, 255), 2)


def draw(frame, det):
    """
    Рисует на кадре 1 детекцию

    :param frame: Кадр для отрисовки
    :param det: Обнаруженный объект
    """
    label, score, (x1, y1, x2, y2) = det
    color = (0, 255, 0) if label == "person" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{label}:{score:.2f}", (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

