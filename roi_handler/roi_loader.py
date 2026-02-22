# -------------------- БИБЛИОТЕКИ --------------------
from json import load as json_load
from pathlib import Path
from cv2 import pointPolygonTest
from numpy import array, ndarray, int32


# -------------------- МЕТОДЫ ЗАГРУЗКИ REGION OF INTEREST (ROI) --------------------
# Загрузка ROI с масштабированием под размер текущего кадра
def load_roi(roi_json_path: Path) -> dict:
    """
    Открывает JSON файл ROI и возвращает полигоны в формате np.ndarray shape (-1,1,2)
    """
    with open(roi_json_path, "r", encoding="utf-8") as f:
        data = json_load(f)

    polygons = data.get("polygons", data)

    crosswalk = array(polygons["crosswalk"], dtype=int32).reshape((-1, 1, 2))
    risk = array(polygons["risk"], dtype=int32).reshape((-1, 1, 2))

    return {"crosswalk": crosswalk, "risk": risk}


# Возвращает внутри полигона точка, или нет
def point_in_polygon(polygon: ndarray, x: int, y: int) -> bool:
    """
    Если возвращаемое число:
    - больше 0, то точка внутри полигона
    - равно 0, на границе полигона
    - меньше 0, снаружи полигона
    """
    return pointPolygonTest(polygon, (float(x), float(y)), False) >= 0








# DEPRICATED

# Возвращает центр ББ
# def bbox_center(x1, y1, x2, y2):
#     """
#     :param x1: Левый верхний угол
#     :param y1:
#     :param x2: Правый нижний угол
#     :param y2:
#     :return: Центр bounding box'a
#     """
#     return int((x1 + x2) / 2), int((y1 + y2) / 2)