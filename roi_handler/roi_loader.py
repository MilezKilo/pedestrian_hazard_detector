# ---------------------------------------- БИБЛИОТЕКИ ----------------------------------------
from json import load as json_load
from pathlib import Path
import numpy as np


# ---------------------------------------- МЕТОДЫ  ----------------------------------------
def poly_to_cv(points):
    """
    Функция проверки и изменение размерности массива. Было (N, 2), Стало (N, 1, 2)

    :param points: Массив точек [[x,y], ...]
    :return np.ndarray int32 shape (-1,1,2)
    """
    arr = np.asarray(points, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Размерность должна быть (N,2), получили {arr.shape}")
    return arr.reshape((-1, 1, 2))



def load_roi(roi_json_path: Path) -> dict:
    """
    Открывает JSON файл ROI и возвращает полигоны


    :param roi_json_path: Пусть к папке с ROI
    :return: Возвращает словарь с 2 типами полигонов
    """
    with open(roi_json_path, "r", encoding="utf-8") as f:
        data = json_load(f)

    polygons = data.get("polygons", data)

    crosswalks = [poly_to_cv(p) for p in polygons["crosswalks"]]
    risks = [poly_to_cv(p) for p in polygons["risks"]]

    return {"crosswalks": crosswalks, "risks": risks}








