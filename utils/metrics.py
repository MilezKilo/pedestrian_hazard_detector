# ----------------------- БИБЛИОТЕКИ -----------------------
import csv
import numpy as np
from utils.paths import GT_DIR, METRICS_DIR, EVENTS_DIR

# ----------------------- ПАРАМЕТРЫ -----------------------
IOU_THRESH = 0.5  # минимальный IoU для совпадения эпизода
GAP_SEC = 1.0  # допуск при склейке соседних True-кадров в эпизод


# -------------------------- ЗАГРУЗКА ДАННЫХ --------------------------
def load_gt(clip_name: str) -> list[tuple[float, float]]:
    """
    Загружает GT-эпизоды для клипа.

    :param clip_name: Имя клипа (например, vid_000_0005)
    :return: Список кортежей (start_sec, end_sec)
    """
    gt_path = GT_DIR / f"{clip_name}_gt.csv"
    if not gt_path.exists():
        return []
    with open(gt_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    episodes = []
    for r in rows:
        try:
            s, e = float(r["start_sec"]), float(r["end_sec"])
            if e > s:
                episodes.append((s, e))
        except (ValueError, KeyError):
            continue
    return episodes


def load_system_episodes(clip_name: str) -> list[tuple[float, float]]:
    """
    Загружает лог событий и склеивает соседние danger_same_zone=True
    в непрерывные эпизоды с допуском GAP_SEC.

    :param clip_name: Имя клипа
    :return: Список кортежей (start_sec, end_sec)
    """
    evt_path = EVENTS_DIR / f"{clip_name}_events_dets.csv"
    if not evt_path.exists():
        return []

    with open(evt_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    danger_times = sorted([
        float(r["timestamp_sec"])
        for r in rows
        if r.get("danger_same_zone", "").strip() == "True"
    ])

    if not danger_times:
        return []

    # Склейка в эпизоды
    episodes = []
    start = danger_times[0]
    prev = danger_times[0]

    for t in danger_times[1:]:
        if t - prev > GAP_SEC:
            episodes.append((start, prev))
            start = t
        prev = t
    episodes.append((start, prev))
    return episodes


def load_frame_level_data(clip_name: str) -> list[dict]:
    """
    Загружает покадровые данные из лога событий.

    :param clip_name: Имя клипа
    :return: Список словарей с полями frame_idx, timestamp_sec, danger_same_zone
    """
    evt_path = EVENTS_DIR / f"{clip_name}_events_dets.csv"
    if not evt_path.exists():
        return []
    with open(evt_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))



# -------------------------- IoU + MATCHING --------------------------
def iou_time(a: tuple[float, float], b: tuple[float, float]) -> float:
    """
    Temporal IoU двух временных отрезков.

    :param a: Первый отрезок (start, end)
    :param b: Второй отрезок (start, end)
    :return: Значение IoU от 0 до 1
    """
    inter_start = max(a[0], b[0])
    inter_end = min(a[1], b[1])
    intersection = max(0.0, inter_end - inter_start)

    duration_a = a[1] - a[0]
    duration_b = b[1] - b[0]
    union = duration_a + duration_b - intersection

    return intersection / union if union > 0 else 0.0


def match_episodes(gt: list, pred: list) -> tuple[int, int, int]:
    """
    Жадное сопоставление GT и предсказанных эпизодов по temporal IoU.

    Для каждого GT-эпизода ищем лучший (по IoU) ещё не назначенный
    предсказанный эпизод. Если IoU >= IOU_THRESH — считаем TP.

    :param gt: Список GT-эпизодов
    :param pred: Список предсказанных эпизодов
    :return: (TP, FP, FN)
    """
    if not gt and not pred:
        return 0, 0, 0
    if not gt:
        return 0, len(pred), 0
    if not pred:
        return 0, 0, len(gt)

    # Матрица IoU
    iou_matrix = np.zeros((len(gt), len(pred)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            iou_matrix[i, j] = iou_time(g, p)

    matched_gt = set()
    matched_pred = set()

    # Жадный matching: берём пары в порядке убывания IoU
    while True:
        if len(matched_gt) == len(gt) or len(matched_pred) == len(pred):
            break

        # Зануляем уже назначенные
        mask = iou_matrix.copy()
        for i in matched_gt:
            mask[i, :] = 0
        for j in matched_pred:
            mask[:, j] = 0

        best_idx = np.unravel_index(np.argmax(mask), mask.shape)
        best_iou = mask[best_idx]

        if best_iou < IOU_THRESH:
            break

        matched_gt.add(best_idx[0])
        matched_pred.add(best_idx[1])

    tp = len(matched_gt)
    fp = len(pred) - len(matched_pred)
    fn = len(gt) - len(matched_gt)

    return tp, fp, fn


# -------------------------- БИНАРНАЯ КЛАССИФИКАЦИЯ --------------------------
def clip_level_label(episodes: list) -> bool:
    """
    Определяет, есть ли в клипе «опасность» (есть хотя бы 1 эпизод).

    :param episodes: Список эпизодов
    :return: True если клип опасный
    """
    return len(episodes) > 0



# -------------------------- ПОКАДРОВЫЕ МЕТРИКИ --------------------------
def timestamp_in_episodes(ts: float, episodes: list[tuple[float, float]]) -> bool:
    """
    Проверяет, попадает ли временная метка в хотя бы один эпизод.

    :param ts: Временная метка в секундах
    :param episodes: Список эпизодов (start, end)
    :return: True если попадает
    """
    for start, end in episodes:
        if start <= ts <= end:
            return True
    return False


def compute_frame_level(clip_name: str,
                        gt_episodes: list[tuple[float, float]]) -> dict | None:
    """
    Считает покадровые TP, FP, FN, TN для одного клипа.

    Использует все кадры из лога событий. Для каждого кадра:
      - gt_label = попадает ли timestamp в GT-эпизод
      - pred_label = danger_same_zone из лога событий

    :param clip_name: Имя клипа
    :param gt_episodes: GT-эпизоды для данного клипа
    :return: Словарь с TP, FP, FN, TN или None если нет данных
    """
    rows = load_frame_level_data(clip_name)
    if not rows:
        return None

    tp = fp = fn = tn = 0

    for r in rows:
        try:
            ts = float(r["timestamp_sec"])
        except (ValueError, KeyError):
            continue

        gt_danger = timestamp_in_episodes(ts, gt_episodes)
        pred_danger = r.get("danger_same_zone", "").strip() == "True"

        if gt_danger and pred_danger:
            tp += 1
        elif not gt_danger and pred_danger:
            fp += 1
        elif gt_danger and not pred_danger:
            fn += 1
        else:
            tn += 1

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "total": tp + fp + fn + tn}



# -------------------------- ФОРМУЛЫ МЕТРИК --------------------------
def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """
    Precision, Recall, F1-score.

    :return: (precision, recall, f1) — округлены до 3 знаков
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return round(precision, 3), round(recall, 3), round(f1, 3)


def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN).

    :return: accuracy округлённая до 3 знаков
    """
    total = tp + tn + fp + fn
    return round((tp + tn) / total, 3) if total > 0 else 0.0



# -------------------------- МЕЙН ТЕЛО --------------------------
def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(GT_DIR.glob("*_gt.csv"))
    if not gt_files:
        print(f"GT-файлы не найдены в {GT_DIR}")
        return

    # ---------------------------------------------------------------
    #  1) EPISODE-LEVEL
    # ---------------------------------------------------------------
    episode_results = []
    total_tp = total_fp = total_fn = 0

    print(f"\n{'=' * 75}")
    print(f"  EPISODE-LEVEL МЕТРИКИ (temporal IoU >= {IOU_THRESH})")
    print(f"{'=' * 75}")
    print(f"{'Клип':<20} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 75)

    for gt_file in gt_files:
        clip_name = gt_file.stem.replace("_gt", "")
        gt_eps = load_gt(clip_name)
        pred_eps = load_system_episodes(clip_name)

        tp, fp, fn = match_episodes(gt_eps, pred_eps)
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)

        # Пропускаем клипы где gt=0 и pred=0 (нет что считать на episode-level)
        if not gt_eps and not pred_eps:
            episode_results.append({
                "clip_name": clip_name,
                "gt_episodes": 0, "pred_episodes": 0,
                "TP": 0, "FP": 0, "FN": 0,
                "precision": "-", "recall": "-", "f1": "-",
            })
            print(f"{clip_name:<20} {0:>4} {0:>5}    -    -    -       -       -       -")
            continue

        total_tp += tp
        total_fp += fp
        total_fn += fn

        episode_results.append({
            "clip_name": clip_name,
            "gt_episodes": len(gt_eps),
            "pred_episodes": len(pred_eps),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "f1": f1,
        })

        print(f"{clip_name:<20} {len(gt_eps):>4} {len(pred_eps):>5} "
              f"{tp:>4} {fp:>4} {fn:>4} "
              f"{prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")

    # Итого по эпизодам
    print("-" * 75)
    t_prec, t_rec, t_f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    print(f"{'ИТОГО':<20} {'':>4} {'':>5} "
          f"{total_tp:>4} {total_fp:>4} {total_fn:>4} "
          f"{t_prec:>7.3f} {t_rec:>7.3f} {t_f1:>7.3f}")

    # Сохраняем episode-level CSV
    ep_csv = METRICS_DIR / "episode_metrics.csv"
    with open(ep_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["clip_name", "gt_episodes", "pred_episodes",
                  "TP", "FP", "FN", "precision", "recall", "f1"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episode_results)
        writer.writerow({
            "clip_name": "ИТОГО", "gt_episodes": "", "pred_episodes": "",
            "TP": total_tp, "FP": total_fp, "FN": total_fn,
            "precision": t_prec, "recall": t_rec, "f1": t_f1,
        })
    print(f"Сохранено: {ep_csv}")

    # ---------------------------------------------------------------
    #  2) CLIP-LEVEL — бинарная классификация
    # ---------------------------------------------------------------
    clip_tp = clip_tn = clip_fp = clip_fn = 0
    clip_results = []

    print(f"\n{'=' * 75}")
    print(f"  CLIP-LEVEL МЕТРИКИ (бинарная классификация: опасный/безопасный)")
    print(f"{'=' * 75}")
    print(f"{'Клип':<20} {'GT':>10} {'Pred':>10} {'Результат':>12}")
    print("-" * 55)

    for gt_file in gt_files:
        clip_name = gt_file.stem.replace("_gt", "")
        gt_eps = load_gt(clip_name)
        pred_eps = load_system_episodes(clip_name)

        gt_danger = clip_level_label(gt_eps)
        pred_danger = clip_level_label(pred_eps)

        if gt_danger and pred_danger:
            label = "TP"
            clip_tp += 1
        elif not gt_danger and not pred_danger:
            label = "TN"
            clip_tn += 1
        elif not gt_danger and pred_danger:
            label = "FP"
            clip_fp += 1
        else:
            label = "FN"
            clip_fn += 1

        gt_str = "danger" if gt_danger else "safe"
        pred_str = "danger" if pred_danger else "safe"

        clip_results.append({
            "clip_name": clip_name,
            "gt_label": gt_str,
            "pred_label": pred_str,
            "result": label,
        })

        print(f"{clip_name:<20} {gt_str:>10} {pred_str:>10} {label:>12}")

    print("-" * 55)
    c_prec, c_rec, c_f1 = precision_recall_f1(clip_tp, clip_fp, clip_fn)
    c_acc = accuracy(clip_tp, clip_tn, clip_fp, clip_fn)

    print(f"TP={clip_tp}  TN={clip_tn}  FP={clip_fp}  FN={clip_fn}")
    print(f"Accuracy={c_acc:.3f}  Precision={c_prec:.3f}  "
          f"Recall={c_rec:.3f}  F1={c_f1:.3f}")

    # Сохраняем clip-level CSV
    cl_csv = METRICS_DIR / "clip_metrics.csv"
    with open(cl_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["clip_name", "gt_label", "pred_label", "result"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clip_results)
        writer.writerow({
            "clip_name": "ИТОГО",
            "gt_label": f"TP={clip_tp} TN={clip_tn}",
            "pred_label": f"FP={clip_fp} FN={clip_fn}",
            "result": f"Acc={c_acc} F1={c_f1}",
        })
    print(f"Сохранено: {cl_csv}")

    # ---------------------------------------------------------------
    #  3) FRAME-LEVEL — покадровые метрики
    # ---------------------------------------------------------------
    frame_total_tp = frame_total_fp = frame_total_fn = frame_total_tn = 0
    frame_results = []

    print(f"\n{'=' * 75}")
    print(f"  FRAME-LEVEL МЕТРИКИ (покадровая классификация)")
    print(f"{'=' * 75}")
    print(f"{'Клип':<20} {'Frames':>7} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'Acc':>7}")
    print("-" * 85)

    for gt_file in gt_files:
        clip_name = gt_file.stem.replace("_gt", "")
        gt_eps = load_gt(clip_name)

        fl = compute_frame_level(clip_name, gt_eps)
        if fl is None:
            continue

        f_prec, f_rec, f_f1 = precision_recall_f1(fl["TP"], fl["FP"], fl["FN"])
        f_acc = accuracy(fl["TP"], fl["TN"], fl["FP"], fl["FN"])

        frame_total_tp += fl["TP"]
        frame_total_tn += fl["TN"]
        frame_total_fp += fl["FP"]
        frame_total_fn += fl["FN"]

        frame_results.append({
            "clip_name": clip_name,
            "total_frames": fl["total"],
            "TP": fl["TP"], "TN": fl["TN"],
            "FP": fl["FP"], "FN": fl["FN"],
            "precision": f_prec, "recall": f_rec,
            "f1": f_f1, "accuracy": f_acc,
        })

        print(f"{clip_name:<20} {fl['total']:>7} "
              f"{fl['TP']:>6} {fl['TN']:>6} {fl['FP']:>6} {fl['FN']:>6} "
              f"{f_prec:>7.3f} {f_rec:>7.3f} {f_f1:>7.3f} {f_acc:>7.3f}")

    # Итого по кадрам
    print("-" * 85)
    ft_prec, ft_rec, ft_f1 = precision_recall_f1(frame_total_tp, frame_total_fp, frame_total_fn)
    ft_acc = accuracy(frame_total_tp, frame_total_tn, frame_total_fp, frame_total_fn)
    ft_total = frame_total_tp + frame_total_tn + frame_total_fp + frame_total_fn

    print(f"{'ИТОГО':<20} {ft_total:>7} "
          f"{frame_total_tp:>6} {frame_total_tn:>6} "
          f"{frame_total_fp:>6} {frame_total_fn:>6} "
          f"{ft_prec:>7.3f} {ft_rec:>7.3f} {ft_f1:>7.3f} {ft_acc:>7.3f}")

    # Сохраняем frame-level CSV
    fr_csv = METRICS_DIR / "frame_metrics.csv"
    with open(fr_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["clip_name", "total_frames", "TP", "TN", "FP", "FN",
                  "precision", "recall", "f1", "accuracy"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame_results)
        writer.writerow({
            "clip_name": "ИТОГО",
            "total_frames": ft_total,
            "TP": frame_total_tp, "TN": frame_total_tn,
            "FP": frame_total_fp, "FN": frame_total_fn,
            "precision": ft_prec, "recall": ft_rec,
            "f1": ft_f1, "accuracy": ft_acc,
        })
    print(f"Сохранено: {fr_csv}")

    # ---------------------------------------------------------------
    #  ИТОГОВАЯ СВОДКА
    # ---------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print(f"  ИТОГОВАЯ СВОДКА")
    print(f"{'=' * 75}")
    print(f"  Episode-level:  Precision={t_prec:.3f}  Recall={t_rec:.3f}  F1={t_f1:.3f}")
    print(f"  Clip-level:     Accuracy={c_acc:.3f}   Precision={c_prec:.3f}  "
          f"Recall={c_rec:.3f}  F1={c_f1:.3f}")
    print(f"  Frame-level:    Accuracy={ft_acc:.3f}   Precision={ft_prec:.3f}  "
          f"Recall={ft_rec:.3f}  F1={ft_f1:.3f}")
    print(f"{'=' * 75}\n")


if __name__ == "__main__":
    main()



# OLD CODE (DEPRICATED)
# import csv
# from utils.paths import GT_DIR, METRICS_DIR, EVENTS_DIR
#
# # ----------------------- ПАРАМЕТРЫ -----------------------
# IOU_THRESH = 0.5   # минимальный IoU для совпадения эпизода
# GAP_SEC = 1.0   # допуск при склейке соседних True кадров в эпизод
#
#
# # ----------------------- ЗАГРУЗКА ДАННЫХ -----------------------
# def load_gt(clip_name: str) -> list[tuple[float, float]]:
#     """
#     Загружает GT эпизоды для клипа.
#
#     :param clip_name: Имя клипа
#     :return: список тюплов (start_sec, end_sec)
#     """
#     gt_path = GT_DIR / f"{clip_name}_gt.csv"
#     if not gt_path.exists():
#         return []
#     with open(gt_path, "r", encoding="utf-8") as f:
#         rows = list(csv.DictReader(f))
#     # EXM [(13.77, 18.74)]
#     eps = [(float(r["start_sec"]), float(r["end_sec"])) for r in rows]
#     return [(s, e) for s, e in eps]
#
#
# def load_system_episodes(clip_name: str) -> list[tuple[float, float]]:
#     """
#     Загружает лог событий и склеивает соседние danger_same_zone=True
#     в непрерывные эпизоды с допуском GAP_SEC.
#     """
#     evt_path = EVENTS_DIR / f"{clip_name}_events_dets.csv"
#     if not evt_path.exists():
#         return []
#
#     with open(evt_path, "r", encoding="utf-8") as f:
#         rows = list(csv.DictReader(f))
#
#     # Собираем все timestamp где danger=True
#     danger_times = [
#         float(r["timestamp_sec"])
#         for r in rows
#         if r.get("danger_same_zone") == "True"
#     ]
#
#     if not danger_times:
#         return []
#
#     danger_times.sort()
#
#     # Склеиваем в эпизоды
#     episodes = []
#     start = danger_times[0]
#     prev = danger_times[0]
#
#     for t in danger_times[1:]:
#         if t - prev > GAP_SEC:
#             episodes.append((start, prev))
#             start = t
#         prev = t
#     episodes.append((start, prev))
#     # EXM [(13.77, 18.74)]
#     return episodes
#
#
# # ----------------------- СОПОСТАВЛЕНИЕ -----------------------
# def iou_time(a: tuple[float, float], b: tuple[float, float]) -> float:
#     """
#     IoU двух временных отрезков
#
#     :param a: Отрезок с gt
#     :param b: Отрезок с результатами модели
#     :return:
#     """
#     overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
#     union   = max(a[1], b[1]) - min(a[0], b[0])
#     return overlap / union if union > 0 else 0.0
#
#
# def match_episodes(gt: list, pred: list) -> tuple[int, int, int]:
#     """
#     Сопоставляет GT и предсказанные эпизоды.
#     Возвращает (TP, FP, FN).
#     """
#     matched_pred = set()
#     matched_gt = set()
#
#     for i, g in enumerate(gt):
#         for j, p in enumerate(pred):
#             if j in matched_pred:
#                 continue
#             if iou_time(g, p) >= IOU_THRESH:
#                 matched_gt.add(i)
#                 matched_pred.add(j)
#                 break
#
#     tp = len(matched_gt)
#     fp = len(pred) - len(matched_pred)
#     fn = len(gt) - len(matched_gt)
#
#     return tp, fp, fn
#
#
# # ----------------------- МЕТРИКИ -----------------------
# def compute_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
#     """
#     Считает Precision, Recall, F1.
#
#     :param tp: True positive, то есть то, что правильно было обнаружено детектером
#     :param fp: False positive, неправильно обнаруженные детекции
#     :param fn:
#     :return:
#     """
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1        = (2 * precision * recall / (precision + recall)
#                  if (precision + recall) > 0 else 0.0)
#     return round(precision, 3), round(recall, 3), round(f1, 3)
#
#
# # ----------------------- MAIN -----------------------
# def main():
#     gt_files = sorted(GT_DIR.glob("*_gt.csv"))
#     if not gt_files:
#         print(f"GT файлы не найдены в {GT_DIR}")
#         return
#
#     results = []
#     total_tp = total_fp = total_fn = 0
#
#     print(f"\n{'Клип':<20} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
#           f"{'Prec':>7} {'Rec':>7} {'F1':>7}")
#     print("-" * 70)
#
#     for gt_file in gt_files:
#         clip_name = gt_file.stem.replace("_gt", "")
#
#         gt_eps = load_gt(clip_name)
#         pred_eps = load_system_episodes(clip_name)
#
#         if not gt_eps:
#             continue
#
#         print(f"GT: {gt_eps}, Pred: {pred_eps}")
#
#         if not pred_eps:
#             # Система ничего не нашла — все GT эпизоды FN
#             tp, fp, fn = 0, 0, len(gt_eps)
#         else:
#             tp, fp, fn = match_episodes(gt_eps, pred_eps)
#
#         precision, recall, f1 = compute_metrics(tp, fp, fn)
#
#         total_tp += tp
#         total_fp += fp
#         total_fn += fn
#
#         results.append({
#             "clip_name": clip_name,
#             "gt_episodes": len(gt_eps),
#             "pred_episodes": len(pred_eps),
#             "TP": tp, "FP": fp, "FN": fn,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#         })
#
#         print(f"{clip_name:<20} {len(gt_eps):>4} {len(pred_eps):>5} "
#               f"{tp:>4} {fp:>4} {fn:>4} "
#               f"{precision:>7.3f} {recall:>7.3f} {f1:>7.3f}")
#
#     # Итоговые метрики
#     print("-" * 70)
#     total_p, total_r, total_f1 = compute_metrics(total_tp, total_fp, total_fn)
#     print(f"{'ИТОГО':<20} {'':>4} {'':>5} "
#           f"{total_tp:>4} {total_fp:>4} {total_fn:>4} "
#           f"{total_p:>7.3f} {total_r:>7.3f} {total_f1:>7.3f}")
#
#     # Сохраняем в CSV
#     metrics_csv = METRICS_DIR / "metrics.csv"
#     with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
#         fields = ["clip_name", "gt_episodes", "pred_episodes",
#                   "TP", "FP", "FN", "precision", "recall", "f1"]
#         writer = csv.DictWriter(f, fieldnames=fields)
#         writer.writeheader()
#         writer.writerows(results)
#         writer.writerow({
#             "clip_name": "ИТОГО", "gt_episodes": "", "pred_episodes": "",
#             "TP": total_tp, "FP": total_fp, "FN": total_fn,
#             "precision": total_p, "recall": total_r, "f1": total_f1,
#         })
#
#     print(f"\nСохранено: {metrics_csv}")
#
#
# if __name__ == "__main__":
#     main()