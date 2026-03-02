# -------------------- БИБЛИОТЕКИ --------------------
import json
import cv2
import numpy as np
from utils.paths import ROIS_DIR, CLIPS_DIR, clip_check, roi_from_clip


# -------------------- ФУНКЦИИ РИСОВАНИЯ И СОХРАНЕНИЯ REGION OF INTEREST (ROI) --------------------
def draw_polygon(img, pts, color, closed=False, thickness=2):
    """
    Функция рисование (Если точек 2 и больше рисует линию)

    :param img: Изображение для рисования
    :param pts: Набор точек для создания региона
    :param color: Цвет рисования
    :param closed: Закрыт ли регион? (Если точки 3 и больше, последняя точка соединяется с первой)
    :param thickness: Толщина линии
    """
    # Если точек 2 или больше, то рисует линию
    if len(pts) >= 2:
        cv2.polylines(img, [pts], isClosed=closed, color=color, thickness=thickness)
    for (x, y) in pts:
        cv2.circle(img, (x, y), 5, color, -1)


def save_rois(out_path, clip_id, frame_w, frame_h, crosswalks, risks):
    """
        Функция сохранения ROI в формате JSON

        :param out_path: Путь сохранения
        :param clip_id: ID клипа
        :param frame_w: Ширина скрина из клипа
        :param frame_h: Высота скрина из клипа
        :param crosswalks: Регионы пешеходных переходов
        :param risks: Регионы дорог (опасная зона для пешехода)
        """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'clip_id': clip_id,
        'frame_size': [frame_w, frame_h],
        "polygons": {
            "crosswalks": crosswalks,
            "risks": risks
        }
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {out_path}")



def redraw_frame(frame, help_text, mode, crosswalk_polys, risk_polys, cur_pts):
    """
        Функция перерисовки кадра

        :param frame: Кадр для рисования
        :param help_text: Текст помощи
        :param mode: Текущий режим рисования
        :param crosswalk_polys: Регионы пешеходного перехода
        :param risk_polys: Регионы опасных зон (дорога)
        :param cur_pts:
        """
    img = frame.copy()

    # -------------------------------------- ТЕКСТ НА ЭКРАНЕ --------------------------------------
    # Черная панель, чтобы текст был читаемый
    cv2.rectangle(img, (5, 5), (750, 305), (0, 0, 0), -1)

    # Отображение текущего режима
    cv2.putText(img, f"Mode: {mode}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    # Отображение счетчиков областей
    cv2.putText(img, f"crosswalks: {len(crosswalk_polys)}", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    cv2.putText(img, f"risks: {len(risk_polys)}", (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

    cv2.putText(img, "---------------------------", (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    y = 155
    for t in help_text:
        cv2.putText(img, t, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y += 28

    # -------------------------------------- ОТРИСОВКА УЖЕ ЗАВЕРШЕННЫХ ПОЛИГОНОВ --------------------------------------
    for poly in crosswalk_polys:
        pts = np.array(poly, dtype=np.int32)
        draw_polygon(img, pts, (0, 255, 0), closed=True, thickness=2)

    for poly in risk_polys:
        pts = np.array(poly, dtype=np.int32)
        draw_polygon(img, pts, (0, 0, 255), closed=True, thickness=2)

    # -------------------------------------- ТЕКУЩИЙ НЕЗАВЕРШЕННЫЙ ПОЛИГОН --------------------------------------
    if len(cur_pts) > 0:
        pts = np.array(cur_pts, dtype=np.int32)
        if mode == "crosswalk":
            draw_polygon(img, pts, (0, 255, 0), closed=False, thickness=3)
        else:
            draw_polygon(img, pts, (0, 0, 255), closed=False, thickness=3)

    cv2.imshow("ROI Tool", img)



def drawing(roi_dir, clips_dir, clip_name):
    """
    Мейн функция рисования

    :param roi_dir: Директория с областями интересов
    :param clips_dir: Директория с клипами
    :param clip_name: На котором хотим рисовать области интересов
    """
    # ---------------------------------- ПРОВЕРКА ПУТЕЙ ----------------------------------
    # Получаем id клипа и директорию для сохранения json данных
    clip_path = clip_check(clips_dir=clips_dir, clip_name=clip_name)
    clip_title = clip_path.stem

    # ------------------------ ЛОКАЛЬНЫЕ ПЕРЕМЕННЫЕ И ФУНКЦИИ ------------------------
    # Точки координат интересов
    crosswalk_polys = []  # список полигонов crosswalk
    risk_polys = []  # список полигонов risk
    cur_pts = []  # точки текущего рисуемого полигона

    mode = "crosswalk" # Текущий режим
    roi_id = roi_from_clip(clip_title)
    out_path = roi_dir / f"{roi_id}.json" # Директория сохранения

    # Помощь с использованием
    help_text = [
        "LMB: add point",
        "BACKSPACE: undo last point",
        "N: finish current polygon",
        "TAB: switch polygon type (crosswalk/risk)",
        "S: save JSON",
        "ESC/Q: exit without saving"
    ]

    # Замыкание обработчик событий мыши
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cur_pts.append([int(x), int(y)])
            redraw_frame(frame, help_text, mode, crosswalk_polys, risk_polys, cur_pts)


    # Замыкание закрытия полигона (Закрывается линия последней и первой точки)
    def finalize_current_polygon():
        if len(cur_pts) < 3:
            print("Нужно минимум 3 точки, чтобы завершить полигон.")
            return False

        if mode == "crosswalk":
            crosswalk_polys.append(cur_pts[:])
            print("Добавлен crosswalk #{}".format(len(crosswalk_polys)))
        else:
            risk_polys.append(cur_pts[:])
            print("Добавлен risk #{}".format(len(risk_polys)))

        cur_pts.clear()
        return True

    # ------------------------ РАБОТА С ВИДЕО ------------------------

    # Захватываем видео и читаем первый кадр
    cap = cv2.VideoCapture(str(clip_path))
    ok, frame = cap.read()
    cap.release()

    # Если первый кадр не прочитан, выходим из цикла
    if not ok:
        print("Ошибка чтения кадра")
        return

    # Устанавливаем окно и замыкания для работы с мышью
    cv2.namedWindow("ROI Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Tool", on_mouse)

    # Высота и ширина кадра
    h, w = frame.shape[:2]

    redraw_frame(frame, help_text, mode, crosswalk_polys, risk_polys, cur_pts)

    while True:
        key = cv2.waitKey(0) & 0xFF

    # ------------------------ КЛАВИШИ УПРАВЛЕНИЯ ------------------------
        if key in [27, ord('q'), ord('Q')]:
            print("Выход без сохранения.")
            break

        # N - завершить текущий полигон
        if key in [ord('n'), ord('N')]:
            finalize_current_polygon()
            redraw_frame(frame, help_text, mode, crosswalk_polys, risk_polys, cur_pts)
            continue

        # TAB - смена режима (только если последний регион завершен)
        if key == 9:
            if len(cur_pts) > 0:
                print("Сначала завершите текущий полигон (N) или удалите точки (BACKSPACE).")
            else:
                mode = "risk" if mode == "crosswalk" else "crosswalk"
            redraw_frame(frame, help_text, mode, crosswalk_polys, risk_polys, cur_pts)
            continue

        # BACKSPACE - удалить последнюю точку ТЕКУЩЕГО полигона
        if key in [8, 127]:
            if cur_pts:
                cur_pts.pop()
            redraw_frame(frame, help_text, mode, crosswalk_polys, risk_polys, cur_pts)
            continue

        # S - сохранить
        if key in [ord('s'), ord('S')]:
            if len(cur_pts) > 0:
                print("Есть незавершенный полигон. Нажмите N чтобы завершить, либо удалите точки.")
                continue

            if len(crosswalk_polys) < 1 or len(risk_polys) < 1:
                print("Нужно минимум 1 полигон для crosswalk и 1 для risk.")
                continue

            save_rois(out_path, clip_title, w, h, crosswalk_polys, risk_polys)
            break
    cv2.destroyAllWindows()



if __name__ == "__main__":
    clip_name = input(f"Название клипа, например vid_001_0000: ").strip()
    drawing(roi_dir=ROIS_DIR, clips_dir=CLIPS_DIR, clip_name=clip_name)