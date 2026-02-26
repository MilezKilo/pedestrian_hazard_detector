from moviepy import VideoFileClip
from utils.paths import RAW_VID_DIR, CLIPS_DIR


# -------------------- ФУНКЦИИ --------------------
def cut_video(video_path, clips_path, clip_length: int = 20):
    """
    Нарезает видео на клипы по CLIP_LENGTH сек

    :param video_path: Путь к видео
    :param clips_path: Путь к папке, куда будут сохранятся клипы
    :param clip_length: Длина клипа (дефолтно 20 секунд)
    """

    clips_path.mkdir(parents=True, exist_ok=True)
    video_name = video_path.stem
    print(f'Cutting {video_path.name}')

    clip = VideoFileClip(str(video_path))
    try:
        duration = int(clip.duration)

        part = 0

        for start in range(0, duration, clip_length):
            end = min(start + clip_length, duration)

            out_path = clips_path/f'{video_name}_{part:04d}.mp4'
            print(f'-> Clip {part}: {start}-{end} sec.')

            subclip = clip.subclipped(start, end)
            subclip.write_videofile(str(out_path), codec="libx264", audio=False, logger=None)
            subclip.close()

            part += 1
    finally:
        clip.close()


# -------------------- ЗАПУСК --------------------
if __name__ == "__main__":
    vid_name = input('Введите название видео (пример: vid_001 / vid_002): ').strip()
    video_path = RAW_VID_DIR / f'{vid_name}.mp4'

    if not video_path.is_file():
        print('Файл не найден')
    else:
        cut_video(video_path=video_path, clips_path=CLIPS_DIR)