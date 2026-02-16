from moviepy import VideoFileClip
from utils.paths import RAW_VID_DIR, CLIPS_DIR

# -------------------- КОНСТАНТЫ --------------------
CLIP_LENGTH = 20

# Нарезает видео на клипы по CLIP_LENGTH сек
def cut_video(video_path):
    """
    :param video_path: Путь к видео
    :return: None
    """
    video_name = video_path.stem
    print(f'Cutting {video_path.name}')

    clip = VideoFileClip(str(video_path))
    duration = int(clip.duration)

    part = 0

    for start in range(0, duration, CLIP_LENGTH):
        end = min(start + CLIP_LENGTH, duration)

        out_path = CLIPS_DIR/f'{video_name}_{part:04d}.mp4'
        print(f'-> Clip {part}: {start}-{end} sec.')

        subclip = clip.subclipped(start, end)
        subclip.write_videofile(str(out_path), codec="libx264", audio=False, logger=None)

        part += 1

    clip.close()


# -------------------- ЗАПУСК --------------------
if __name__ == "__main__":
    vid_name = input('Введите название видео (пример: vid_001 / vid_002): ')
    if not (RAW_VID_DIR / f'{vid_name}.mp4').exists():
        print('Файл не найден')
    else:
        cut_video(RAW_VID_DIR / f'{vid_name}.mp4')