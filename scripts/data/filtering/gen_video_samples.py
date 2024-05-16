import os
from moviepy.editor import AudioFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.video.tools.subtitles import SubtitlesClip
from IPython.display import Video
from open_whisper import preprocess
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from moviepy.config import change_settings
import traceback

change_settings({"IMAGEMAGICK_BINARY": "/mmfs1/home/hvn2002/software/bin/magick"})


def generate_video(audio_file, transcript_file, output_file):
    try:
        audio = AudioFileClip(audio_file)
        generator = lambda text: TextClip(
            text, font="Cantarell-Regular", fontsize=24, color="white"
        )

        if open(transcript_file).read() == "":
            print("Silent segment")
            video = ColorClip(size=(800, 420), color=(0, 0, 0), duration=audio.duration)
            video = video.set_audio(audio)
            video.write_videofile(output_file, fps=24)
        else:
            subtitles = SubtitlesClip(transcript_file, generator).set_position(
                ("center", "bottom")
            )
            video = CompositeVideoClip([subtitles], size=(800, 420))
            video = video.set_audio(audio)
            video.write_videofile(output_file, fps=24)
        return 1
    except Exception as e:
        print(traceback.format_exc())
        with open(
            "/mmfs1/gscratch/efml/hvn2002/open_whisper/logs/data/filtering/error_gen_video.txt",
            "a",
        ) as f:
            f.write(f"{audio_file} {transcript_file}\t{e}\n")
        return 0


def parallel_generate_video(args):
    status = generate_video(*args)
    return status


def view_video(output_file):
    return Video(output_file, width=800, height=420)


def main():
    transcript_dir = "data/transcripts"
    transcript_files = []
    for root, dirs, files in os.walk(transcript_dir):
        if "segments" in root:
            for f in os.listdir(root):
                transcript_files.append(os.path.join(root, f))

    rng = np.random.default_rng(42)
    sample_t_segs = list(rng.choice(transcript_files, 1000, replace=False))
    sample_a_segs = [
        path.replace("transcripts", "audio").replace("srt", "m4a")
        for path in sample_t_segs
    ]

    atv_list = []
    for i in range(len(sample_t_segs)):
        original_path = sample_t_segs[i]
        output_path = (
            original_path.replace("transcripts", "filtering")
            .replace("/segments", "")
            .replace("srt", "mp4")
            .replace(":", "_")
            .replace(".", "_", 2)
        )

        os.makedirs(f"data/filtering/{original_path.split('/')[2]}", exist_ok=True)
        atv_list.append((sample_a_segs[i], original_path, output_path))

    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(parallel_generate_video, atv_list, chunksize=5),
                total=len(atv_list),
            )
        )


if __name__ == "__main__":
    main()
