import os
import glob
from tqdm import tqdm
import multiprocessing
from itertools import chain, repeat
import traceback
import numpy as np
from scipy.io.wavfile import write
from moviepy.editor import AudioFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.video.tools.subtitles import SubtitlesClip
from IPython.display import Video
from fire import Fire


def npy_to_wav(npy_file, wav_file, sample_rate=16000):
    # Load the audio data from the npy file
    audio_data = np.load(npy_file)

    # Ensure audio data is in the correct format (int16) for wav files
    audio_data = audio_data

    # Write to a wav file
    write(wav_file, sample_rate, audio_data)


def generate_video(npy_file, transcript_file, output_file, sample_rate=16000):
    try:
        # Convert the npy file to a wav file
        wav_file = f"{npy_file.split('.')[0]}.wav"
        npy_to_wav(npy_file, wav_file, sample_rate)

        # Load the converted wav file as the audio
        audio = AudioFileClip(wav_file)

        # Create subtitle generator
        generator = lambda text: TextClip(
            text, font="Cantarell-Regular", fontsize=24, color="white"
        )

        # If the transcript file is empty, create a silent video
        if open(transcript_file).read() == "":
            print("Silent segment")
            video = ColorClip(size=(1000, 1000), color=(0, 0, 0), duration=audio.duration)
            video = video.set_audio(audio)
            video.write_videofile(output_file, fps=24)
        else:
            # Create subtitles from the transcript file
            subtitles = SubtitlesClip(transcript_file, generator).set_position(
                ("center", "bottom")
            )
            # Create the video with subtitles and audio
            video = CompositeVideoClip([subtitles], size=(800, 420))
            video = video.set_audio(audio)
            video.write_videofile(output_file, fps=24)

        return 1

    except Exception as e:
        print(traceback.format_exc())
        with open(
            "logs/data/filtering/error_gen_video.txt",
            "a",
        ) as f:
            f.write(f"{npy_file} {transcript_file}\t{e}\n")
        return 0


def parallel_generate_video(args):
    status = generate_video(*args)
    return status


def view_video(output_file):
    return Video(output_file, width=800, height=420)


def gen_file_list(seg_dir: str, video_dir: str):
    video_seg_list = []
    npy_files = sorted(glob.glob(f"{seg_dir}/*.npy"))
    srt_files = sorted(glob.glob(f"{seg_dir}/*.srt"))

    for i in range(len(npy_files)):
        video_seg_path = os.path.join(video_dir, seg_dir.split("/")[-1], f"{npy_files[i].split('/')[-1].split('.')[0].replace(':', ',')}.mp4")
        os.makedirs(os.path.dirname(video_seg_path), exist_ok=True)
        video_seg_list.append((npy_files[i], srt_files[i], video_seg_path))

    return video_seg_list


def parallel_gen_file_list(args):
    return gen_file_list(*args)


def main(data_dir: str, video_dir: str):
    with multiprocessing.Pool() as pool:
        video_seg_list = list(
            chain(
                *tqdm(
                    pool.imap_unordered(
                        parallel_gen_file_list,
                        zip(glob.glob(f"{data_dir}/*"), repeat(video_dir)),
                    ),
                    total=len(glob.glob(f"{data_dir}/*")),
                )
            )
        )

    print(video_seg_list[:5])

    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(parallel_generate_video, video_seg_list),
                total=len(video_seg_list),
            )
        )

        return results


if __name__ == "__main__":
    main(data_dir="data/00000_seg", video_dir="data/filtering_viz")
