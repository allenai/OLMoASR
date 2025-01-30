import os
import glob
from tqdm import tqdm
import multiprocessing
from itertools import chain, repeat
import traceback
import numpy as np
from scipy.io.wavfile import write
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import TextClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip
from IPython.display import Video
from fire import Fire
from typing import List, Optional, Dict
import json
import shutil


def npy_to_wav(npy_file, wav_file, sample_rate=16000):
    # Load the audio data from the npy file
    audio_data = np.load(npy_file)

    # Ensure audio data is in the correct format (int16) for wav files
    audio_data = audio_data

    # Write to a wav file
    write(wav_file, sample_rate, audio_data)


def generate_video(npy_file, transcript_file, output_file, wav_file, sample_rate=16000):
    try:
        # Convert the npy file to a wav file
        npy_to_wav(npy_file, wav_file, sample_rate)

        # Load the converted wav file as the audio
        audio = AudioFileClip(wav_file)

        # Create subtitle generator
        generator = lambda text: TextClip(
            font="/stage/Helvetica.ttf", text=text, font_size=24, color="white", vertical_align="center", horizontal_align="center"
        )

        # If the transcript file is empty, create a silent video
        if open(transcript_file).read() == "":
            print("Silent segment")
            video = ColorClip(size=(1000, 1000), color=(0, 0, 0), duration=audio.duration)
            video.audio = audio
            video.write_videofile(output_file, fps=24)
        else:
            # Create subtitles from the transcript file
            subtitles = SubtitlesClip(transcript_file, make_textclip=generator)
            subtitles.with_position(("center", "bottom"))
            # Create the video with subtitles and audio
            video = CompositeVideoClip(clips=[subtitles], size=(800, 420))
            video.audio = audio
            video.write_videofile(output_file, fps=24, codec="libx264")

        os.remove(wav_file)
        return "http://localhost:8080" + output_file
    except Exception as e:
        return (npy_file, transcript_file, e)


def parallel_generate_video(args):
    status = generate_video(*args)
    return status


def view_video(output_file):
    return Video(output_file, width=800, height=420)


def gen_file_list(seg_dir: str, video_dir: str, wav_dir: str):
    video_seg_list = []
    npy_files = sorted(glob.glob(f"{seg_dir}/*.npy"))
    srt_files = sorted(glob.glob(f"{seg_dir}/*.srt"))

    for i in range(len(npy_files)):
        video_seg_path = os.path.join(video_dir, seg_dir.split("/")[-1], f"{npy_files[i].split('/')[-1].split('.')[0].replace(':', ',')}.mp4")
        os.makedirs(os.path.dirname(video_seg_path), exist_ok=True)
        wav_file = f"{wav_dir}/{npy_files[i].split('.')[0]}.wav"
        os.makedirs(os.path.dirname(wav_file), exist_ok=True)
        video_seg_list.append((npy_files[i], srt_files[i], video_seg_path, wav_file))

    return video_seg_list


def parallel_gen_file_list(args):
    return gen_file_list(*args)


def open_dicts_file(samples_dicts_file) -> List[Dict]:
    with open(samples_dicts_file, "r") as f:
        samples_dicts = list(
            chain.from_iterable(
                json_line.get("sample_dicts")
                for json_line in map(json.loads, f)
                if json_line.get("sample_dicts") is not None
            )
        )
    return samples_dicts

def main(wav_dir: str, video_dir: Optional[str] = None, samples_dicts_file: Optional[str] = None, data_dir: Optional[str] = None):
    os.makedirs(wav_dir, exist_ok=True)
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
    if data_dir is not None:
        with multiprocessing.Pool() as pool:
            video_seg_list = list(
                chain(
                    *tqdm(
                        pool.imap_unordered(
                            parallel_gen_file_list,
                            zip(glob.glob(f"{data_dir}/*"), repeat(video_dir), repeat(wav_dir)),
                        ),
                        total=len(glob.glob(f"{data_dir}/*")),
                    )
                )
            )

        print(video_seg_list[:5])
    else:
        samples_dicts = open_dicts_file(samples_dicts_file)
        rng = np.random.default_rng(42)
        sample_segs = rng.choice(samples_dicts, 1000, replace=False).tolist()
        print(sample_segs[:5])

        video_seg_list = []
        for d in sample_segs:
            original_path = d["audio"]
            output_path = (
                original_path.replace("ow_seg", video_dir.split("/")[-1])
                .replace("npy", "mp4")
                .replace(":", ",")
            )
            
            wav_file = f"{wav_dir}/{d['audio'].split('.')[0]}.wav"
            video_seg_list.append((d["audio"], d["transcript"], output_path, wav_file))
            os.makedirs(os.path.dirname(wav_file), exist_ok=True)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(video_seg_list[:5])
        
    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(parallel_generate_video, video_seg_list),
                total=len(video_seg_list),
            )
        )
    
    shutil.rmtree(wav_dir)
    print([r for r in results if type(r) == tuple])
    results = [r for r in results if type(r) != tuple]
    return results


if __name__ == "__main__":
    output_paths = main(wav_dir="/weka/huongn/ow_samples_wav", video_dir="/weka/huongn/ow_samples_viz", samples_dicts_file="/weka/huongn/samples_dicts/filtered/mixed_no_repeat_min_comma_period_full_1_2_3_4/004/samples_dicts.jsonl")
    data = [{"segment": output_path, "category": "", "video_id": output_path.split("/")[5]} for output_path in output_paths]
    print(data[:5])
    with open("/weka/huongn/ow_viz/data.json", "w") as f:
        json.dump(data, f, indent=2)