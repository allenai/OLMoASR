import os

def get_baseline_data():
    audio_files_train = []
    for root, *_ in os.walk("data/audio"):
        if "segments" in root:
            for f in os.listdir(root):
                audio_files_train.append(os.path.join(root, f))

    transcript_files_train = []
    for root, *_ in os.walk("data/transcripts"):
        if "segments" in root:
            for f in os.listdir(root):
                transcript_files_train.append(os.path.join(root, f))

    return audio_files_train, transcript_files_train

def get_manual_data():
    audio_files_train = []
    with open("logs/data/filtering/manual_audio.txt", "r") as f:
        for line in f:
            audio_files_train.append(line.strip())

    transcript_files_train = []
    with open("logs/data/filtering/manual_text.txt", "r") as f:
        for line in f:
            transcript_files_train.append(line.strip())
    
    return audio_files_train, transcript_files_train