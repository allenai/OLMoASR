# # Load the audio waveform
# audio_arr = audio.load_audio(AUDIO_FILE, sr=16000)
# # Pad or trim the audio array to N_SAMPLES, as expected by the encoder
# audio_arr = audio.pad_or_trim(audio_arr)
# # Convert to mel spectrogram
# # this results in a tensor of shape (80, 3000), but n_audio_ctx = 1500. maybe this is due to the conv1d layer (with stride 2 applied to spectrogram?)
# mel_spec = audio.log_mel_spectrogram(audio_arr, device=DEVICE)

# # not sure if this is the right way to normalize feature
# mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
# mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())

# # Load transcript file
# with open(file=TRANSCRIPT_FILE, mode="r") as f:
#     transcript = f.read().strip()
# # Load the tokenizer
# # question - why when I set multilingual to False, the language and task tokens are set to None?
# # what is the @cached_property decorator?
# # how to have sot_sequence specify no decoding with timestamps
# tokenizer = tokenizer.get_tokenizer(multilingual=True, language="en", task="transcribe")
# # tokenize and encode text
# text_tokens = tokenizer.encode(transcript)
# # add start sequence and end tokens
# # sot/eot token only used when at first/last audio/transcript segment
# text_tokens = list(tokenizer.sot_sequence_including_notimestamps) + text_tokens
# # padding of text tokens
# text_tokens = np.pad(
#     text_tokens,
#     pad_width=(0, n_text_ctx - len(text_tokens)),
#     mode="constant",
#     constant_values=tokenizer.no_speech,
# )
# # convert text tokens to tensor
# text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=DEVICE)

# transcript_dir = "data/transcripts/eh77AUKedyM/segments"
# transcript_files = [os.path.join(transcript_dir, transcript_file) for transcript_file in sorted(os.listdir(transcript_dir))]
# for file_index, transcript_file in enumerate(transcript_files):
#     with open(file=transcript_file, mode="r") as f:
#         transcript = f.read().strip()
#     text_tokens = tokenizer.encode(transcript)

#     if file_index == 0:
#         text_tokens = (
#             list(tokenizer.sot_sequence_including_notimestamps) + text_tokens
#         )

#     text_tokens = np.pad(
#         text_tokens,
#         pad_width=(0, n_text_ctx - len(text_tokens)),
#         mode="constant",
#         constant_values=tokenizer.no_speech,
#     )

#     if file_index == len(transcript_files) - 1:
#         text_tokens[-1] = tokenizer.eot

#     text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=DEVICE)

# def custom_cross_entropy(logits, targets):
#     """
#     Calculate cross-entropy loss manually.

#     Args:
#     - logits: Tensor of logits (model outputs before softmax). Shape: [batch_size, num_classes]
#     - targets: Tensor of target class indices. Shape: [batch_size]

#     Returns:
#     - loss: Calculated cross-entropy loss.
#     """
#     # Apply softmax to convert logits to probabilities
#     probs = F.softmax(logits, dim=1)

#     # Gather the probabilities corresponding to the true classes
#     true_class_probs = probs[range(len(targets)), targets]

#     # Compute the log of these probabilities
#     log_true_class_probs = torch.log(true_class_probs)

#     # Calculate negative average of these logs
#     loss = -torch.mean(log_true_class_probs)

#     return loss

# class AudioDataset(Dataset):
#     def __init__(self, audio_dir):
#         self.audio_files = [
#             os.path.join(audio_dir, audio_file) for audio_file in os.listdir(audio_dir)
#         ]

#     def __len__(self):
#         return len(self.audio_files)

#     def __getitem__(self, index):
#         return self.preprocess_audio(self.audio_files[index])

#     def preprocess_audio(self, audio_file):
#         audio_arr = audio.load_audio(audio_file, sr=16000)
#         audio_arr = audio.pad_or_trim(audio_arr)
#         mel_spec = audio.log_mel_spectrogram(audio_arr, device=DEVICE)
#         mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
#         mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())
#         return mel_spec_scaled


# class TextDataset(Dataset):
#     def __init__(self, transcript_dir, tokenizer, n_text_ctx):
#         self.transcript_files = [
#             os.path.join(transcript_dir, transcript_file)
#             for transcript_file in os.listdir(transcript_dir)
#         ]
#         self.tokenizer = tokenizer
#         self.n_text_ctx = n_text_ctx

#     def __len__(self):
#         return len(self.transcript_files)

#     def __getitem__(self, index):
#         return self.preprocess_text(self.transcript_files[index], index)

#     def preprocess_text(self, transcript_file, file_index):
#         with open(file=transcript_file, mode="r") as f:
#             transcript = f.read().strip()
#         text_tokens = self.tokenizer.encode(transcript)

#         if file_index == 0:
#             text_tokens = (
#                 list(self.tokenizer.sot_sequence_including_notimestamps) + text_tokens
#             )

#         text_tokens = np.pad(
#             text_tokens,
#             pad_width=(0, self.n_text_ctx - len(text_tokens)),
#             mode="constant",
#             constant_values=self.tokenizer.no_speech,
#         )

#         if file_index == len(self.transcript_files) - 1:
#             text_tokens = (
#                 text_tokens[: -len(self.tokenizer.no_speech)] + self.tokenizer.eot
#             )

#         text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=DEVICE)
#         return text_tokens

# # evaluation dataset
# dirs = []
# for root, dir, files in os.walk(EVAL_DIR):
#     if len(root.split("/")) == 6:
#         dirs.append(root)


# class AudioTextDatasetEval(Dataset):
#     def __init__(self, file_dirs, tokenizer, device, n_text_ctx):
#         self.audio_files = []
#         self.transcript_texts = []
#         self.tokenizer = tokenizer
#         self.device = device
#         self.n_text_ctx = n_text_ctx

#         files = [
#             os.path.join(file_dir, audio_file)
#             for file_dir in sorted(file_dirs)
#             for audio_file in sorted(os.listdir(file_dir))
#         ]

#         self.audio_files = sorted([f for f in files if not f.endswith("txt")])

#         transcript_files = sorted([f for f in files if f.endswith("txt")])

#         for transcript_file in transcript_files:
#             with open(file=transcript_file, mode="r") as f:
#                 transcript_text = [" ".join(line.strip().split(" ")[1:]) for line in f]

#             self.transcript_texts.extend(transcript_text)

#     def __len__(self):
#         return len(self.audio_files)

#     def __getitem__(self, index):
#         audio_file, audio_input = self.preprocess_audio(self.audio_files[index])
#         text_tokens = self.preprocess_text(self.transcript_texts[index], index)
#         text_input = text_tokens[:-1]
#         text_y = text_tokens[1:]
#         return audio_file, audio_input, text_input, text_y

#     def preprocess_audio(self, audio_file):
#         audio_arr = audio.load_audio(audio_file, sr=16000)
#         audio_arr = audio.pad_or_trim(audio_arr)
#         mel_spec = audio.log_mel_spectrogram(audio_arr, device=self.device)
#         mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
#         mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())
#         return audio_file, mel_spec_scaled

#     def preprocess_text(self, transcript_file, file_index):
#         with open(file=transcript_file, mode="r") as f:
#             transcript = f.read().strip()
#         text_tokens = self.tokenizer.encode(transcript)

#         if file_index == 0:
#             text_tokens = (
#                 list(self.tokenizer.sot_sequence_including_notimestamps) + text_tokens
#             )

#         if file_index == len(self.transcript_files) - 1:
#             text_tokens.append(tokenizer.eot)

#         text_tokens = np.pad(
#             text_tokens,
#             pad_width=(0, self.n_text_ctx - len(text_tokens)),
#             mode="constant",
#             constant_values=self.tokenizer.no_speech,
#         )

#         text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device)
#         return text_tokens
