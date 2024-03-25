# class AudioTextEval(AudioTextDataset):
#     def __init__(
#         self,
#         audio_files: List,
#         transcript_files: List,
#         tokenizer,
#         device: torch.DeviceObjType,
#         n_text_ctx: int,
#     ):
#         super().__init__(audio_files, transcript_files, tokenizer, device, n_text_ctx)

#         self.transcript_texts = []
#         for file in transcript_files:
#             with open(file, "r") as f:
#                 transcript_text = [
#                     (line.split(" ")[0], " ".join(line.split(" ")[1:]).strip())
#                     for line in f
#                 ]
#             self.transcript_texts.extend(transcript_text)

#     def __getitem__(self, index):
#         audio_file, audio_input = self.preprocess_audio(self.audio_files[index])
#         text_tokens = self.preprocess_text(*self.transcript_texts[index])

#         return audio_file, audio_input, text_tokens[1:]

#     def preprocess_text(self, text_id, transcript_text):
#         text_tokens = self.tokenizer.encode(transcript_text)

#         text_tokens = (
#             list(self.tokenizer.sot_sequence_including_notimestamps) + text_tokens
#         )

#         text_tokens.append(tokenizer.eot)

#         text_tokens = np.pad(
#             text_tokens,
#             pad_width=(0, self.n_text_ctx - len(text_tokens)),
#             mode="constant",
#             constant_values=51864,
#         )

#         text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device)
#         return text_id, text_tokens


# data_dirs_val = []
# for root, dirs, files in os.walk("data/eval/LibriSpeech/test-clean"):
#     if len(root.split("/")) == 6:
#         data_dirs_val.append(root)

# transcript_files = []
# audio_files = []

# for d in data_dirs_val:
#     for f in os.listdir(d):
#         if f.endswith("txt"):
#             transcript_files.append(os.path.join(d, f))
#         else:
#             audio_files.append(os.path.join(d, f))

# audio_text_dataset_val = AudioTextEval(
#     audio_files=sorted(audio_files),
#     transcript_files=sorted(transcript_files),
#     tokenizer=tokenizer,
#     device=DEVICE,
#     n_text_ctx=448,
# )

# val_batch_size = 8
# audio_text_val_dataloader = DataLoader(
#     audio_text_dataset_val, batch_size=val_batch_size
# )
