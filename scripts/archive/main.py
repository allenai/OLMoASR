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
