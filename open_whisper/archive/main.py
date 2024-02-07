# def chunk_transcript(transcript_file: str, output_dir: str) -> None:
#     output_dir = output_dir + "/segments"
#     os.makedirs(output_dir, exist_ok=True)

#     transcript, *_ = read_vtt(transcript_file)
#     a = 0
#     b = 0
#     timestamps = list(transcript.keys())
#     start = timestamps[a][0]
#     # end = timestamps[b][1]
#     # text = transcript[(start, end)]
#     init_diff = 0
#     text = ""
#     added_silence = False
#     remain_text = ""

#     while a < len(transcript) + 1:
#         if init_diff < 30000:
#             if init_diff == 0 and convert_to_milliseconds(
#                 adjust_timestamp(start, 30)
#             ) < convert_to_milliseconds(timestamps[a][0]):
#                 init_diff = 30000
#             elif a == len(transcript) - 1:
#                 text += transcript[(timestamps[a][0], timestamps[a][1])]
#                 init_diff = 30000
#                 a += 1
#             else:
#                 init_diff = calculate_difference(start, timestamps[b][1])
#                 text += transcript[(timestamps[a][0], timestamps[a][1])]
#                 b += 1
#                 if convert_to_milliseconds(timestamps[b][0]) > convert_to_milliseconds(
#                     timestamps[a][1]
#                 ):
#                     init_diff += calculate_difference(
#                         timestamps[b][0],
#                         timestamps[a][1],
#                     )
#                     added_silence = True
#                 a += 1
#         else:
#             new_start = adjust_timestamp(start, 30)
#             output_file = f"{output_dir}/{start}_{new_start}.txt"
#             remain_text = ""
#             if init_diff >= 31000:
#                 if not added_silence:
#                     if (init_diff - 30000) // 1000 > len(
#                         transcript[(timestamps[a - 1][0], timestamps[a - 1][1])].split(
#                             " "
#                         )
#                     ):
#                         keep_len = int(
#                             np.ceil(
#                                 ((init_diff - 30000) // 1000)
#                                 / len(
#                                     transcript[
#                                         (timestamps[a - 1][0], timestamps[a - 1][1])
#                                     ].split(" ")
#                                 )
#                                 * 1.0
#                             )
#                         )
#                         tokens = text.split(" ")
#                         tokens = tokens[
#                             : -(
#                                 len(
#                                     transcript[
#                                         (timestamps[a - 1][0], timestamps[a - 1][1])
#                                     ].split(" ")
#                                 )
#                                 - keep_len
#                             )
#                         ]
#                         text = " ".join(tokens)
#                         remain_text = transcript[
#                             (timestamps[a - 1][0], timestamps[a - 1][1])
#                         ][keep_len:]
#                     else:  # < or ==
#                         tokens = text.split(" ")
#                         text = " ".join(
#                             tokens[
#                                 : -len(
#                                     transcript[
#                                         (timestamps[a - 1][0], timestamps[a - 1][1])
#                                     ].split(" ")
#                                 )
#                                 + 1
#                             ]
#                         )
#                         remain_text = " ".join(
#                             transcript[
#                                 (timestamps[a - 1][0], timestamps[a - 1][1])
#                             ].split(" ")[1:]
#                         )

#                     a -= 1
#                     b -= 1

#             transcript_file = open(output_file, "w")
#             transcript_file.write(text)
#             transcript_file.close()

#             if a == len(transcript):
#                 break

#             init_diff = 0
#             text = remain_text
#             start = new_start

# def chunk_audio(audio_file: str, output_dir: str, transcript_start: str) -> None:
#     output_dir = output_dir + "/segments"
#     os.makedirs(output_dir, exist_ok=True)

#     command = [
#         "ffmpeg",
#         "-i",
#         audio_file,
#         "-f",
#         "segment",
#         "-segment_time",
#         "30",
#         "-c",
#         "copy",
#         f"{output_dir}/.{audio_file.split('.')[-1]}",
#     ]

#     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     for file in sorted(os.listdir(output_dir), key=lambda x: int(x.split(".")[0])):
#         new_time = adjust_timestamp(transcript_start, 30)
#         os.rename(
#             os.path.join(output_dir, file),
#             os.path.join(
#                 output_dir, f"{transcript_start}_{new_time}.{file.split('.')[-1]}"
#             ),
#         )
#         transcript_start = new_time


# def chunk_audio_transcript_text(transcript_file: str, audio_file: str):
#     # if transcript or audio files doesn't exist
#     if not os.path.exists(transcript_file):
#         with open(f"logs/failed_download_t.txt", "a") as f:
#             f.write(f"{transcript_file}\n")
#         if not os.path.exists(audio_file):
#             with open(f"logs/failed_download_a.txt", "a") as f:
#                 f.write(f"{audio_file}\n")

#         return None

#     t_output_dir = "/".join(transcript_file.split("/")[:3]) + "/segments"
#     a_output_dir = "/".join(audio_file.split("/")[:3]) + "/segments"
#     os.makedirs(t_output_dir, exist_ok=True)
#     os.makedirs(a_output_dir, exist_ok=True)

#     cleaned_transcript = clean_transcript(transcript_file)
#     if cleaned_transcript is None:
#         with open(f"logs/empty_transcript.txt", "a") as f:
#             f.write(f"{transcript_file}\n")
#         return None

#     transcript, *_ = read_vtt(transcript_file)

#     # if transcript file is empty
#     if transcript == {}:
#         with open(f"logs/empty_transcript.txt", "a") as f:
#             f.write(f"{transcript_file}\n")
#         return None

#     a = 0
#     b = 0

#     timestamps = list(transcript.keys())
#     diff = 0
#     init_diff = 0
#     text = ""

#     while a < len(transcript) + 1:
#         init_diff = calculate_difference(timestamps[a][0], timestamps[b][1])
#         if init_diff < 30000:
#             diff = init_diff
#             if text != "":
#                 if text[-1] != " ":
#                     text += " "

#             text += transcript[(timestamps[b][0], timestamps[b][1])]
#             b += 1
#         else:
#             t_output_file = (
#                 f"{t_output_dir}/{timestamps[a][0]}_{timestamps[b - 1][1]}.txt"
#             )
#             transcript_file = open(t_output_file, "w")
#             transcript_file.write(text)
#             transcript_file.close()

#             trim_audio(
#                 audio_file,
#                 timestamps[a][0],
#                 timestamps[b - 1][1],
#                 0,
#                 0,
#                 a_output_dir,
#             )
#             text = ""
#             init_diff = 0
#             diff = 0
#             a = b

#         if b == len(transcript) and diff < 30000:
#             t_output_file = (
#                 f"{t_output_dir}/{timestamps[a][0]}_{timestamps[b - 1][1]}.txt"
#             )
#             transcript_file = open(t_output_file, "w")
#             transcript_file.write(text)
#             transcript_file.close()

#             trim_audio(
#                 audio_file, timestamps[a][0], timestamps[b - 1][1], 0, 0, a_output_dir
#             )

#             break


# def parallel_chunk_audio_transcript_text(args) -> None:
#     chunk_audio_transcript_text(*args)
