TRAIN_TABLE_COLS = [
    "audio_file",
    "audio_input",
    "transcript_file",
    "pred_text",
    "unnorm_pred_text (pre padding removal)",
    "unnorm_pred_text",
    "tgt_text",
    "unnorm_tgt_text",
    "substitutions",
    "deletions",
    "insertions",
    "tgt_text_len",
    "wer",
]

VAL_TABLE_COLS = [
    "audio_file",
    "audio_input",
    "transcript_file",
    "pred_text",
    "unnorm_pred_text (pre padding removal)",
    "unnorm_pred_text",
    "tgt_text",
    "unnorm_tgt_text",
    "substitutions",
    "deletions",
    "insertions",
    "tgt_text_len",
    "wer",
]

EVAL_TABLE_COLS = [
    "corpi",
    "audio_file",
    "audio_input",
    "pred_text",
    "unnorm_pred_text",
    "tgt_text",
    "wer",
]
