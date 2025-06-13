# Load model directly
from transformers import AutoProcessor, AutoModelForCTC

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

def map_to_pred(batch):
    input_values = processor(batch["audio"]["array"], return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch

result = librispeech_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["audio"])

print("WER:", wer(result["text"], result["transcription"]))