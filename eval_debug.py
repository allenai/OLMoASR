from scripts.eval.eval import EvalDataset
from torch.utils.data import DataLoader
import torch
from open_whisper import load_model
from whisper.normalizers import EnglishTextNormalizer
from tqdm import tqdm
from whisper import audio, DecodingOptions
import jiwer
from fire import Fire

def main(
    ckpt: str,
    eval_dir: str = "data/eval",
):
    eval_sets = ["librispeech_clean", "librispeech_other"]
    eval_loaders = []
    for eval_set in eval_sets:
        eval_dataset = EvalDataset(eval_set=eval_set, eval_dir=eval_dir)

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=23,
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
        )
        eval_loaders.append((eval_set, eval_dataloader))

    device = torch.device("cuda")

    model = load_model(name=ckpt, device=device, inference=True, in_memory=True)
    model.eval() 

    normalizer = EnglishTextNormalizer()

    for eval_set, eval_dataloader in eval_loaders:
        print(f"Evaluating {eval_set}\n")

        hypotheses = []
        references = []  
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(eval_dataloader), total=len(eval_dataloader)
            ):
                audio_fp, _, audio_input, text_y = batch
                audio_input = audio_input.to(device)

                options = DecodingOptions(language="en", without_timestamps=True)

                results = model.decode(audio_input, options=options)
                pred_text = [result.text for result in results]
                norm_pred_text = [normalizer(text) for text in pred_text]
                hypotheses.extend(norm_pred_text)
                norm_tgt_text = [normalizer(text) for text in text_y]
                references.extend(norm_tgt_text)

            avg_wer = jiwer.wer(references, hypotheses) * 100

            print(f"{eval_set} WER: {avg_wer:.2f}\n")


if __name__ == "__main__":
    Fire(main)
