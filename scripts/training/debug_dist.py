import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from open_whisper import utils
from torch.nn.utils import clip_grad_norm_

DEVICE = torch.device("cuda")

# model setup
@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


model_dims = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_vocab=51864,
    n_text_ctx=448,
    n_text_state=384,
    n_text_head=6,
    n_text_layer=4,
)


class WhisperModule(pl.LightningModule):
    def __init__(self, model_dims, optimizer, learning_rate, betas, eps, weight_decay, warmup_steps, lr_lambda, model):
        super().__init__()

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.lr_lambda = lr_lambda

        self.save_hyperparameters("optimizer", "learning_rate", "betas", "eps", "weight_decay", "warmup_steps")
        self.model = model

    def forward(self, audio_input, text_input, padding_mask):
        return self.model(audio_input, text_input, padding_mask)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda=self.lr_lambda),
            'interval': 'step',
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        audio_files, audio_input, text_input, text_y, padding_mask = batch
        logits = self.forward(audio_input, text_input, padding_mask)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)

        pred_text = []
        for pred_instance in pred.cpu().numpy():
            pred_instance_text = tokenizer.decode(list(pred_instance))
            pred_instance_text = pred_instance_text.rsplit("<|endoftext|>", 1)[0]
            pred_text.append(pred_instance_text)

        tgt_text = []
        for text_y_instance in text_y.cpu().numpy():
            tgt_y_instance_text = tokenizer.decode(list(text_y_instance))
            tgt_y_instance_text = tgt_y_instance_text.split("<|endoftext|>")[0]
            tgt_text.append(tgt_y_instance_text)

        tgt_pred_pairs = utils.clean_text(list(zip(tgt_text, pred_text)), "english")

        average_wer = utils.average_wer(tgt_pred_pairs)

        with open(f"logs/training/training_results.txt", "a") as f:
            for tgt_text_instance, pred_text_instance in tgt_pred_pairs[:10:2]:
                f.write(f"{pred_text_instance=}\n")
                f.write(f"{len(pred_text_instance)=}\n")
                f.write(f"{tgt_text_instance=}\n")
                f.write(f"{len(tgt_text_instance)=}\n")
            f.write(f"{average_wer=}\n\n")


        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), text_y.view(-1), ignore_index=0)
        
        clip_grad_norm_(model.parameters(), max_grad_norm)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Similar to training_step, but with validation data
        pass

    def test_step(self, batch, batch_idx):
        # Similar to validation_step, but possibly with additional metrics
        pass
