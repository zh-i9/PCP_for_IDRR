from cmath import nan
import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import BertConfig, BertModel, BertForMaskedLM
from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM
from q_snippets.tensor_ops import mean_pooling
from dataclasses import dataclass


@dataclass
class Output:
    logits : torch.Tensor
    loss : torch.Tensor = None


class PromptBert(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = RobertaForMaskedLM.from_pretrained(self.config.pretrained)
        # self.model = BertForMaskedLM.from_pretrained(self.config.pretrained)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(self, batch):
        output = self.model(
            input_ids=batch.input_ids, 
            attention_mask=batch.attention_mask,
            # token_type_ids=batch.token_type_ids,
        )
        logits = self.get_mask_embs(output.logits, batch.mask_token_ids)
        
        loss = None
        if batch.labels is not None:
            loss = self.ce_loss(logits, batch.labels)  # loss一定要自己计算 不要用output.loss

        return Output(logits, loss)

    def get_mask_embs(self, last_hidden_state, mask_token_ids):
        mask_embs = []
        for token_embs, mask_token_idx in zip(last_hidden_state, mask_token_ids):
            mask_embs.append(token_embs[mask_token_idx])
        mask_embs_tensor = torch.stack(mask_embs)  # batch_size * hidden_size
        return mask_embs_tensor


if __name__ == '__main__':
    pass