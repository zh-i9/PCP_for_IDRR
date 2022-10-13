import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torchmetrics.functional import accuracy, f1
from q_snippets.object import Config
from q_snippets.tensor_ops import label_smoothing

from model import PromptBert
from data_utils import PromptDataModule, LabelMapping
from sklearn.metrics import classification_report


class TaskFrame(pl.LightningModule):
    model_dict = {
        'prompt': PromptBert
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.model = self.model_dict[config.model](config)
        self.tokenizer = PromptDataModule(config).tokenizer
        self.label_mapping = LabelMapping(config.data.mapping_path)
        self.words2label = self.label_mapping.words2label
        self.label2id = self.label_mapping.label2id
        self.val_pred = []
        self.val_gold = []

    def forward(self, batch):
        output = self.model(batch)
        # return output
        pred_token_ids = output.logits.argmax(-1).cpu().numpy().tolist()
        gold_token_ids = batch.labels.cpu().numpy().tolist()
        pred_label_ids = self.labelizer(pred_token_ids)
        gold_label_ids = self.labelizer(gold_token_ids)
        return pred_label_ids, gold_label_ids, output

    def training_step(self, batch, batch_index):
        output = self(batch)
        self.log('train_loss', output.loss, prog_bar=True)  # 调用TensorboardLogger 记录训练数据
        return output.loss

    def labelizer(self, token_ids):
        label_ids = []
        for toekn_id in token_ids:
            word = self.tokenizer.convert_ids_to_tokens(toekn_id)
            label = self.words2label.get(word, "Other")  # roberta
            # label = self.words2label.get("Ġ"+word, "Other")  # bert
            label_id = self.label2id[label]
            label_ids.append(label_id)
        return label_ids

    def validation_step(self, batch, batch_index):
        output = self(batch)
        pred_token_ids = output.logits.argmax(-1).cpu().numpy().tolist()
        gold_token_ids = batch.labels.cpu().numpy().tolist()
        pred_label_ids = self.labelizer(pred_token_ids)
        gold_label_ids = self.labelizer(gold_token_ids)
        self.val_pred.extend(pred_label_ids)
        self.val_gold.extend(gold_label_ids)
        self.log('val_loss', output.loss, prog_bar=True)

    def validation_epoch_end(self, outputs) -> None:
        f1_score = f1(
            preds=torch.tensor(self.val_pred), 
            target=torch.tensor(self.val_gold), 
            average='macro', 
            num_classes=len(self.label2id)
        )
        # - ``'macro'``: Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).
        # - ``'weighted'``: Calculate the metric for each class separately, and average the metrics across classes, weighting each class by its support (``tp + fn``).

        # 调用sklearn库中的方法输出每个类别的f1
        val_result = classification_report(
            y_pred=self.val_pred, y_true=self.val_gold, digits=4, 
            labels = [x for x in range(0, len(self.label2id))],
            target_names = list(self.label2id.keys())
        )
        print(val_result)

        save_path = os.path.join(os.getcwd(), "./lightning_logs/", self.config.version)
        with open(save_path + "/per_epoch_res.txt", "a") as f:
            f.write("epoch {}\n".format(self.current_epoch) + val_result + "\n\n")

        self.log('val_f1', f1_score*100, prog_bar=True)
        self.val_pred = []
        self.val_gold = []

    def get_grouped_params(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        # Group parameters to those that will and will not have weight decay applied
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=1e-4)
        return optimizer
        # optimizer = optim.AdamW(self.get_grouped_params(), lr=self.config.lr)
        # steps_per_epoch = int(len(self.trainer.datamodule.train_dataloader()) / self.config.accumulate_grads)  # 累计梯度
        # warmup_step =  int(steps_per_epoch * 0.25)
        # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=steps_per_epoch*self.config.max_epochs)
        # return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1, 'strict': True, 'monitor': None}]


def train_model(config):
    model = TaskFrame(config)
    # print(model)
    dm = PromptDataModule(config=config)
    logger = TensorBoardLogger(
        save_dir="./lightning_logs/",
        name=None,                # 指定experiment, ./lightning_logs/exp_name/version_name
        version=config.version,   # 指定version, ./lightning_logs/version_name
    )

    # 设置保存模型的路径及参数
    CUR_DIR = os.getcwd()
    dirname = os.path.join(CUR_DIR, "./lightning_logs/", config.version)
    ckpt_callback = ModelCheckpoint(
        dirpath=dirname,
        filename="{epoch}_{val_f1:.2f}",   # 模型保存名称， epoch信息以及验证集分数
        monitor='val_f1',
        mode='max',
        save_top_k=3,
        verbose=True,
    )

    # 设置训练器
    lrm = LearningRateMonitor('step')
    es = EarlyStopping('val_f1', patience=5, mode="max")
    trainer = pl.Trainer(
        # accumulate_grad_batches=config.accumulate_grads,  # 每k个batch累计一次梯度
        logger=logger,
        # resume_from_checkpoint=os.path.join(CUR_DIR, "./lightning_logs/bert_tokens_mean_152rel/epoch=24_val_f1=48.22.ckpt"),
        # num_sanity_val_steps=0,    # 设置训练前是否跑几个step的验证，防止验证过程出问题
        # limit_train_batches=1024,  # 限制训练集数量，方便快速调试
        # limit_val_batches=128,     # 一般直接用全量测试数据吧, 否则验证函数可能会报错
        max_epochs=config.max_epochs,
        callbacks=[ckpt_callback, es, lrm],
        gpus=1,
        deterministic=True,
        # accelerator="dp"
    )
    # 开始训练模型
    dm.setup('fit')
    trainer.fit(model, dm)


class Predictor():
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        yaml_config = os.path.join(os.path.dirname(config.preds.ckpt_path), "hparams.yaml")
        if os.path.exists(yaml_config):
            _config = Config.load(yaml_config)
            _config.preds.update(config.preds)
            _config.data.update(config.data)
            _config.mode = 'predict'
            print(f"config updated:\n{_config}", end="\n"+"="*20+"\n")
        self.config = _config

        self.model = TaskFrame.load_from_checkpoint(config.preds.ckpt_path, config=self.config).to(self.device)

        self.dm = PromptDataModule(config=config)
        self.dm.setup('test')

        self.labelmapping = LabelMapping(self.config.data.mapping_path)

    def save_results(self, preds):
        with open(self.config.preds.result_path, 'w', encoding='utf8') as f:
            for pred, ts_sample in zip(preds, self.dm.testset.samples):
                res_dict = {}
                res_dict["Arg1"] = ts_sample.arg1
                res_dict["Arg2"] = ts_sample.arg2
                res_dict["Connective"] = ts_sample.connective
                res_dict["gold_label"] = ts_sample.sense
                res_dict["pred_label"] = self.labelmapping.id2label[pred]
                f.write(json.dumps(res_dict)+"\n")
        print(f"prediction results saved as {self.config.preds.result_path}")

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            test_pred = []
            test_gold = []

            for batch in tqdm(self.dm.test_dataloader(), total=len(self.dm.test_dataloader())):
                batch = batch.to(self.device)
                pred_token_ids, gold_token_ids, output = self.model(batch)
                test_pred.extend(pred_token_ids)
                test_gold.extend(gold_token_ids)

            # compute f1 sorce in testset
            test_result = classification_report(
                y_pred=test_pred, 
                y_true=test_gold, 
                digits=4,
                labels=[x for x in range(len(self.labelmapping.label2id))],
                target_names=list(self.labelmapping.label2id.keys())
            )
            print(test_result)

        # self.save_results(test_pred)


def run_prediction(config):
    p = Predictor(config)
    p.predict()
