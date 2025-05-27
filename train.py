from config import Config
from dataset import get_dataloaders
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')
from pytorch_lightning.callbacks import ModelCheckpoint

class FFN(nn.Module):
    def __init__(self, in_feat):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(in_feat, in_feat)
        self.linear2 = nn.Linear(in_feat, in_feat)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out


class EncoderEmbedding(nn.Module):
    def __init__(self, n_exercises, n_categories, n_dims, seq_len):
        super(EncoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.exercise_embed = nn.Embedding(n_exercises, n_dims)
        self.category_embed = nn.Embedding(n_categories, n_dims)
        self.position_embed = nn.Embedding(seq_len, n_dims)

    def forward(self, exercises, categories):
        e = self.exercise_embed(exercises)
        c = self.category_embed(categories)
        seq = torch.arange(self.seq_len, device=Config.device).unsqueeze(0)
        p = self.position_embed(seq)
        return p + c + e


class DecoderEmbedding(nn.Module):
    def __init__(self, n_responses, n_dims, seq_len):
        super(DecoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.response_embed = nn.Embedding(n_responses, n_dims)
        self.time_embed = nn.Linear(1, n_dims, bias=False)
        self.position_embed = nn.Embedding(seq_len, n_dims)

    def forward(self, responses):
        e = self.response_embed(responses)
        seq = torch.arange(self.seq_len, device=Config.device).unsqueeze(0)
        p = self.position_embed(seq)
        return p + e


class StackedNMultiHeadAttention(nn.Module):
    def __init__(self, n_stacks, n_dims, n_heads, seq_len, n_multihead=1, dropout=0.0):
        super(StackedNMultiHeadAttention, self).__init__()
        self.n_stacks = n_stacks
        self.n_multihead = n_multihead
        self.n_dims = n_dims
        self.norm_layers = nn.LayerNorm(n_dims)
        # n_stacks has n_multiheads each
        self.multihead_layers = nn.ModuleList(n_stacks*[nn.ModuleList(n_multihead*[nn.MultiheadAttention(embed_dim=n_dims,
                                                                                                         num_heads=n_heads,
                                                                                                         dropout=dropout), ]), ])
        self.ffn = nn.ModuleList(n_stacks*[FFN(n_dims)])
        self.mask = torch.triu(torch.ones(seq_len, seq_len),
                               diagonal=1).to(dtype=torch.bool)

    def forward(self, input_q, input_k, input_v, encoder_output=None, break_layer=None):
        for stack in range(self.n_stacks):
            for multihead in range(self.n_multihead):
                norm_q = self.norm_layers(input_q)
                norm_k = self.norm_layers(input_k)
                norm_v = self.norm_layers(input_v)
                heads_output, _ = self.multihead_layers[stack][multihead](query=norm_q.permute(1, 0, 2),
                                                                          key=norm_k.permute(
                                                                              1, 0, 2),
                                                                          value=norm_v.permute(
                                                                              1, 0, 2),
                                                                          attn_mask=self.mask.to(Config.device))
                heads_output = heads_output.permute(1, 0, 2)
                #assert encoder_output != None and break_layer is not None
                if encoder_output != None and multihead == break_layer:
                    assert break_layer <= multihead, " break layer should be less than multihead layers and postive integer"
                    input_k = input_v = encoder_output
                    input_q = input_q + heads_output
                else:
                    input_q = input_q + heads_output
                    input_k = input_k + heads_output
                    input_v = input_v + heads_output
            last_norm = self.norm_layers(heads_output)
            ffn_output = self.ffn[stack](last_norm)
            ffn_output = ffn_output + heads_output
        # after loops = input_q = input_k = input_v
        return ffn_output


class PlusSAINTModule(pl.LightningModule):
    def __init__(self):
        # n_encoder,n_detotal_responses,seq_len,max_time=300+1
        super(PlusSAINTModule, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.encoder_layer = StackedNMultiHeadAttention(n_stacks=Config.NUM_DECODER,
                                                        n_dims=Config.EMBED_DIMS,
                                                        n_heads=Config.DEC_HEADS,
                                                        seq_len=Config.MAX_SEQ,
                                                        n_multihead=1, dropout=0.0)
        self.decoder_layer = StackedNMultiHeadAttention(n_stacks=Config.NUM_ENCODER,
                                                        n_dims=Config.EMBED_DIMS,
                                                        n_heads=Config.ENC_HEADS,
                                                        seq_len=Config.MAX_SEQ,
                                                        n_multihead=2, dropout=0.0)
        self.encoder_embedding = EncoderEmbedding(n_exercises=Config.TOTAL_EXE,
                                                  n_categories=Config.TOTAL_CAT,
                                                  n_dims=Config.EMBED_DIMS, seq_len=Config.MAX_SEQ)
        self.decoder_embedding = DecoderEmbedding(
            n_responses=3, n_dims=Config.EMBED_DIMS, seq_len=Config.MAX_SEQ)
        self.elapsed_time = nn.Linear(1, Config.EMBED_DIMS)
        self.fc = nn.Linear(Config.EMBED_DIMS, 1)

    def forward(self, x, y):
        enc = self.encoder_embedding(
            exercises=x["input_ids"], categories=x['input_cat'])
        dec = self.decoder_embedding(responses=y)
        elapsed_time = x["input_rtime"].unsqueeze(-1).float()
        ela_time = self.elapsed_time(elapsed_time)
        dec = dec + ela_time
        # this encoder
        encoder_output = self.encoder_layer(input_k=enc,
                                            input_q=enc,
                                            input_v=enc)
        #this is decoder
        decoder_output = self.decoder_layer(input_k=dec,
                                            input_q=dec,
                                            input_v=dec,
                                            encoder_output=encoder_output,
                                            break_layer=1)
        # fully connected layer
        out = self.fc(decoder_output)
        return out.squeeze()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_ids):
        input, labels = batch
        target_mask = (input["input_ids"] != 0)
        out = self(input, labels)
        # 统计标签信息（每10个batch打印一次）
        if batch_ids % 10 == 0:
            valid_labels = torch.masked_select(labels, target_mask)

            # 打印原始标签值（前20个）
            raw_labels = valid_labels[:20].cpu().numpy()  # 取前20个并转为numpy数组
            print(f"\nStep {self.global_step} 原始标签值: {raw_labels.tolist()}")

            # 保留原有的统计记录（可选）
            label_stats = {
                "样本量": valid_labels.shape[0],
                "正样本比例": torch.mean(valid_labels.float()).item()
            }
            self.logger.experiment.add_text(
                "标签统计",
                str(label_stats),
                global_step=self.global_step
            )
        loss = self.loss(out.float(), labels.float())
        out = torch.masked_select(out, target_mask)
        out = torch.sigmoid(out)
        labels = torch.masked_select(labels, target_mask)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        # Accumulate outputs for epoch end
        if not hasattr(self, 'train_epoch_outputs'):
            self.train_epoch_outputs = []
        self.train_epoch_outputs.append({"outs": out.detach().cpu(), "labels": labels.detach().cpu()})
        return {"loss": loss, "outs": out, "labels": labels}

    def on_train_epoch_end(self):
        if hasattr(self, 'train_epoch_outputs') and len(self.train_epoch_outputs) > 0:
            out = np.concatenate([i["outs"].numpy() for i in self.train_epoch_outputs]).reshape(-1)
            labels = np.concatenate([i["labels"].numpy() for i in self.train_epoch_outputs]).reshape(-1)
            auc = roc_auc_score(labels, out)
            self.print("train auc", auc)
            self.log("train_auc", auc)
            self.train_epoch_outputs = []  # Clear for next epoch

    def validation_step(self, batch, batch_ids):
        input, labels = batch
        target_mask = (input["input_ids"] != 0)
        out = self(input, labels)
        loss = self.loss(out.float(), labels.float())
        out = torch.masked_select(out, target_mask)
        out = torch.sigmoid(out)
        labels = torch.masked_select(labels, target_mask)
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        # Accumulate outputs for epoch end
        if not hasattr(self, 'val_epoch_outputs'):
            self.val_epoch_outputs = []
        self.val_epoch_outputs.append({"outs": out.detach().cpu(), "labels": labels.detach().cpu()})
        return {"val_loss": loss, "outs": out, "labels": labels}

    def on_validation_epoch_end(self):
        if hasattr(self, 'val_epoch_outputs') and len(self.val_epoch_outputs) > 0:
            out = np.concatenate([i["outs"].numpy() for i in self.val_epoch_outputs]).reshape(-1)
            labels = np.concatenate([i["labels"].numpy() for i in self.val_epoch_outputs]).reshape(-1)
            auc = roc_auc_score(labels, out)
            self.print("val auc", auc)
            self.log("val_auc", auc)
            self.val_epoch_outputs = []  # Clear for next epoch


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    saint_plus = PlusSAINTModule()
    # 初始化 TensorBoardLogger
    logger = TensorBoardLogger("logs/", name="saint_plus_experiment")
    # 在 Trainer 中添加回调函数
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc',  # 监控验证集 AUC
        mode='max',  # 取最大值
        save_top_k=1,  # 保存最好的一个模型
        dirpath='./saved_models',  # 保存路径
        filename='best_model'  # 文件名
    )
    # 修改代码参数
    trainer = pl.Trainer(
        logger=logger,
        accelerator="auto",  # 自动检测硬件
        devices="auto",  # 自动使用可用设备
        max_epochs=2,
        enable_progress_bar=True,  # 替换旧版progress_bar_refresh_rate
        enable_model_summary=True,  # 新版默认需要显式启用摘要
        callbacks=[checkpoint_callback]
    )
    trainer.fit(
        model=saint_plus,
        train_dataloaders=train_loader,  # 使用复数形式参数名
        val_dataloaders=val_loader,
    )
