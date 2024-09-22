import math

import lightning as L
import torch
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


class CopyDataset(Dataset):
    def __init__(self, V=11, num_items=400):
        self.data = torch.randint(1, V, size=(num_items, 10))
        self.pad = 0

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        src = self.data[idx]
        src_mask = (src != self.pad).unsqueeze(-2)
        tgt = self.data[idx][:-1]
        tgt_y = self.data[idx][1:]
        tgt_mask = (tgt != self.pad).unsqueeze(-2)

        subsequent_mask = (
            torch.triu(torch.ones((1, tgt.size(-1), tgt.size(-1))), diagonal=1).type(
                torch.uint8
            )
            == 0
        )
        tgt_mask = tgt_mask & subsequent_mask.type_as(tgt_mask.data)[0]
        ntokens = (tgt_y != self.pad).data.sum()

        return src, tgt, src_mask, tgt_mask, tgt_y, ntokens


class EncoderDecoder(L.LightningModule):
    def __init__(self, h, d_model, d_ff, dropout, N, src_vocab, tgt_vocab):
        super().__init__()

        self.eps = 1e-6

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        self.encoder_block_1 = EncoderLayer(h, d_model, d_ff, dropout)
        self.encoder_block_2 = EncoderLayer(h, d_model, d_ff, dropout)
        self.encoder_out_norm_gain = nn.Parameter(torch.ones(d_model))
        self.encoder_out_norm_bias = nn.Parameter(torch.zeros(d_model))

        self.decoder_block_1 = DecoderLayer(h, d_model, d_ff, dropout)
        self.decoder_block_2 = DecoderLayer(h, d_model, d_ff, dropout)
        self.decoder_out_norm_gain = nn.Parameter(torch.ones(d_model))
        self.decoder_out_norm_bias = nn.Parameter(torch.zeros(d_model))
        #        self.decoder = Decoder(h, d_model, d_ff, dropout, N)

        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)

        self.generator = nn.Sequential(
            nn.Linear(d_model, tgt_vocab), nn.LogSoftmax(dim=-1)
        )

        self.criterion = LabelSmoothing(size=11, padding_idx=0, smoothing=0.0)

        self._init_weights()
        self._init_positional_encodings()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_positional_encodings(self, max_len=5000):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        src_pe = pe.unsqueeze(0).requires_grad_(False)
        self.register_buffer("src_pe", src_pe)

        tgt_pe = pe.unsqueeze(0).requires_grad_(False)
        self.register_buffer("tgt_pe", tgt_pe)

    def pe(self, x, mode):
        pe = self.src_pe if mode == "src" else self.tgt_pe
        return self.dropout(x + pe[:, : x.size(1)])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: 1.0
            * (
                self.d_model ** (-0.5)
                * min(max(step, 1) ** (-0.5), max(step, 1) * 400 ** (-1.5))
            ),
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, batch):
        src, tgt, src_mask, tgt_mask = batch

        src_embedding = self.src_embed(src) * math.sqrt(self.d_model)
        src_embedding = self.pe(src_embedding, mode="src")
        tgt_embedding = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_embedding = self.pe(tgt_embedding, mode="tgt")

        e_x = self.encoder_block_1(src_embedding, src_mask)
        e_x = self.encoder_block_2(e_x, src_mask)

        memory = (
            self.encoder_out_norm_gain
            * (e_x - e_x.mean(-1, keepdim=True))
            / (e_x.std(-1, keepdim=True) + self.eps)
            + self.encoder_out_norm_bias
        )

        d_x = self.decoder_block_1(tgt_embedding, memory, src_mask, tgt_mask)
        d_x = self.decoder_block_2(d_x, memory, src_mask, tgt_mask)
        decoder_output = (
            self.decoder_out_norm_gain
            * (d_x - d_x.mean(-1, keepdim=True))
            / (d_x.std(-1, keepdim=True) + self.eps)
            + self.decoder_out_norm_bias
        )

        return decoder_output

    def loss(self, x, y, norm):
        x = self.generator(x)
        return (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )

    def training_step(self, batch, batch_idx):
        src, tgt, src_mask, tgt_mask, tgt_y, ntokens = batch
        out = self.forward((src, tgt, src_mask, tgt_mask))
        loss = self.loss(out, tgt_y, ntokens.sum())

        self.lr_schedulers().step()
        self.log("train_loss", loss)
        self.log("learning_rate", self.lr_schedulers()._last_lr[0])
        return loss

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        tgt = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for _ in range(max_len - 1):
            tgt_mask = (
                torch.triu(torch.ones((1, tgt.size(1), tgt.size(1))), diagonal=1).type(
                    torch.uint8
                )
                == 0
            ).type_as(src.data)
            out = self.forward((src, tgt, src_mask, tgt_mask))
            prob = self.generator(out[:, -1])
            next_word = torch.argmax(prob, dim=1).item()
            tgt = torch.cat(
                [tgt, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return tgt


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm_gain = nn.Parameter(torch.ones(size))
        self.norm_bias = nn.Parameter(torch.zeros(size))
        self.eps = 1e-6

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        normalized_x = (
            self.norm_gain
            * (x - x.mean(-1, keepdim=True))
            / (x.std(-1, keepdim=True) + self.eps)
            + self.norm_bias
        )
        return x + self.dropout(sublayer(normalized_x))


class EncoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
#        self.norm_gain_1 = nn.Parameter(torch.ones(d_model))
#        self.norm_gain_2 = nn.Parameter(torch.ones(d_model))
#        self.norm_bias_1 = nn.Parameter(torch.zeros(d_model))
#        self.norm_bias_2 = nn.Parameter(torch.zeros(d_model))

        self.sublayer = nn.ModuleList(
            [
                SublayerConnection(d_model, dropout),
                SublayerConnection(d_model, dropout),
            ]
        )
        self.size = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, h, d_model, d_ff, dropout):
        super().__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.src_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = nn.ModuleList(
            [
                SublayerConnection(d_model, dropout),
                SublayerConnection(d_model, dropout),
                SublayerConnection(d_model, dropout),
            ]
        )

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList(
            [
                nn.Linear(d_model, d_model),
                nn.Linear(d_model, d_model),
                nn.Linear(d_model, d_model),
                nn.Linear(d_model, d_model),
            ]
        )
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def main():
    V = 11

    model = EncoderDecoder(
        h=8,
        d_model=512,
        d_ff=2048,
        dropout=0.1,
        N=2,
        src_vocab=V,
        tgt_vocab=V,
    )

    dataset = CopyDataset(V, num_items=40000)
    dataloader = DataLoader(
        dataset, batch_size=80, shuffle=True, num_workers=11, persistent_workers=True
    )
    iter(dataloader)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = Trainer(max_epochs=1, devices=1, logger=tb_logger)
    trainer.fit(model, dataloader)

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(model.greedy_decode(src, src_mask, max_len=max_len, start_symbol=0))


if __name__ == "__main__":
    main()
