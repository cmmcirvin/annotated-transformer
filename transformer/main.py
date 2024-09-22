import math
import time

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

        subsequent_mask = torch.triu( torch.ones((1, tgt.size(-1), tgt.size(-1))), diagonal=1).type(torch.uint8) == 0
        tgt_mask = tgt_mask & subsequent_mask.type_as(tgt_mask.data)[0]
        ntokens = (tgt_y != self.pad).data.sum()

        return src, tgt, src_mask, tgt_mask, tgt_y, ntokens


class EncoderDecoder(L.LightningModule):
    def __init__(self, h, d_model, d_ff, dropout, N, src_vocab, tgt_vocab):
        super().__init__()

        self.d_model = d_model

        self.encoder = Encoder(h, d_model, d_ff, dropout, N)
        self.decoder = Decoder(h, d_model, d_ff, dropout, N)

        self.src_embed = Embeddings(d_model, src_vocab)
        self.src_pos_enc = PositionalEncoding(d_model, dropout)
        self.tgt_embed = Embeddings(d_model, tgt_vocab)
        self.tgt_pos_enc = PositionalEncoding(d_model, dropout)

        self.generator = nn.Sequential(
            nn.Linear(d_model, tgt_vocab), nn.LogSoftmax(dim=-1)
        )

        self.criterion = LabelSmoothing(size=11, padding_idx=0, smoothing=0.0)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

        src_embedding = self.src_embed(src)
        src_embedding = self.src_pos_enc(src_embedding)
        tgt_embedding = self.tgt_embed(tgt)
        tgt_embedding = self.tgt_pos_enc(tgt_embedding)

        memory = self.encoder(src_embedding, src_mask)
        decoder_output = self.decoder(tgt_embedding, memory, src_mask, tgt_mask)

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
        src_embedding = self.src_embed(src)
        src_embedding = self.src_pos_enc(src_embedding)

        memory = self.encoder(src_embedding, src_mask)

        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for _ in range(max_len - 1):
            subsequent_mask = torch.triu( torch.ones((1, ys.size(1), ys.size(1))), diagonal=1).type(torch.uint8) == 0
            out = self.decoder(
                self.tgt_pos_enc(self.tgt_embed(ys)),
                memory,
                src_mask,
                subsequent_mask.type_as(src.data),
            )
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return ys


class Encoder(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout, N):
        super().__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(h, d_model, d_ff, dropout) for _ in range(N)]
        )
        self.out_norm_gain = nn.Parameter(torch.ones(d_model))
        self.out_norm_bias = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-6

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return (
            self.out_norm_gain
            * (x - x.mean(-1, keepdim=True))
            / (x.std(-1, keepdim=True) + self.eps)
            + self.out_norm_bias
        )


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm_gain = nn.Parameter(torch.ones(size))
        self.norm_bias = nn.Parameter(torch.ones(size))
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
        self.sublayer = nn.ModuleList(
            [SublayerConnection(d_model, dropout), SublayerConnection(d_model, dropout)]
        )
        self.size = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout, N):
        super().__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(h, d_model, d_ff, dropout) for _ in range(N)]
        )
        self.out_norm_gain = nn.Parameter(torch.ones(d_model))
        self.out_norm_bias = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-6

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return (
            self.out_norm_gain
            * (x - x.mean(-1, keepdim=True))
            / (x.std(-1, keepdim=True) + self.eps)
            + self.out_norm_bias
        )


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


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


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
        "Implements Figure 2"
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
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

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


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


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
    dataloader = DataLoader(dataset, batch_size=80, shuffle=True, num_workers=11, persistent_workers=True)
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
