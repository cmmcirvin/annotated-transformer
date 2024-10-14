import math

import lightning as L
import torch
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

import datasets

class ShakespeareDataset(Dataset):

    def __init__(self, seq_len=128, num_batches=10000):
        self.pad = 0
        self.seq_len = seq_len
        self.num_batches = num_batches

        self.data = datasets.load_dataset('tiny_shakespeare')['train']['text'][0]
        self.vocabulary = sorted(set(self.data))
        self.V = len(self.vocabulary)
        self.mapping = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.reverse_mapping = {idx: char for idx, char in enumerate(self.vocabulary)}

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        start_idx = torch.randint(0, len(self.data) - self.seq_len - 1, (1,)).item()

        # Get text from the dataset of length self.seq_len
        text = self.data[start_idx:start_idx + self.seq_len]

        # Apply character-level encoding
        encoded_text = torch.tensor([self.mapping[char] for char in text])

        src = encoded_text
        src_mask = (src != self.pad).unsqueeze(-2)
        tgt = encoded_text[:-1]
        tgt_y = encoded_text[1:]
        tgt_mask = (tgt != self.pad).unsqueeze(-2)

        subsequent_mask = torch.triu(torch.ones((1, tgt.size(-1), tgt.size(-1))), diagonal=1).type( torch.uint8) == 0
        tgt_mask = tgt_mask & subsequent_mask.type_as(tgt_mask.data)[0]
        ntokens = (tgt_y != self.pad).data.sum()

        return src, tgt, src_mask, tgt_mask, tgt_y, ntokens

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

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(h, d_model, d_ff, dropout) for _ in range(N)
        ])
        self.encoder_out_norm_gain = nn.Parameter(torch.ones(d_model))
        self.encoder_out_norm_bias = nn.Parameter(torch.zeros(d_model))

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(h, d_model, d_ff, dropout) for _ in range(N)
        ])
        self.decoder_out_norm_gain = nn.Parameter(torch.ones(d_model))
        self.decoder_out_norm_bias = nn.Parameter(torch.zeros(d_model))

        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)

        self.generator = nn.Sequential(
            nn.Linear(d_model, tgt_vocab), nn.LogSoftmax(dim=-1)
        )

        self.criterion = nn.KLDivLoss(reduction="sum")

        self.V = tgt_vocab

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

        e_x = src_embedding
        for block in self.encoder_blocks:
            e_x = block(e_x, src_mask)

        memory = self.encoder_out_norm_gain * (e_x - e_x.mean(-1, keepdim=True)) / (e_x.std(-1, keepdim=True) + self.eps) + self.encoder_out_norm_bias

        d_x = tgt_embedding
        for block in self.decoder_blocks:
            d_x = block(d_x, memory, src_mask, tgt_mask)

        decoder_output = self.decoder_out_norm_gain * (d_x - d_x.mean(-1, keepdim=True)) / (d_x.std(-1, keepdim=True) + self.eps) + self.decoder_out_norm_bias

        return decoder_output

    def loss(self, x, y, norm):
        x = self.generator(x)
        x = x.contiguous().view(-1, x.size(-1))
        y = y.contiguous().view(-1)
        size = self.V
        padding_idx = 0
        smoothing = 0.0
        confidence = 1 - smoothing

        assert x.size(1) == size
        true_dist = x.data.clone()
        true_dist.fill_(smoothing / (size - 2))
        true_dist.scatter_(1, y.data.unsqueeze(1), confidence)
        true_dist[:, padding_idx] = 0
        mask = torch.nonzero(y.data == padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        return self.criterion(x, true_dist.clone().detach()) / norm

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
            tgt_mask = (torch.triu(torch.ones((1, tgt.size(1), tgt.size(1))), diagonal=1).type( torch.uint8) == 0).type_as(src.data)
            out = self.forward((src, tgt, src_mask, tgt_mask))
            prob = self.generator(out[:, -1])
            next_word = torch.argmax(prob, dim=1).item()
            tgt = torch.cat(
                [tgt, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return tgt


class EncoderBlock(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.ff_w_1 = nn.Linear(d_model, d_ff)
        self.ff_w_2 = nn.Linear(d_ff, d_model)

        self.layer_norm_gain_1 = nn.Parameter(torch.ones(d_model))
        self.layer_norm_gain_2 = nn.Parameter(torch.ones(d_model))
        self.layer_norm_bias_1 = nn.Parameter(torch.zeros(d_model))
        self.layer_norm_bias_2 = nn.Parameter(torch.zeros(d_model))

        self.dropout = nn.Dropout(p=dropout)
        self.eps = 1e-6

    def normalize(self, x):
        return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps)

    def forward(self, x, mask):
        layer_norm_1_x = (
            self.layer_norm_gain_1 * self.normalize(x) + self.layer_norm_bias_1
        )
        x = x + self.dropout(
            self.self_attn(layer_norm_1_x, layer_norm_1_x, layer_norm_1_x, mask)
        )
        layer_norm_2_x = (
            self.layer_norm_gain_2 * self.normalize(x) + self.layer_norm_bias_2
        )
        x = x + self.dropout(
            self.ff_w_2(self.dropout(self.ff_w_1(layer_norm_2_x).relu()))
        )

        return x


class DecoderBlock(nn.Module):

    def __init__(self, h, d_model, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.src_attn = MultiHeadedAttention(h, d_model)
        self.ff_w_1 = nn.Linear(d_model, d_ff)
        self.ff_w_2 = nn.Linear(d_ff, d_model)

        self.layer_norm_gain_1 = nn.Parameter(torch.ones(d_model))
        self.layer_norm_gain_2 = nn.Parameter(torch.ones(d_model))
        self.layer_norm_gain_3 = nn.Parameter(torch.ones(d_model))
        self.layer_norm_bias_1 = nn.Parameter(torch.zeros(d_model))
        self.layer_norm_bias_2 = nn.Parameter(torch.zeros(d_model))
        self.layer_norm_bias_3 = nn.Parameter(torch.zeros(d_model))

        self.dropout = nn.Dropout(p=dropout)
        self.eps = 1e-6

    def normalize(self, x):
        return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory

        layer_norm_1_x = (
            self.layer_norm_gain_1 * self.normalize(x) + self.layer_norm_bias_1
        )
        x = x + self.dropout(
            self.self_attn(layer_norm_1_x, layer_norm_1_x, layer_norm_1_x, tgt_mask)
        )

        layer_norm_2_x = (
            self.layer_norm_gain_2 * self.normalize(x) + self.layer_norm_bias_2
        )
        x = x + self.dropout(self.src_attn(layer_norm_2_x, m, m, src_mask))

        layer_norm_3_x = (
            self.layer_norm_gain_3 * self.normalize(x) + self.layer_norm_bias_3
        )
        x = x + self.dropout(
            self.ff_w_2(self.dropout(self.ff_w_1(layer_norm_3_x).relu()))
        )

        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query = (
            self.query_linear(query)
            .view(nbatches, -1, self.h, self.d_k)
            .transpose(1, 2)
        )
        key = self.key_linear(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = (
            self.value_linear(value)
            .view(nbatches, -1, self.h, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.out_linear(x)


def main():
#    dataset = CopyDataset(V, num_items=100)
    dataset = ShakespeareDataset(num_batches=100000)

    model = EncoderDecoder(
        h=8,
        d_model=512,
        d_ff=2048,
        dropout=0.1,
        N=4,
        src_vocab=dataset.V,
        tgt_vocab=dataset.V,
    )


    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=11, persistent_workers=True)
    iter(dataloader)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = Trainer(max_epochs=1, devices=1, logger=tb_logger)
    trainer.fit(model, dataloader)

    model.eval()
    query_string = "Talk not to me: "
    src = torch.tensor([[dataset.mapping[char] for char in query_string]])
    src_mask = (src != 0).unsqueeze(-2)

    max_len = src.shape[1]
    out = model.greedy_decode(src, src_mask, max_len=max_len, start_symbol=0)

    print("".join([dataset.reverse_mapping[int(idx.item())] for idx in out.squeeze()]))


if __name__ == "__main__":
    main()
