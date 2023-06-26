import torch
import torch.nn as nn
from .layers.Embed import DataEmbedding
from .layers.ETSformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder, Transform


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2202.01381
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.d_feat = configs.d_feat
        # we don't need decoder, so just ignore this
        # assert configs.e_layers == configs.d_layers, "Encoder and decoder layers must be equal"

        # Embedding
        self.enc_embedding = DataEmbedding(configs.d_feat, configs.d_feat, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_feat, configs.n_heads, configs.d_feat, configs.seq_len, self.pred_len, configs.top_k,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ]
        )

        self.act = torch.nn.functional.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_feat * configs.seq_len, 1)

    def regression(self, x_enc, x_mark_enc):
        x_enc = x_enc.reshape(len(x_enc), self.d_feat, -1)  # [N, F, T]
        x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
        x_mark_enc = x_mark_enc.reshape(len(x_mark_enc), self.d_feat, -1)
        x_mark_enc = x_mark_enc.permute(0, 2, 1)
        res = self.enc_embedding(x_enc, None)
        _, growths, seasons = self.encoder(res, x_enc, attn_mask=None)

        growths = torch.sum(torch.stack(growths, 0), 0)[:, :self.seq_len, :]
        seasons = torch.sum(torch.stack(seasons, 0), 0)[:, :self.seq_len, :]

        enc_out = growths + seasons
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        # Output
        output = (output * x_mark_enc).unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.regression(x_enc, x_mark_enc)
        return dec_out.squeeze()  # [B, N]
