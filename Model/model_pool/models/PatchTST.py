import torch
from torch import nn
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = configs.seq_len
        self.d_feat = configs.d_feat
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(configs.d_feat, patch_len, stride, padding, configs.dropout)

        # Encoder
        # try to change d_model to higher number in order to contain more info?
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_feat, configs.n_heads),
                    configs.d_feat,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_feat)
        )

        # Prediction Head
        self.head_nf = configs.d_feat * int((configs.seq_len - patch_len) / stride + 2)

        self.flatten = nn.Flatten(start_dim=-2)
        self.pre_projection = nn.Linear(self.head_nf * configs.d_feat, configs.d_bp)
        self.dropout = nn.Dropout(configs.dropout)
        # self.projection = nn.Linear(self.head_nf * configs.d_feat, 1)
        self.projection = nn.Linear(configs.d_bp, 1)

    def regression(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        x_enc = x_enc.reshape(len(x_enc), self.d_feat, -1)  # [N, F, T]
        x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
        x_mark_enc = x_mark_enc.reshape(len(x_mark_enc), self.d_feat, -1)
        x_mark_enc = x_mark_enc.permute(0, 2, 1)

        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [batch_num, d_feat, seq_len]
        # u: [bs * nvars x patch_num x d_model] / [batch_num*(seq_len/batch_num), batch_num, d_feat]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.pre_projection(output)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.regression(x_enc, x_mark_enc)
        return dec_out.squeeze()  # [B, N]
