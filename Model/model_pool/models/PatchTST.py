import torch
from torch import nn
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import PatchEmbedding
from .layers.RevIN import RevIN

'From Time Series Library, enc_in is equal to d_feat, d_model is another value, like hidden_size'


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
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.de_norm = configs.de_norm

        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(configs.d_feat, affine=configs.affine, subtract_last=configs.subtract_last)
        padding = stride

        # patching and embedding
        # self.patch_embedding = PatchEmbedding(configs.d_feat, patch_len, stride, padding, configs.dropout)
        self.patch_embedding = PatchEmbedding(configs.hidden_size, patch_len, stride, padding, configs.dropout)

        # Encoder
        # try to change d_model to higher number in order to contain more info?
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.hidden_size, configs.n_heads),
                    configs.hidden_size,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.hidden_size)
        )

        # Prediction Head
        self.head_nf = configs.hidden_size * int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast':
            self.head = FlattenHead(configs.d_feat, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'regression':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.d_feat, 1)
        elif self.task_name == 'multi-class':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.d_feat, configs.num_class)
        elif self.task_name == 'rep_learning':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        x_enc = x_enc.reshape(len(x_enc), self.d_feat, -1)  # [N, F, T]
        if self.revin:
            x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
            x_enc = self.revin_layer(x_enc, 'norm')
        else:
            x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [bs, d_feat, seq_len]
        # u: [bs * nvars x patch_num x d_model], nvars == d_feat
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
        dec_out = self.head(enc_out)  # z: [N, F, T]
        dec_out = dec_out.permute(0, 2, 1)  # dec_out [N, T, F]

        # De-Normalization from Non-stationary Transformer
        if self.revin:
            dec_out = self.revin_layer(dec_out, 'denorm')
            return dec_out
        elif self.de_norm:
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            return dec_out
        else:
            return dec_out

    def regression(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        x_enc = x_enc.reshape(len(x_enc), self.d_feat, -1)  # [N, F, T]
        if self.revin:
            x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
            x_enc = self.revin_layer(x_enc, 'norm')
        else:
            x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
            x_mark_enc = x_mark_enc.reshape(len(x_mark_enc), self.d_feat, -1)
            x_mark_enc = x_mark_enc.permute(0, 2, 1)
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [batch_num, d_feat, seq_len]
        # u: [bs * nvars x patch_num x d_model] / [batch_num*(seq_len/batch_num), batch_num, d_feat]
        enc_out, n_vars = self.patch_embedding(x_enc)
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        # the encoder part won't change the shape of enc_out no matter how many layer we use
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        if self.revin:
            enc_out = enc_out.permute(0, 1, 3, 2)
            enc_out = torch.reshape(enc_out, (enc_out.shape[0], -1, enc_out.shape[-1]))
            enc_out = self.revin_layer(enc_out, 'denorm')

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def rep_learning(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer or use REVin
        x_enc = x_enc.reshape(len(x_enc), self.d_feat, -1)  # [N, F, T]
        if self.revin:
            x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
            x_enc = self.revin_layer(x_enc, 'norm')
        else:
            x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
            x_mark_enc = x_mark_enc.reshape(len(x_mark_enc), self.d_feat, -1)
            x_mark_enc = x_mark_enc.permute(0, 2, 1)
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [batch_num, d_feat, seq_len]
        # u: [bs * nvars x patch_num x d_model] / [batch_num*(seq_len/batch_num), batch_num, d_feat]
        enc_out, n_vars = self.patch_embedding(x_enc)
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        # the encoder part won't change the shape of enc_out no matter how many layer we use
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        if self.revin:
            enc_out = enc_out.permute(0, 1, 3, 2)
            enc_out = torch.reshape(enc_out, (enc_out.shape[0], -1, enc_out.shape[-1]))
            enc_out = self.revin_layer(enc_out, 'denorm')

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        return output  # shape [B, head_nf*d_feat]

    def classification(self, x_enc, x_mark_enc):
        x_enc = x_enc.reshape(len(x_enc), self.d_feat, -1)  # [N, F, T]
        if self.revin:
            x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
            x_enc = self.revin_layer(x_enc, 'norm')
        else:
            # not do the normalization while classification
            x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
            x_mark_enc = x_mark_enc.reshape(len(x_mark_enc), self.d_feat, -1)
            x_mark_enc = x_mark_enc.permute(0, 2, 1)
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # batch_size, d_feat, seq_len
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [batch_size*d_feat, patch_num, hidden_size]

        # Encoder
        # [batch_size*d_feat, patch_num, hidden_size]
        enc_out, attns = self.encoder(enc_out)  # encoder never change the dim of enc_out
        # [batch_size， d_feat, patch_num, hidden_size]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # z: [bs x d_feat x hidden_size x patch_num]

        # classification task not need to do denormalize since we don't need forecasting series
        # if self.revin:
        #     # revin used in long-term-forecasting, but here we use in multi-class to try to avoid distribution shift
        #     enc_out = enc_out.permute(0, 2, 3, 1)  # [bs, hidden_size, patch_num, d_feat]
        #     enc_out = torch.reshape(enc_out, (enc_out.shape[0], -1, enc_out.shape[-1]))
        #     enc_out = self.revin_layer(enc_out, 'denorm')

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc):
        if self.task_name == 'regression':
            dec_out = self.regression(x_enc, x_mark_enc)
            return dec_out.squeeze()  # [B, N]
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'multi-class':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, Number of classes]
        if self.task_name == 'rep_learning':
            dec_out = self.rep_learning(x_enc, x_mark_enc)
            return dec_out
        return None
