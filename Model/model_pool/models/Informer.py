import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention_Family import ProbAttention, AttentionLayer
from .layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_feat = configs.d_feat

        # Embedding
        self.enc_embedding = DataEmbedding(configs.d_feat, configs.d_feat, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_feat, configs.n_heads),
                    configs.d_feat,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_feat
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_feat)
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_feat * configs.seq_len, 1)

    def regression(self, x_enc, x_mark_enc):
        # enc
        x_enc = x_enc.reshape(len(x_enc), self.d_feat, -1)  # [N, F, T]
        x_enc = x_enc.permute(0, 2, 1)  # [N, T, F]
        x_mark_enc = x_mark_enc.reshape(len(x_mark_enc), self.d_feat, -1)
        x_mark_enc = x_mark_enc.permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = (output * x_mark_enc).unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.regression(x_enc, x_mark_enc)
        return dec_out.squeeze()  # [B, N]
