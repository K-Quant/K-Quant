import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding
from .layers.AutoCorrelation import AutoCorrelationLayer
from .layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from .layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs, version='Wavelets', mode_select='random', modes=2):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        self.d_feat = configs.d_feat

        # Decomp
        self.enc_embedding = DataEmbedding(configs.d_feat, configs.d_feat, configs.embed, configs.freq,
                                           configs.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_feat, L=1, base='legendre')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_feat,
                                            out_channels=configs.d_feat,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        configs.d_feat, configs.n_heads),
                    configs.d_feat,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_feat)
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
        output = self.act(enc_out)
        output = self.dropout(output)
        output = (output * x_mark_enc).unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.regression(x_enc, x_mark_enc)
        return dec_out.squeeze()  # [B, N]

