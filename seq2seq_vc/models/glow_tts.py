import math
import torch
import torch.nn.functional as F

from seq2seq_vc.layers.positional_encoding import RelPositionalEncoding, PositionalEncoding
from seq2seq_vc.modules.conformer.encoder import Encoder as ConformerEncoder
from seq2seq_vc.modules.duration_predictor import DurationPredictor
from seq2seq_vc.modules.glow_tts.decoder import FlowDecoder

from seq2seq_vc.monotonic_align import maximum_path

class GlowTTS(torch.nn.Module):
    def __init__(
        self,
        idim,
        odim,
        adim=384,
        aheads=4,
        elayers=6,
        eunits=1536,
        encoder_normalize_before=True,
        encoder_concat_after=False,
        encoder_dropout_rate: float = 0.1,
        encoder_positional_dropout_rate: float = 0.1,
        encoder_attn_dropout_rate: float = 0.1,
        encoder_positionwise_layer_type: str = "conv1d",
        encoder_positionwise_conv_kernel_size: int = 3,
        encoder_pos_enc_layer_type: str = "rel_pos",
        encoder_self_attn_layer_type: str = "rel_selfattn",
        encoder_use_macaron_style: bool = True,
        encoder_use_cnn: bool = True,
        encoder_zero_triu: bool = False,
        encoder_kernel_size: int = 7,
        # GlowTTS related
        mean_only=True,
        duration_predictor_chans=384,
        duration_predictor_kernel_size=3,
        duration_predictor_dropout_rate=0.1,
        decoder_kernel_size=5,
        decoder_dilation_rate=1,
        decoder_n_blocks=12,
        decoder_n_block_layers=4,
        decoder_dropout_rate=0.05,
        decoder_n_split=4,
        decoder_n_sqz=2,
        decoder_sigmoid_scale=False,
        # speaker embedding related
        spk_embed_dim=None,
        spk_embed_integration_type="add",
    ):
        # initialize base classes
        torch.nn.Module.__init__(self)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.adim = adim
        self.spk_embed_dim = spk_embed_dim
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type
        self.mean_only = mean_only
        self.n_split = decoder_n_split
        self.n_sqz = decoder_n_sqz

        # use idx 0 as padding idx
        self.padding_idx = 0

        # define encoder
        self.emb = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
        )
        torch.nn.init.normal_(self.emb.weight, 0.0, adim**-0.5)
        self.encoder = ConformerEncoder(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=None,
            normalize_before=encoder_normalize_before,
            concat_after=encoder_concat_after,
            positionwise_layer_type=encoder_positionwise_layer_type,  # V
            positionwise_conv_kernel_size=encoder_positionwise_conv_kernel_size,  # V
            # the following args are unique to Conformer
            dropout_rate=encoder_dropout_rate,
            positional_dropout_rate=encoder_positional_dropout_rate,
            attention_dropout_rate=encoder_attn_dropout_rate,
            macaron_style=encoder_use_macaron_style,
            pos_enc_layer_type=encoder_pos_enc_layer_type,
            selfattention_layer_type=encoder_self_attn_layer_type,
            use_cnn_module=encoder_use_cnn,
            cnn_module_kernel=encoder_kernel_size,
            zero_triu=encoder_zero_triu,
        )

        self.proj_m = torch.nn.Linear(adim, odim)
        if not mean_only:
            self.proj_s = torch.nn.Linear(adim, odim)
        self.duration_predictor = DurationPredictor(
            adim,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate
        )

        # define projection layer
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define decoder
        self.decoder = FlowDecoder(
            odim, 
            adim,
            decoder_kernel_size, 
            decoder_dilation_rate, 
            decoder_n_blocks,
            decoder_n_block_layers, 
            p_dropout=decoder_dropout_rate, 
            n_split=decoder_n_split,
            n_sqz=decoder_n_sqz,
            sigmoid_scale=decoder_sigmoid_scale,
        )

    def _encode(self, x, x_mask=None):
        """
            x: shape [B, t]
            x_mask: shape [B, 1, t]
        """

        # forward encoder
        x = self.emb(x) * math.sqrt(self.adim)
        x, h_mask = self.encoder(x, x_mask)
        h_mask = torch.squeeze(h_mask, 1)  # [B, 1, t] -> [B, t]

        # stop gradient flow to duration predictor
        x_dp = torch.detach(x)
        log_durations = self.duration_predictor(x_dp, h_mask)

        # pred
        x_m = self.proj_m(x) * torch.unsqueeze(h_mask, -1)
        if not self.mean_only:
            x_logs = self.proj_s(x) * torch.unsqueeze(h_mask, -1)
        else:
            x_logs = torch.zeros_like(x_m)
        
        return x_m, x_logs, log_durations, h_mask.unsqueeze(1)

    def forward(self, x, x_lengths, y, y_lengths=None, g=None, *args, **kwargs):
        """
            x: [B, t, h]
            y: [B, t', 80]
        """

        # forward encoder
        # x_m: [B, t, d]
        # x_logs: [B, t, d]
        # x_mask: [B, 1, t]
        # log_durations: [B, t]
        x_mask = self.sequence_mask(x_lengths).to(x.device) # x_mask: [B, t]
        x_m, x_logs, log_durations, x_mask = self._encode(
            x,
            x_mask=x_mask.unsqueeze(-2) # required by transformer encoder
        ) # x_m: [B, t, d], x_mask: [B, 1, t]
        print("predicted durations")
        print((torch.exp(log_durations) * x_mask.squeeze(1))[:3])

         # flow decoder takes [B, 80, t']
        y = torch.transpose(y, 1, 2)

        y_max_length = y.size(2) # t'
        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        z_mask = torch.unsqueeze(self.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype) # [B, 1, t']
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2) # [B, 1, t, t']

        z, logdet = self.decoder(y, z_mask, g=None, reverse=False) # z: [B, d, t]
        with torch.no_grad():
            x_s_sq_r = torch.exp(-2 * x_logs) # [B, t, d]
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [2]).unsqueeze(-1) # [b, t, 1]
            logp2 = torch.matmul(x_s_sq_r, -0.5 * (z ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((x_m * x_s_sq_r), z) # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [2]).unsqueeze(-1) # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']

            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
        z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
        log_durations_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
        print("durations derived from attn")
        print((torch.sum(attn, -1) * x_mask).squeeze(1)[:3])
        return (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, log_durations, log_durations_)

    def inference(self, x, x_lengths, y=None, y_lengths=None, g=None, noise_scale=1., length_scale=1.):
        x = x.unsqueeze(0) # [1, t]
        x_mask = self.sequence_mask(x_lengths.unsqueeze(0), x.size(1)).to(x.device) # x_mask: [B, t]

        # forward encoder
        # x_m: [1, t, d]
        # x_logs: [1, t, d]
        # x_mask: [1, 1, t]
        # log_durations: [1, t]
        x_m, x_logs, log_durations, x_mask = self._encode(
            x,
            x_mask,
        )

        # calculate duration with predicted duration
        durations = torch.exp(log_durations) * x_mask * length_scale # [1, t]
        durations_ceil = torch.ceil(durations)

        # generate attention
        y_lengths = torch.clamp_min(torch.sum(durations_ceil, [1, 2]), 1).long()
        y_max_length = None
        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        z_mask = torch.unsqueeze(self.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)  # [1, 1, t']
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2) # [1, 1, t, t']
        attn = self.generate_path(durations_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1).float() # [1, 1, t, t']

        # generate more stuff
        z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, t', d] -transpose-> [b, d, t']
        z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, t', d] -transpose-> [b, d, t']
        log_durations_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

        # forward decoder
        z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask # [B, d, t]
        y, logdet = self.decoder(z, z_mask, g=g, reverse=True)

        y = y.transpose(1, 2).squeeze(0) # [1, d, t] -> [t, d]

        return (y, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, log_durations, log_durations_)

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:,:,:y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length

    def _integrate_with_spk_embed(self, hs, spembs):
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim)

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def sequence_mask(self, length, max_length=None):
        # this is almost the same as self._source_mask except supporting max_length
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)

    def generate_path(self, duration, mask):
        """
        duration: [b, t_x]
        mask: [b, t_x, t_y]
        """
        device = duration.device

        b, t_x, t_y = mask.shape
        cum_duration = torch.cumsum(duration, 1)
        path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

        cum_duration_flat = cum_duration.view(b * t_x)
        path = self.sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
        path = path.view(b, t_x, t_y)

        # "-" is not supported for bool. Use "^" (logical or).
        # path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:,:-1]
        path = path ^ F.pad(path, [0, 0, 1, 0, 0, 0])[:,:-1]
        
        path = path * mask
        return path
    
    def convert_pad_shape(self, pad_shape):
        l = pad_shape[::-1]
        pad_shape = [item for sublist in l for item in sublist]
        return pad_shape