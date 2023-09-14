#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import torch
import torch.nn.functional as F

from typing import Literal

from seq2seq_vc.layers.positional_encoding import ScaledPositionalEncoding
from seq2seq_vc.modules.transformer.encoder import Encoder as TransformerEncoder
from seq2seq_vc.modules.conformer.encoder import Encoder as ConformerEncoder
from seq2seq_vc.modules.transformer.decoder import Decoder
from seq2seq_vc.modules.pre_postnets import Prenet, Postnet
from seq2seq_vc.modules.transformer.mask import subsequent_mask
from seq2seq_vc.layers.utils import make_pad_mask, make_non_pad_mask
from seq2seq_vc.modules.transformer.attention import MultiHeadedAttention
from seq2seq_vc.modules.duration_predictor import (
    DurationPredictor,
    StochasticDurationPredictor,
)
from seq2seq_vc.modules.length_regulator import GaussianUpsampling

from seq2seq_vc.modules.transformer.subsampling import Conv2dSubsampling

from seq2seq_vc.modules.alignments import (
    AlignmentModule,
    average_by_duration,
    viterbi_decode
)

MAX_DP_OUTPUT = 10


class AASVC(torch.nn.Module):
    def __init__(
        self,
        idim,
        odim,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        dlayers: int = 6,
        dunits: int = 1536,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_input_layer: str = "linear",
        encoder_input_conv_kernel_size: int = 3,
        encoder_normalize_before: bool = False,
        decoder_normalize_before: bool = False,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        duration_predictor_use_encoder_outputs: bool = True,
        duration_predictor_input_dim: int = None,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        encoder_reduction_factor: int = 1,
        post_encoder_reduction_factor: int = 1,
        decoder_reduction_factor: int = 1,
        encoder_type: str = "conformer",
        decoder_type: str = "conformer",
        duration_predictor_type: str = "deterministic",
        # only for conformer
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        conformer_enc_kernel_size: int = 7,
        conformer_dec_kernel_size: int = 31,
        # pretrained spk emb
        spk_embed_dim: int = None,
        spk_embed_integration_type: str = "add",
        # training related
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        duration_predictor_dropout_rate: float = 0.1,
        postnet_dropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        # diffsinger
        diffsinger_denoiser_residual_channels: int = 256,
        # prodiff
        prodiff_denoiser_layers: int = 20,
        prodiff_denoiser_channels: int = 256,
        prodiff_diffusion_steps: int = 1000,
        prodiff_diffusion_timescale: int = 1,
        prodiff_diffusion_beta: float = 40.0,
        prodiff_diffusion_scheduler: str = "vpsde",
        prodiff_diffusion_cycle_ln: int = 1,
        # stochastic duration predictor
        stochastic_duration_predictor_kernel_size: int = 3,
        stochastic_duration_predictor_dropout_rate: float = 0.5,
        stochastic_duration_predictor_flows: int = 4,
        stochastic_duration_predictor_dds_conv_layers: int = 3,
        stochastic_duration_predictor_noise_scale: float = 0.8,
    ):
        # initialize base classes
        torch.nn.Module.__init__(self)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.spk_embed_dim = spk_embed_dim
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type
        self.encoder_reduction_factor = encoder_reduction_factor
        self.post_encoder_reduction_factor = post_encoder_reduction_factor
        self.decoder_reduction_factor = decoder_reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.duration_predictor_type = duration_predictor_type
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.encoder_input_layer = encoder_input_layer
        self.duration_predictor_use_encoder_outputs = (
            duration_predictor_use_encoder_outputs
        )
        self.viterbi_func = viterbi_decode
        self.stochastic_duration_predictor_noise_scale = (
            stochastic_duration_predictor_noise_scale
        )

        # define encoder
        if encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                idim=idim * encoder_reduction_factor,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=encoder_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_enc_kernel_size,
            )
        else:
            raise NotImplementedError

        # define projection layer
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define duration predictor
        if duration_predictor_type == "deterministic":
            self.duration_predictor = DurationPredictor(
                idim=adim,
                n_layers=duration_predictor_layers,
                n_chans=duration_predictor_chans,
                kernel_size=duration_predictor_kernel_size,
                dropout_rate=duration_predictor_dropout_rate,
            )
        elif duration_predictor_type == "stochastic":
            self.duration_predictor = StochasticDurationPredictor(
                channels=adim,
                kernel_size=stochastic_duration_predictor_kernel_size,
                dropout_rate=stochastic_duration_predictor_dropout_rate,
                flows=stochastic_duration_predictor_flows,
                dds_conv_layers=stochastic_duration_predictor_dds_conv_layers,
                global_channels=-1,  # not used for now
            )
        else:
            raise ValueError(
                f"Duration predictor type: {duration_predictor_type} is not supported."
            )

        # define extra projection layer
        if not self.duration_predictor_use_encoder_outputs:
            self.duration_predictor_projection = Conv2dSubsampling(
                duration_predictor_input_dim, adim, 0.0, use_pos_enc=False
            )

        # define AlignmentModule
        self.alignment_module = AlignmentModule(
            adim * post_encoder_reduction_factor, odim * decoder_reduction_factor
        )

        # define length regulator
        self.length_regulator = GaussianUpsampling()

        # define decoder
        if not decoder_type in ["diffsinger", "transformer", "conformer", "prodiff"]:
            raise ValueError(f"Decoder type: {decoder_type} is not supported.")
        if decoder_type == "diffsinger":
            self.decoder = GaussianDiffusion(
                in_dim=adim,
                out_dim=odim * decoder_reduction_factor,
                denoise_fn=DiffNet(
                    encoder_hidden_dim=adim,
                    residual_channels=diffsinger_denoiser_residual_channels,
                ),
            )
        else:
            if decoder_type == "prodiff":
                self.decoder = SpectogramDenoiser(
                    odim * decoder_reduction_factor,
                    adim=adim * post_encoder_reduction_factor,
                    layers=prodiff_denoiser_layers,
                    channels=prodiff_denoiser_channels,
                    timesteps=prodiff_diffusion_steps,
                    timescale=prodiff_diffusion_timescale,
                    max_beta=prodiff_diffusion_beta,
                    scheduler=prodiff_diffusion_scheduler,
                    cycle_length=prodiff_diffusion_cycle_ln,
                )
            else:
                # NOTE: we use encoder as decoder
                # because fastspeech's decoder is the same as encoder
                if decoder_type == "conformer":
                    self.decoder = ConformerEncoder(
                        idim=0,
                        attention_dim=adim * post_encoder_reduction_factor,
                        attention_heads=aheads,
                        linear_units=dunits,
                        num_blocks=dlayers,
                        input_layer=None,
                        dropout_rate=transformer_dec_dropout_rate,
                        positional_dropout_rate=transformer_dec_positional_dropout_rate,
                        attention_dropout_rate=transformer_dec_attn_dropout_rate,
                        normalize_before=decoder_normalize_before,
                        concat_after=decoder_concat_after,
                        positionwise_layer_type=positionwise_layer_type,
                        positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                        macaron_style=use_macaron_style_in_conformer,
                        pos_enc_layer_type=conformer_pos_enc_layer_type,
                        selfattention_layer_type=conformer_self_attn_layer_type,
                        use_cnn_module=use_cnn_in_conformer,
                        cnn_module_kernel=conformer_dec_kernel_size,
                    )
                else:
                    raise NotImplementedError

                # define final projection
                self.feat_out = torch.nn.Linear(
                    adim * post_encoder_reduction_factor,
                    odim * decoder_reduction_factor,
                )

            # only diffsinger does not have postnet
            self.postnet = (
                None
                if postnet_layers == 0
                else Postnet(
                    idim=idim,
                    odim=odim,
                    n_layers=postnet_layers,
                    n_chans=postnet_chans,
                    n_filts=postnet_filts,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=postnet_dropout_rate,
                )
            )

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: torch.Tensor = None,
        olens: torch.Tensor = None,
        dp_inputs: torch.Tensor = None,
        dplens: torch.Tensor = None,
        spembs: torch.Tensor = None,
        is_inference: bool = False,
    ):
        ret = {}

        # check encoder reduction factor
        if self.encoder_reduction_factor > 1:
            # reshape inputs if use reduction factor for encoder
            # (B, Tmax, idim) ->  (B, Tmax // r_e, idim * r_e)
            batch_size, max_length, dim = xs.shape
            if max_length % self.encoder_reduction_factor != 0:
                xs = xs[:, : -(max_length % self.encoder_reduction_factor)]
            xs = xs.contiguous().view(
                batch_size,
                max_length // self.encoder_reduction_factor,
                dim * self.encoder_reduction_factor,
            )
            ilens = ilens.new([ilen // self.encoder_reduction_factor for ilen in ilens])

        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)

        # adjust ilens if using downsampling conv2d
        if self.encoder_input_layer == "conv2d":
            ilens = ilens.new([((ilen - 2 + 1) // 2 - 2 + 1) // 2 for ilen in ilens])

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # check post encoder reduction factor
        if self.post_encoder_reduction_factor > 1:
            # reshape inputs if use reduction factor for encoder
            # (B, Tmax, idim) ->  (B, Tmax // r_e, idim * r_e)
            batch_size, max_length, dim = hs.shape
            if max_length % self.post_encoder_reduction_factor != 0:
                hs = hs[:, : -(max_length % self.post_encoder_reduction_factor)]
            hs = hs.contiguous().view(
                batch_size,
                max_length // self.post_encoder_reduction_factor,
                dim * self.post_encoder_reduction_factor,
            )
            ilens = ilens.new(
                [ilen // self.post_encoder_reduction_factor for ilen in ilens]
            )

        # interpolate duration predictor input to match that of encoder output
        if self.duration_predictor_use_encoder_outputs:
            _dp_inputs = hs
        else:
            _dp_inputs, _ = self.duration_predictor_projection(dp_inputs, None)

            # interpolate
            B, _, C = _dp_inputs.shape
            _dp_inputs_interpolated = torch.zeros(
                B, hs.shape[1], C, device=_dp_inputs.device
            )
            for i in range(_dp_inputs.shape[0]):
                elem = _dp_inputs[i].unsqueeze(0).permute(0, 2, 1)  # [1, C, T]
                _elem = (
                    F.interpolate(elem, size=hs[i].shape[0]).permute(0, 2, 1).squeeze(0)
                )
                _dp_inputs_interpolated[i] = _elem
            _dp_inputs = _dp_inputs_interpolated

        # adjust ys and olens
        if self.decoder_reduction_factor > 1 and ys is not None:
            batch_size, max_y_length, y_dim = ys.shape
            if max_y_length % self.decoder_reduction_factor != 0:
                ys = ys[:, : -(max_y_length % self.decoder_reduction_factor)]
            ys = ys.contiguous().view(
                batch_size,
                max_y_length // self.decoder_reduction_factor,
                y_dim * self.decoder_reduction_factor,
            )
            olens_reduced = olens.new(
                [olen // self.decoder_reduction_factor for olen in olens]
            )
        else:
            olens_reduced = olens

        # forward duration predictor and length regulator
        h_masks = make_pad_mask(ilens).to(hs.device)
        if is_inference:
            if ys is None:
                log_p_attn = None
                ds = None
                bin_loss = 0.0
            else:
                log_p_attn = self.alignment_module(hs, ys, h_masks)
                ds, bin_loss = self.viterbi_func(
                    log_p_attn, ilens, olens_reduced
                )

            # forward duration predictor
            if self.duration_predictor_type == "deterministic":
                d_outs = self.duration_predictor.inference(_dp_inputs, None)
            elif self.duration_predictor_type == "stochastic":
                _h_masks = make_non_pad_mask(ilens).to(_dp_inputs.device)
                d_outs = self.duration_predictor(
                    _dp_inputs.transpose(1, 2),
                    _h_masks.unsqueeze(1),
                    inverse=True,
                    noise_scale=self.stochastic_duration_predictor_noise_scale,
                ).squeeze(1)
            d_outs = torch.clamp(d_outs, max=MAX_DP_OUTPUT)
            ret["d_outs"] = d_outs

            # upsampling
            d_masks = make_non_pad_mask(ilens).to(d_outs.device)
            hs = self.length_regulator(hs, d_outs, None, d_masks)  # (B, T_feats, adim)
        else:
            # forward alignment module and obtain duration
            log_p_attn = self.alignment_module(hs, ys, h_masks)
            ds, bin_loss = self.viterbi_func(
                log_p_attn, ilens, olens_reduced
            )

            # forward duration predictor
            h_masks = make_non_pad_mask(ilens).to(hs.device)
            if self.duration_predictor_type == "deterministic":
                d_outs = self.duration_predictor(_dp_inputs, h_masks)
                d_outs = torch.clamp(d_outs, max=MAX_DP_OUTPUT)
                ret["d_outs"] = d_outs
            elif self.duration_predictor_type == "stochastic":
                dur_nll = self.duration_predictor(
                    _dp_inputs.transpose(1, 2),  # (B, T, C)
                    h_masks.unsqueeze(1),
                    w=ds.unsqueeze(1),  # (B, 1, T_text)
                )
                dur_nll = dur_nll / torch.sum(h_masks)
                ret["dur_nll"] = dur_nll

            # upsampling (expand)
            hs = self.length_regulator(
                hs,
                ds,
                make_non_pad_mask(olens_reduced).to(hs.device),
                make_non_pad_mask(ilens).to(ds.device),
            )  # (B, T_feats, adim)

        # forward decoder
        if olens is not None and not is_inference:
            h_masks = self._source_mask(olens_reduced)
        else:
            h_masks = None

        if self.decoder_type == "diffsinger":
            if is_inference:
                after_outs = self.decoder.inference(hs)
                ret["after_outs"] = after_outs
            else:
                noise, x_recon = self.decoder(hs, olens_reduced, ys)
                ret["noise"] = noise
                ret["x_recon"] = x_recon
        else:
            if self.decoder_type == "prodiff":  # no feat_out
                before_outs = self.decoder(
                    hs, ys, h_masks, is_inference
                )  # (B, T_feats, odim)
            else:
                zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
                before_outs = self.feat_out(zs).view(
                    zs.size(0), -1, self.odim
                )  # (B, Lmax, odim)

            # postnet -> (B, Lmax//r * r, odim)
            if self.postnet is None:
                after_outs = before_outs
            else:
                after_outs = before_outs + self.postnet(
                    before_outs.transpose(1, 2)
                ).transpose(1, 2)

            ret["before_outs"] = before_outs
            ret["after_outs"] = after_outs

        ret["ds"] = ds
        ret["ilens"] = ilens
        ret["bin_loss"] = bin_loss
        ret["log_p_attn"] = log_p_attn
        ret["olens_reduced"] = olens_reduced

        return ret

    def forward(
        self,
        src_speech: torch.Tensor,
        src_speech_lengths: torch.Tensor,
        tgt_speech: torch.Tensor,
        tgt_speech_lengths: torch.Tensor,
        dp_inputs: torch.Tensor = None,
        dp_lengths: torch.Tensor = None,
        spembs: torch.Tensor = None,
    ):
        """Forward propagation.

        Args:
            src_speech (Tensor): Batch of padded source features (B, Tmax, odim).
            src_speech_lengths (LongTensor): Batch of the lengths of each source (B,).
            tgt_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            tgt_speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            dp_inputs (Tensor): Batch of padded duration predictor input features (B, Tmax, dpidim).
            dp_lengths (LongTensor): Batch of the lengths of each duration prdictor input feature (B,).
            spembs (Tensor, optional): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Dict: return values (see `_forward`).

        """
        src_speech = src_speech[:, : src_speech_lengths.max()]  # for data-parallel
        tgt_speech = tgt_speech[:, : tgt_speech_lengths.max()]  # for data-parallel

        batch_size = src_speech.size(0)

        xs, ys = src_speech, tgt_speech
        ilens, olens = src_speech_lengths, tgt_speech_lengths

        # forward propagation
        ret = self._forward(
            xs,
            ilens,
            ys,
            olens,
            dp_inputs=dp_inputs,
            dplens=dp_lengths,
            spembs=spembs,
            is_inference=False,
        )

        # modifiy mod part of groundtruth
        if self.decoder_reduction_factor > 1:
            olens = olens.new(
                [olen - olen % self.decoder_reduction_factor for olen in olens]
            )
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        ret["olens"] = olens
        ret["ys"] = ys

        return ret

    def inference(
        self,
        src_speech: torch.Tensor,
        tgt_speech: torch.Tensor = None,
        spembs: torch.Tensor = None,
        dp_input: torch.Tensor = None,
        use_teacher_forcing: bool = False,
    ):
        """Forward pass during inference.

        Args:
            src_speech (Tensor): Source feature sequence (T, idim).
            tgt_speech (Tensor, optional): Target feature sequence (L, idim).
            spembs (Tensor, optional): Speaker embedding vector (spk_embed_dim,).
            dp_inputs (Tensor): Batch of padded duration predictor input features (T, dpidim).
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output predicted durations (L, ).

        """
        x, y = src_speech, tgt_speech
        spemb = spembs

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs = x.unsqueeze(0)
        dp_input = dp_input.unsqueeze(0)
        if y is not None:
            ys = y.unsqueeze(0)
            olens = torch.tensor([y.shape[0]], dtype=torch.long, device=y.device)
        else:
            ys = None
            olens = None
        if spemb is not None:
            spembs = spemb.unsqueeze(0)

        if use_teacher_forcing:
            # use groundtruth duration
            ds = d.unsqueeze(0)
            ret = self._forward(
                xs,
                ilens,
                ds=ds,
                spembs=spembs,
                dp_inputs=dp_input,
            )  # (1, L, odim)
            outs = ret["after_outs"]
        else:
            # inference
            ret = self._forward(
                xs,
                ilens,
                ys=ys,
                olens=olens,
                spembs=spembs,
                dp_inputs=dp_input,
                is_inference=True,
            )  # (1, L, odim)
            outs = ret["after_outs"]
            ds = ret["ds"]
            d_outs = ret["d_outs"]
            ilens_ = ret["ilens"]
            log_p_attn = ret["log_p_attn"]

        # inference without gt
        if ds is None and log_p_attn is None:
            return outs[0], d_outs[0]
        # inference with gt (debug usage)
        else:
            return outs[0], d_outs[0], ds[0], log_p_attn[0], ilens_[0]

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).

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

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)
