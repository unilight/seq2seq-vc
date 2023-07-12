import logging
import torch

from seq2seq_vc.layers.positional_encoding import ScaledPositionalEncoding
from seq2seq_vc.modules.transformer.encoder import Encoder as TransformerEncoder
from seq2seq_vc.modules.conformer.encoder import Encoder as ConformerEncoder
from seq2seq_vc.modules.transformer.decoder import Decoder
from seq2seq_vc.modules.pre_postnets import Prenet, Postnet
from seq2seq_vc.modules.transformer.mask import subsequent_mask
from seq2seq_vc.layers.utils import make_non_pad_mask
from seq2seq_vc.modules.transformer.attention import MultiHeadedAttention


class VTN(torch.nn.Module):
    def __init__(
        self,
        idim,
        odim,
        dprenet_layers=2,
        dprenet_units=256,
        adim=384,
        aheads=4,
        encoder_type="transformer",
        decoder_type="transformer",
        elayers=6,
        eunits=1536,
        dlayers=6,
        dunits=1536,
        postnet_layers=5,
        postnet_filts=5,
        postnet_chans=256,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        dprenet_dropout_rate=0.5,
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        use_batch_norm=True,
        encoder_normalize_before=True,
        decoder_normalize_before=False,
        encoder_concat_after=False,
        decoder_concat_after=False,
        decoder_reduction_factor=2,
        spk_embed_dim=None,
        spk_embed_integration_type="add",
        # only for scaled pos encoding
        initial_encoder_alpha=1.0,
        initial_decoder_alpha=1.0,
        # guided attention loss related
        use_guided_attn_loss=False,
        num_heads_applied_guided_attn=2,
        num_layers_applied_guided_attn=2,
        # only for conformer
        conformer_rel_pos_type: str = "legacy",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        zero_triu: bool = False,
        conformer_enc_kernel_size: int = 7,
        conformer_dec_kernel_size: int = 31,
    ):
        # initialize base classes
        torch.nn.Module.__init__(self)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.spk_embed_dim = spk_embed_dim
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type
        self.decoder_reduction_factor = decoder_reduction_factor
        self.use_guided_attn_loss = use_guided_attn_loss
        self.num_heads_applied_guided_attn = num_heads_applied_guided_attn
        self.num_layers_applied_guided_attn = num_layers_applied_guided_attn
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        # use idx 0 as padding idx
        padding_idx = 0

        # check relative positional encoding compatibility
        if encoder_type == "conformer":
            if conformer_rel_pos_type == "legacy":
                if conformer_pos_enc_layer_type == "rel_pos":
                    conformer_pos_enc_layer_type = "legacy_rel_pos"
                    logging.warning(
                        "Fallback to conformer_pos_enc_layer_type = 'legacy_rel_pos' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
                if conformer_self_attn_layer_type == "rel_selfattn":
                    conformer_self_attn_layer_type = "legacy_rel_selfattn"
                    logging.warning(
                        "Fallback to "
                        "conformer_self_attn_layer_type = 'legacy_rel_selfattn' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
            elif conformer_rel_pos_type == "latest":
                assert conformer_pos_enc_layer_type != "legacy_rel_pos"
                assert conformer_self_attn_layer_type != "legacy_rel_selfattn"
            else:
                raise ValueError(f"Unknown rel_pos_type: {conformer_rel_pos_type}")

        # define encoder
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer="conv2d-scaled-pos-enc",
                pos_enc_class=ScaledPositionalEncoding,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,  # V
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,  # V
                dropout_rate=transformer_enc_dropout_rate,
            )
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer="conv2d",
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,  # V
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,  # V
                # the following args are unique to Conformer
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_enc_kernel_size,
                zero_triu=zero_triu,
            )
        else:
            raise NotImplementedError

        # define projection layer
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)
        # define decoder prenet
        decoder_input_layer = torch.nn.Sequential(
            Prenet(
                idim=odim,
                n_layers=dprenet_layers,
                n_units=dprenet_units,
                dropout_rate=dprenet_dropout_rate,
            ),
            torch.nn.Linear(dprenet_units, adim),
        )

        # define decoder
        # the reason why odim=-1 is because the final output dimention is decided by feat_out
        self.decoder = Decoder(
            odim=-1,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            input_layer=decoder_input_layer,
            use_output_layer=False,
            pos_enc_class=ScaledPositionalEncoding,
            normalize_before=decoder_normalize_before,
            concat_after=decoder_concat_after,
        )

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * decoder_reduction_factor)
        self.prob_out = torch.nn.Linear(adim, decoder_reduction_factor)

        # define postnet
        self.postnet = Postnet(
            idim=idim,
            odim=odim,
            n_layers=postnet_layers,
            n_chans=postnet_chans,
            n_filts=postnet_filts,
            use_batch_norm=use_batch_norm,
        )

        # initialize parameters
        self._reset_parameters(
            init_enc_alpha=initial_encoder_alpha,
            init_dec_alpha=initial_decoder_alpha,
        )

    def _reset_parameters(self, init_enc_alpha: float, init_dec_alpha: float):
        # initialize alpha in scaled positional encoding
        if self.encoder_type == "transformer":
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
        if self.decoder_type == "transformer":
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

    def forward(self, xs, ilens, ys, labels, olens, spembs=None, *args, **kwargs):
        max_ilen = max(ilens)
        max_olen = max(olens)
        if max_ilen != xs.shape[1]:
            xs = xs[:, :max_ilen]
        if max_olen != ys.shape[1]:
            ys = ys[:, :max_olen]
            labels = labels[:, :max_olen]

        # forward encoder
        x_masks = self._source_mask(ilens).to(xs.device)
        hs, hs_masks = self.encoder(xs, x_masks)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs_int = self._integrate_with_spk_embed(hs, spembs)
        else:
            hs_int = hs

        # thin out frames for reduction factor (B, Lmax, odim) ->  (B, Lmax//r, odim)
        if self.decoder_reduction_factor > 1:
            ys_in = ys[
                :, self.decoder_reduction_factor - 1 :: self.decoder_reduction_factor
            ]
            olens_in = olens.new(
                [
                    torch.div(
                        olen, self.decoder_reduction_factor, rounding_mode="floor"
                    )
                    for olen in olens
                ]
            )
        else:
            ys_in, olens_in = ys, olens

        # add first zero frame and remove last frame for auto-regressive
        ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

        # forward decoder
        y_masks = self._target_mask(olens_in).to(xs.device)
        zs, _ = self.decoder(ys_in, y_masks, hs_int, hs_masks)
        # (B, Lmax//r, odim * r) -> (B, Lmax//r * r, odim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        # (B, Lmax//r, r) -> (B, Lmax//r * r)
        logits = self.prob_out(zs).view(zs.size(0), -1)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        # modifiy mod part of groundtruth
        if self.decoder_reduction_factor > 1:
            assert olens.ge(
                self.decoder_reduction_factor
            ).all(), "Output length must be greater than or equal to reduction factor."
            olens = olens.new(
                [olen - olen % self.decoder_reduction_factor for olen in olens]
            )
            max_olen = max(olens)
            ys = ys[:, :max_olen]
            labels = labels[:, :max_olen]
            labels = torch.scatter(
                labels, 1, (olens - 1).unsqueeze(1), 1.0
            )  # see #3388

        # calculate guided attention loss
        att_ws = []
        # modify mask because of conv2d. Use ceiling division here
        ilens_ds_st = ilens.new([((ilen - 2 + 1) // 2 - 2 + 1) // 2 for ilen in ilens])
        for idx, layer_idx in enumerate(reversed(range(len(self.decoder.decoders)))):
            att_ws += [
                self.decoder.decoders[layer_idx].src_attn.attn[
                    # :, : self.num_heads_applied_guided_attn
                    :,
                    :,
                ]
            ]
            # if idx + 1 == self.num_layers_applied_guided_attn:
            # break
        # att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_out, T_in)

        return (
            after_outs,
            before_outs,
            logits,
            ys,
            labels,
            olens,
            (att_ws, ilens_ds_st, olens_in),
        )

    def inference(self, x, inference_args, spemb=None, *args, **kwargs):
        """Generate the sequence of features given the sequences of acoustic features.

        Args:
            x (Tensor): Input sequence of acoustic features (T, idim).
            inference_args (dict):
                - threshold (float): Threshold in inference.
                - minlenratio (float): Minimum length ratio in inference.
                - maxlenratio (float): Maximum length ratio in inference.
            spemb (Tensor, optional): Speaker embedding vector (spk_embed_dim).

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Encoder-decoder (source) attention weights (#layers, #heads, L, T).

        """
        # get options
        threshold = inference_args["threshold"]
        minlenratio = inference_args["minlenratio"]
        maxlenratio = inference_args["maxlenratio"]

        # forward encoder
        x = x.unsqueeze(0)
        hs, _ = self.encoder(x, None)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            spembs = spemb.unsqueeze(0)
            hs = self._integrate_with_spk_embed(hs, spembs)

        # set limits of length
        maxlen = int(hs.size(1) * maxlenratio / self.decoder_reduction_factor)
        minlen = int(hs.size(1) * minlenratio / self.decoder_reduction_factor)

        # initialize
        idx = 0
        ys = hs.new_zeros(1, 1, self.odim)
        outs, probs = [], []

        # forward decoder step-by-step
        z_cache = None
        while True:
            # update index
            idx += 1

            # calculate output and stop prob at idx-th step
            y_masks = subsequent_mask(idx).unsqueeze(0).to(x.device)
            z, z_cache = self.decoder.forward_one_step(
                ys, y_masks, hs, cache=z_cache
            )  # (B, adim)
            outs += [
                self.feat_out(z).view(self.decoder_reduction_factor, self.odim)
            ]  # [(r, odim), ...]
            probs += [torch.sigmoid(self.prob_out(z))[0]]  # [(r), ...]

            # update next inputs
            ys = torch.cat(
                (ys, outs[-1][-1].view(1, 1, self.odim)), dim=1
            )  # (1, idx + 1, odim)

            # get attention weights
            att_ws_ = []
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention) and "src" in name:
                    att_ws_ += [m.attn[0, :, -1].unsqueeze(1)]  # [(#heads, 1, T),...]
            if idx == 1:
                att_ws = att_ws_
            else:
                # [(#heads, l, T), ...]
                att_ws = [
                    torch.cat([att_w, att_w_], dim=1)
                    for att_w, att_w_ in zip(att_ws, att_ws_)
                ]

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = (
                    torch.cat(outs, dim=0).unsqueeze(0).transpose(1, 2)
                )  # (L, odim) -> (1, L, odim) -> (1, odim, L)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                break

        # concatenate attention weights -> (#layers, #heads, L, T)
        att_ws = torch.stack(att_ws, dim=0)

        return outs, probs, att_ws

    def calculate_all_attentions(
        self,
        xs,
        ilens,
        ys,
        olens,
        spembs=None,
        skip_output=False,
        keep_tensor=False,
        *args,
        **kwargs,
    ):
        """Calculate all of the attention weights.

        Args:
            xs (Tensor): Batch of padded acoustic features (B, Tmax, idim).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors
                (B, spk_embed_dim).
            skip_output (bool, optional): Whether to skip calculate the final output.
            keep_tensor (bool, optional): Whether to keep original tensor.

        Returns:
            dict: Dict of attention weights and outputs.

        """
        with torch.no_grad():
            # thin out input frames for reduction factor
            # (B, Lmax, idim) ->  (B, Lmax // r, idim * r)
            if self.encoder_reduction_factor > 1:
                B, Lmax, idim = xs.shape
                if Lmax % self.encoder_reduction_factor != 0:
                    xs = xs[:, : -(Lmax % self.encoder_reduction_factor), :]
                xs_ds = xs.contiguous().view(
                    B,
                    int(Lmax / self.encoder_reduction_factor),
                    idim * self.encoder_reduction_factor,
                )
                ilens_ds = ilens.new(
                    [ilen // self.encoder_reduction_factor for ilen in ilens]
                )
            else:
                xs_ds, ilens_ds = xs, ilens

            # forward encoder
            x_masks = self._source_mask(ilens_ds).to(xs.device)
            hs, hs_masks = self.encoder(xs_ds, x_masks)

            # integrate speaker embedding
            if self.spk_embed_dim is not None:
                hs = self._integrate_with_spk_embed(hs, spembs)

            # thin out frames for reduction factor
            # (B, Lmax, odim) ->  (B, Lmax//r, odim)
            if self.reduction_factor > 1:
                ys_in = ys[:, self.reduction_factor - 1 :: self.reduction_factor]
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                ys_in, olens_in = ys, olens

            # add first zero frame and remove last frame for auto-regressive
            ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

            # forward decoder
            y_masks = self._target_mask(olens_in).to(xs.device)
            zs, _ = self.decoder(ys_in, y_masks, hs, hs_masks)

            # calculate final outputs
            if not skip_output:
                before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
                if self.postnet is None:
                    after_outs = before_outs
                else:
                    after_outs = before_outs + self.postnet(
                        before_outs.transpose(1, 2)
                    ).transpose(1, 2)

        # modifiy mod part of output lengths due to reduction factor > 1
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])

        # store into dict
        att_ws_dict = dict()
        if keep_tensor:
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention):
                    att_ws_dict[name] = m.attn
            if not skip_output:
                att_ws_dict["before_postnet_fbank"] = before_outs
                att_ws_dict["after_postnet_fbank"] = after_outs
        else:
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention):
                    attn = m.attn.cpu().numpy()
                    if "encoder" in name:
                        attn = [a[:, :l, :l] for a, l in zip(attn, ilens.tolist())]
                    elif "decoder" in name:
                        if "src" in name:
                            attn = [
                                a[:, :ol, :il]
                                for a, il, ol in zip(
                                    attn, ilens.tolist(), olens_in.tolist()
                                )
                            ]
                        elif "self" in name:
                            attn = [
                                a[:, :l, :l] for a, l in zip(attn, olens_in.tolist())
                            ]
                        else:
                            logging.warning("unknown attention module: " + name)
                    else:
                        logging.warning("unknown attention module: " + name)
                    att_ws_dict[name] = attn
            if not skip_output:
                before_outs = before_outs.cpu().numpy()
                after_outs = after_outs.cpu().numpy()
                att_ws_dict["before_postnet_fbank"] = [
                    m[:l].T for m, l in zip(before_outs, olens.tolist())
                ]
                att_ws_dict["after_postnet_fbank"] = [
                    m[:l].T for m, l in zip(after_outs, olens.tolist())
                ]

        return att_ws_dict

    def _add_first_frame_and_remove_last_frame(self, ys):
        ys_in = torch.cat(
            [ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1
        )
        return ys_in

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

    def _source_mask(self, ilens):
        """Make masks for self-attention.

        Args:
            ilens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                    [[1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens)
        return x_masks.unsqueeze(-2)

    def _target_mask(self, olens):
        """Make masks for masked self-attention.

        Args:
            olens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for masked self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> olens = [5, 3]
            >>> self._target_mask(olens)
            tensor([[[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1]],
                    [[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        y_masks = make_non_pad_mask(olens)
        s_masks = subsequent_mask(y_masks.size(-1), device=y_masks.device).unsqueeze(0)
        return y_masks.unsqueeze(-2) & s_masks
