import torch
from torch import nn
from transformers import GPT2Config, LogitsProcessorList, TypicalLogitsWarper

from mini_tortoise_tts.gpt.gpt2_model_null_embeddings import build_hf_gpt_transformer
from mini_tortoise_tts.gpt.gpt_2_inference_model import GPT2InferenceModel
from mini_tortoise_tts.torch_modules.conditioning_encoder import ConditioningEncoder
from mini_tortoise_tts.torch_modules.mel_encoder import MelEncoder


class UnifiedVoice(nn.Module):
    __slots__ = (
        "number_text_tokens",
        "start_text_token",
        "stop_text_token",
        "number_mel_codes",
        "start_mel_token",
        "stop_mel_token",
        "layers",
        "heads",
        "max_mel_tokens",
        "max_text_tokens",
        "model_dim",
        "max_conditioning_inputs",
        "mel_length_compression",
        "conditioning_encoder",
        "text_embedding",
        "mel_embedding",
        "gpt",
        "mel_pos_embedding",
        "text_pos_embedding",
        "mel_layer_pos_embedding",
        "text_layer_pos_embedding",
        "mel_solo_embedding",
        "text_solo_embedding",
        "final_norm",
        "text_head",
        "mel_head",
    )

    def __init__(
        self,
        layers: int = 8,
        model_dim: int = 512,
        heads: int = 8,
        max_text_tokens: int = 120,
        max_mel_tokens: int = 250,
        max_conditioning_inputs: int = 1,
        mel_length_compression: int = 1024,
        number_text_tokens: int = 256,
        start_text_token: int | None = None,
        number_mel_codes: int = 8194,
        start_mel_token: int = 8192,
        stop_mel_token: int = 8193,
        train_solo_embeddings: bool = False,
        use_mel_codes_as_input: bool = True,
        checkpointing: bool = True,
        types: int = 1,
    ):
        super().__init__()

        self.number_text_tokens = number_text_tokens
        self.start_text_token = number_text_tokens * types if start_text_token is None else start_text_token
        self.stop_text_token = 0
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=heads)
        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        if use_mel_codes_as_input:
            self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        else:
            self.mel_embedding = MelEncoder(model_dim, resblocks_per_reduction=1)

        self.mel_layer_pos_embedding = None  # TODO: is this ever used?
        self.text_layer_pos_embedding = None  # TODO: is this ever used?
        (
            self.gpt,
            self.mel_pos_embedding,
            self.text_pos_embedding,
        ) = build_hf_gpt_transformer(
            layers,
            model_dim,
            heads,
            self.max_mel_tokens + 2 + self.max_conditioning_inputs,
            self.max_text_tokens + 2,
            checkpointing,
        )
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        self.inference_model = GPT2InferenceModel(
            GPT2Config(
                vocab_size=self.max_mel_tokens,
                n_positions=seq_length,
                n_ctx=seq_length,
                n_embd=self.model_dim,
                n_layer=self.layers,
                n_head=self.heads,
                gradient_checkpointing=False,
                use_cache=True,
            ),
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )

        if use_deepspeed and half and torch.cuda.is_available():
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                model=self.inference_model, mp_size=1, replace_with_kernel_inject=True, dtype=torch.float16
            )
            self.inference_model = self.ds_engine.module.eval()
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                model=self.inference_model, mp_size=1, replace_with_kernel_inject=True, dtype=torch.float32
            )
            self.inference_model = self.ds_engine.module.eval()
        else:
            self.inference_model = self.inference_model.eval()

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = nn.functional.pad(input, (1, 0), value=start_token)
        tar = nn.functional.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, wav_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        mel_lengths = torch.div(wav_lengths, self.mel_length_compression, rounding_mode="trunc")
        for b in range(len(mel_lengths)):
            actual_end = (
                mel_lengths[b] + 1
            )  # Due to the convolutional nature of how these tokens are generated, it would be best if the model predicts a token past the actual last token.
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def get_logits(
        self,
        speech_conditioning_inputs,
        first_inputs,
        first_head,
        second_inputs=None,
        second_head=None,
        get_attns=False,
        return_latent=False,
    ):
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions

        enc = gpt_out.last_hidden_state[:, 1:]  # The first logit is tied to the speech_conditioning_input
        enc = self.final_norm(enc)

        if return_latent:
            return (
                enc[
                    :, speech_conditioning_inputs.shape[1] : speech_conditioning_inputs.shape[1] + first_inputs.shape[1]
                ],
                enc[:, -second_inputs.shape[1] :],
            )

        first_logits = enc[:, : first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1] :]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input):
        speech_conditioning_input = (
            speech_conditioning_input.unsqueeze(1)
            if len(speech_conditioning_input.shape) == 3
            else speech_conditioning_input
        )
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        conds = conds.mean(dim=1)
        return conds

    def forward(
        self,
        speech_conditioning_latent,
        text_inputs,
        text_lengths,
        mel_codes,
        wav_lengths,
        types=None,
        text_first=True,
        raw_mels=None,
        return_attentions=False,
        return_latent=False,
        clip_inputs=True,
    ):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        speech_conditioning_input: MEL float tensor, (b,1024)
        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        raw_mels: MEL float tensor (b,80,s)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        If clip_inputs is True, the inputs will be clipped to the smallest input size across each input modality.
        """
        # Types are expressed by expanding the text embedding space.
        if types is not None:
            text_inputs = text_inputs * (1 + types).unsqueeze(-1)

        if clip_inputs:
            # This model will receive micro-batches with a ton of padding for both the text and MELs. Ameliorate this by
            # chopping the inputs by the maximum actual length.
            max_text_len = text_lengths.max()
            text_inputs = text_inputs[:, :max_text_len]
            max_mel_len = wav_lengths.max() // self.mel_length_compression
            mel_codes = mel_codes[:, :max_mel_len]
            if raw_mels is not None:
                raw_mels = raw_mels[:, :, : max_mel_len * 4]
        mel_codes = self.set_mel_padding(mel_codes, wav_lengths)
        text_inputs = nn.functional.pad(text_inputs, (0, 1), value=self.stop_text_token)
        mel_codes = nn.functional.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        conds = speech_conditioning_latent.unsqueeze(1)
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(
            mel_codes, self.start_mel_token, self.stop_mel_token
        )
        if raw_mels is not None:
            mel_inp = nn.functional.pad(raw_mels, (0, 8))
        else:
            mel_inp = mel_codes
        mel_emb = self.mel_embedding(mel_inp)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        if text_first:
            text_logits, mel_logits = self.get_logits(
                conds,
                text_emb,
                self.text_head,
                mel_emb,
                self.mel_head,
                get_attns=return_attentions,
                return_latent=return_latent,
            )
            if return_latent:
                return mel_logits[
                    :, :-2
                ]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.
        else:
            mel_logits, text_logits = self.get_logits(
                conds,
                mel_emb,
                self.mel_head,
                text_emb,
                self.text_head,
                get_attns=return_attentions,
                return_latent=return_latent,
            )
            if return_latent:
                return text_logits[
                    :, :-2
                ]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

        if return_attentions:
            return mel_logits
        loss_text = nn.functional.cross_entropy(text_logits, text_targets.long())
        loss_mel = nn.functional.cross_entropy(mel_logits, mel_targets.long())
        return loss_text.mean(), loss_mel.mean(), mel_logits

    def compute_embeddings(
        self,
        cond_latents,
        text_inputs,
    ):
        text_inputs = nn.functional.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = nn.functional.pad(text_inputs, (1, 0), value=self.start_text_token)
        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        conds = cond_latents.unsqueeze(1)
        emb = torch.cat([conds, emb], dim=1)
        self.inference_model.store_mel_emb(emb)
        gpt_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        gpt_inputs[:, -1] = self.start_mel_token
        return gpt_inputs

    def inference_speech(
        self,
        speech_conditioning_latent,
        text_inputs,
        input_tokens=None,
        num_return_sequences=1,
        max_generate_length=None,
        typical_sampling=False,
        typical_mass=0.9,
        **hf_generate_kwargs
    ):

        text_inputs = nn.functional.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs, _ = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        conds = speech_conditioning_latent.unsqueeze(1)
        emb = torch.cat([conds, text_emb], dim=1)
        self.inference_model.store_mel_emb(emb)

        fake_inputs = torch.full(
            (
                emb.shape[0],
                conds.shape[1] + emb.shape[1],
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        fake_inputs[:, -1] = self.start_mel_token
        trunc_index = fake_inputs.shape[1]
        if input_tokens is None:
            inputs = fake_inputs
        else:
            assert (
                num_return_sequences % input_tokens.shape[0] == 0
            ), "The number of return sequences must be divisible by the number of input sequences"
            fake_inputs = fake_inputs.repeat(num_return_sequences, 1)
            input_tokens = input_tokens.repeat(num_return_sequences // input_tokens.shape[0], 1)
            inputs = torch.cat([fake_inputs, input_tokens], dim=1)

        logits_processor = (
            LogitsProcessorList([TypicalLogitsWarper(mass=typical_mass)]) if typical_sampling else LogitsProcessorList()
        )
        max_length = (
            trunc_index + self.max_mel_tokens - 1 if max_generate_length is None else trunc_index + max_generate_length
        )
        gen = self.inference_model.generate(
            inputs,
            bos_token_id=self.start_mel_token,
            pad_token_id=self.stop_mel_token,
            eos_token_id=self.stop_mel_token,
            max_length=max_length,
            logits_processor=logits_processor,
            num_return_sequences=num_return_sequences,
            **hf_generate_kwargs,
        )
        return gen[:, trunc_index:]

    def get_generator(self, fake_inputs, **hf_generate_kwargs):
        return self.inference_model.generate_stream(
            fake_inputs,
            bos_token_id=self.start_mel_token,
            pad_token_id=self.stop_mel_token,
            eos_token_id=self.stop_mel_token,
            max_length=500,
            do_stream=True,
            **hf_generate_kwargs,
        )
