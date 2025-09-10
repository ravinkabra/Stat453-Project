from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.model import BaseModel
from .config import OldPtM2ATransformerConfig
from ...dataset.old_pt.model_input import OldPtModelInput

# Constants
TRAIN_LENGTH = 192
MAX_STEPS = 1000000

# Indicator: 0
# pitch+duration*2: 3200 (25*128)
N_NORMAL_TOKENS = 3202
N_TOKENS = N_NORMAL_TOKENS + 3
SOS_TOKEN = N_NORMAL_TOKENS
EOS_TOKEN = N_NORMAL_TOKENS + 1
PAD_TOKEN = N_NORMAL_TOKENS + 2


def fill_with_neg_inf(t: torch.Tensor) -> torch.Tensor:
    """FP16 兼容的工具函数：将张量填充为 -inf。

    这个函数用于构造 attention mask（未来位置填充为 -inf），
    在混合精度下先转换为 float 再填充，然后转换回原始类型以保证数值稳定性。
    输入: 任意 shape 的张量 t，通常是 zeros([L, L])；返回: 相同 shape、类型的 -inf 填充值张量。
    """
    return t.float().fill_(float("-inf")).type_as(t)


class OldPtM2ATransformer(BaseModel):
    """基于 RoFormer 的分层局部/全局编码器模型（旧版 Pt 变体）。

    设计要点：
    - 使用局部 encoder/decoder 对每个子序列（subseq）进行编码/解码。
    - 使用全局 encoder 将局部编码的表示按时间序列合并并进行跨子序列的上下文建模。
    - 提供采样、预处理、损失计算等工具函数以便训练和推理。
    """

    def __init__(self, config: OldPtM2ATransformerConfig):
        super().__init__(config)
        model_schema = config

        # Pull RoFormerConfig instances from the three network param fields.
        global_params = model_schema.global_network
        local_enc_params = model_schema.local_encoder_network
        local_dec_params = model_schema.local_decoder_network

        # Expose a few convenient attributes used elsewhere in the model implementation.
        self.hidden_size = global_params.config.hidden_size
        # HF config uses `num_hidden_layers` for layer count
        # Lazy import of transformers RoFormer to avoid heavy import at module load
        from transformers.models.roformer.modeling_roformer import (
            RoFormerEncoder,
        )

        # Use the provided RoFormerConfig instances from the config params. These
        # were supplied via `CustomizedRoFormerEncoderParams.config`.
        main_roformer_config = global_params.config
        self.model = RoFormerEncoder(main_roformer_config)
        local_encoder_config = local_enc_params.config
        local_decoder_config = local_dec_params.config

        self.local_embedding = nn.Embedding(N_TOKENS, self.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        with torch.no_grad():
            self.token_type_embeddings.weight.mul_(2.0)

        self.local_encoder = RoFormerEncoder(local_encoder_config)
        self.local_decoder = RoFormerEncoder(local_decoder_config)
        self.final_decoder = nn.Linear(self.hidden_size, N_TOKENS)
        self.global_sos = nn.Parameter(torch.randn(self.hidden_size))
        self._future_mask = torch.empty(0)

    # --- Small helpers to improve readability ---
    def _prepare_local_inputs(self, x: torch.LongTensor, token_type_ids: torch.LongTensor):
        """Flatten batch/seq to (batch*seq, subseq), prepend SOS, build mask and embeddings.

        Returns (mask, emb, batch_size, seq_len)
        """
        batch_size, seq_len, subseq_len = x.shape
        x_flat = x.view(-1, subseq_len)
        x_flat = torch.cat(
            [torch.full((x_flat.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=x_flat.device), x_flat],
            dim=-1,
        )
        mask = x_flat != PAD_TOKEN
        word_emb = self.local_embedding(x_flat)
        type_emb = token_type_ids.view(batch_size * seq_len, word_emb.shape[1], -1)
        type_emb = self.token_type_embeddings(type_emb)
        emb = word_emb + type_emb
        return mask, emb, batch_size, seq_len

    def _process_triplet_tensor(self, t: torch.LongTensor, pitch_shift: torch.LongTensor, type_value: int):
        """Convert [batch,seq,subseq] triplet representation into model token ids.

        `type_value` is 0 (input) or 1 (target). Returns shape [batch, seq, subseq//3*2].
        """
        batch_size, seq_length, subseq_length = t.shape
        t = t.long().view(batch_size, seq_length, subseq_length // 3, 3)
        out = torch.zeros(
            batch_size,
            seq_length,
            subseq_length // 3,
            2,
            dtype=torch.long,
            device=t.device,
        )
        pad_indices = t[:, :, :, 1] == 255
        eos_indices = t[:, :, :, 0] == 254
        is_not_drum = t[:, :, :, 0] != 127
        out[:, :, :, 0] = type_value
        out[:, :, :, 1] = t[:, :, :, 1] + (t[:, :, :, 2]) * 128 + 2 + pitch_shift[:, None, None] * is_not_drum
        out[pad_indices] = PAD_TOKEN
        out[:, :, :, 0][eos_indices] = EOS_TOKEN
        return out.view(batch_size, seq_length, subseq_length // 3 * 2)

    def local_encode(self, x: torch.LongTensor, token_type_ids: torch.LongTensor):
        """对每个子序列执行局部编码。

        输入:
            - x: LongTensor, 形状 [batch, seq, subseq]
            - token_type_ids: 与 x 对齐的 token type id，用于区分 frame 类型

        处理流程:
            1. 将子序列展平为 (batch*seq, subseq)
            2. 在每个子序列前附加一个 SOS_TOKEN，形成 decoder 输入对齐
            3. 计算 mask（非 PAD 的位置为 True）并通过 embedding + type embedding 得到输入表示
            4. 通过局部 encoder 得到隐藏表示 h

        返回:
            - h[:, 0]: 每个子序列的聚合表示（对应 prepended SOS 的位置）
            - emb[:, :-1]: 用于解码器的 token embedding（去掉最后一个 token，以便与 h 对齐）
        """
        mask, emb, batch_size, seq_len = self._prepare_local_inputs(x, token_type_ids)
        h = self.local_encoder(emb, encoder_attention_mask=mask)[0]
        return h[:, 0], emb[:, :-1]

    def local_decode(self, h: torch.Tensor, emb: torch.Tensor):
        """局部解码器：将局部隐藏表示与 embeddings 拼接并解码为 token logits。

        - h: 局部聚合表示，形状 [batch*seq, hidden]
        - emb: 局部 embeddings，形状 [batch*seq, subseq_len, hidden]
        返回 logits，形状与解码器输出对应（用于后续 softmax/采样）。
        """
        batch_size, subseq_len, _ = emb.shape
        head = h.view(batch_size, 1, -1)
        dec_input = torch.cat([head, emb[:, 1:]], dim=1)
        hidden = self.local_decoder(dec_input, attention_mask=self.buffered_future_mask(dec_input))[0]
        return self.final_decoder(hidden)

    def local_sampling(self, h: torch.Tensor, max_subseq_len: int = 32, temperature: float = 1.0):
        """基于局部 decoder 进行自回归采样。

        - h: 局部聚合向量，作为采样初始上下文
        - temperature: 采样温度，0 表示贪心（取 argmax）
        过程：迭代地通过 local_decoder 预测下一个 token，直到所有样本触发 EOS 或达到最大长度。
        返回采样到的 token id 序列（不含 SOS）。
        """
        batch_size, _ = h.shape
        y = torch.zeros((batch_size, 0), dtype=torch.long, device=h.device)
        emb = h[:, None, :]
        eos_triggered = torch.zeros(batch_size, dtype=torch.bool, device=h.device)

        for _ in range(max_subseq_len):
            dec_h = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
            logits = self.final_decoder(dec_h)
            if temperature == 0:
                next_probs = F.one_hot(logits.argmax(dim=-1), N_TOKENS).float()
            else:
                next_probs = F.softmax(logits[:, -1] / temperature, dim=-1)

            y_next = torch.multinomial(next_probs, 1)
            y_next[eos_triggered, :] = PAD_TOKEN
            eos_triggered = eos_triggered | (y_next.squeeze(1) == EOS_TOKEN)
            y = torch.cat([y, y_next], dim=1)
            if torch.all(eos_triggered):
                break

            emb = torch.cat(
                [emb, self.local_embedding(y_next) + self.token_type_embeddings(torch.ones_like(y_next))], dim=1
            )

        return y

    def buffered_future_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """生成（或重用） causal future mask，用于自回归 attention。

        为效率考虑，mask 被缓存在 self._future_mask 中，并在需要时扩展或迁移到对应设备。
        返回形状为 [dim, dim] 的上三角 -inf mask（主对角线及之前为 0，之后为 -inf）。
        """
        dim = tensor.size(1)
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            # 构造上三角矩阵并用 -inf 填充未来位置
            self._future_mask = torch.triu(fill_with_neg_inf(torch.zeros([dim, dim])), 1)
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def _generate_local_and_update_h(
        self, h: torch.Tensor, subseq_len: int, temperature: float = 1.0, token_type_one: bool = True
    ):
        """Helper: run global model to get local context, sample local tokens, and append their encoding to h.

        Returns (y_next, new_h) where:
        - y_next: [B, L] generated token ids for the local subsequence
        - new_h: the updated global-level tensor h with the new local summary appended (shape [B,1,H])

        token_type_one controls whether the token_type_ids passed to local_encode are ones or zeros
        (the codebase uses ones for accompaniment frames and zeros for melody frames).
        """
        # Run the global encoder to obtain the next-step hidden for the last position
        h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
        # Sample a local subsequence from the local decoder using the last global hidden
        y_next = self.local_sampling(h_out[:, -1], max_subseq_len=subseq_len, temperature=temperature)

        # Build token_type_ids expected by local_encode: shape [B, 1, L+1]
        b, s, L = y_next.unsqueeze(1).shape
        if token_type_one:
            token_type_ids = torch.ones((b, s, L + 1), dtype=torch.long, device=y_next.device)
        else:
            token_type_ids = torch.zeros((b, s, L + 1), dtype=torch.long, device=y_next.device)

        # Encode the newly sampled local tokens to get their global summary and append to h
        local_h = self.local_encode(y_next.unsqueeze(1), token_type_ids=token_type_ids)[0].unsqueeze(1)
        h = torch.cat([h, local_h], dim=1)
        return y_next, h

    def forward(self, x: torch.LongTensor):
        """主前向函数：执行局部编码 -> 合并 -> 全局编码 -> 局部解码 的流程。

        输入 x 的形状为 [batch, seq, subseq]：seq 是帧序列数量，subseq 是每帧内部的 token 数。
        处理要点：
            1. 构造 token_type_ids 来标识帧类型（偶数帧/奇数帧）以供 local encoder 使用
            2. 调用 local_encode 得到每个子序列的聚合向量 h 与用于解码的 emb
            3. 将所有子序列的聚合向量按时间拼接并在前面加入 global SOS，作为全局 encoder 的输入
            4. 通过全局 encoder 建模跨子序列的上下文（使用 future mask 保证自回归顺序）
            5. 调用 local_decode 对每个子序列进行解码，得到最终 logits
        返回值为解码器输出 logits（未经过 softmax）。
        """
        # x: [batch, seq, subseq]
        batch_size, seq_len, subseq_len = x.shape
        assert seq_len % 2 == 0, "Expected even number of frames (2*S interleaved)."

        # 构造 frame 类型（用于 token_type embedding）: 偶数帧/奇数帧交替
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2 == 0).long()
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)

        # 局部编码
        h, emb = self.local_encode(x, token_type_ids)
        h = h.view(batch_size, seq_len, -1)

        # 在全局序列前加入 SOS 并把前一时刻的隐藏拼接进来（自回归偏移）
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)

        # 全局 encoder（使用 buffered_future_mask 保证自回归）
        h = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
        return self.local_decode(h, emb)

    def preprocess(
        self,
        x: torch.LongTensor,
        pitch_shift: torch.LongTensor,
        y: Optional[torch.LongTensor] = None,
    ):
        """将原始三元组表示（type,pitch,duration）转换为模型输入格式。
        shape: [batch,seq,subseq] -parse-> [batch,seq,subseq//3,3] -process-> [batch,seq,subseq//3*2]

        - 输入 x 的最后一维为 3（type, pitch, duration），函数将其重塑并合成两个 token 的表示。
        - 将 pad / eos / drum 的特殊值转为 PAD_TOKEN / EOS_TOKEN 并根据 pitch_shift 调整 pitch 值。
        - 当 y 提供时，返回 (x_processed, y_processed) 用于 teacher forcing；否则仅返回 x_processed 用于推理。
        返回的每个 token 的最后一维长度为 subseq_length // 3 * 2（两个 token 的组合）。
        """
        batch_size, seq_length, subseq_length = x.shape
        x = x.long().view(batch_size, seq_length, subseq_length // 3, 3)
        x_processed = torch.zeros(
            batch_size,
            seq_length,
            subseq_length // 3,
            2,
            dtype=torch.long,
            device=x.device,
        )
        pad_indices = x[:, :, :, 1] == 255
        eos_indices = x[:, :, :, 0] == 254
        is_not_drum = x[:, :, :, 0] != 127
        # 首个 token 的 type 字段（0 表示输入）
        x_processed[:, :, :, 0] = 0
        # 第二个 token 字段合成 pitch + duration*128 + offset + pitch_shift（如果不是 drum）
        x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2]) * 128 + 2 + pitch_shift[:, None, None] * is_not_drum
        x_processed[pad_indices] = PAD_TOKEN
        x_processed[:, :, :, 0][eos_indices] = EOS_TOKEN

        if y is None:
            return x_processed.view(batch_size, seq_length, subseq_length // 3 * 2)
        else:
            # 如果有目标 y，则对 y 做相同的处理，type 字段设置为 1（表示目标）
            x_processed = self._process_triplet_tensor(x, pitch_shift, type_value=0)
            if y is None:
                return x_processed
            y_processed = self._process_triplet_tensor(y, pitch_shift, type_value=1)
            return x_processed, y_processed

    def training_step(self, batch: OldPtModelInput, batch_idx: int):
        # Expect batch to have attributes mel_data, acc_data, pitch_shift
        mel = batch.mel_data
        acc = batch.acc_data
        pitch_shift = batch.pitch_shift
        loss = self.loss(mel, acc, pitch_shift)
        safe_lr = None
        try:
            sched = self.lr_schedulers()
            safe_lr = sched.get_last_lr()[0] if sched else None
        except Exception:
            safe_lr = None
        self.metric_manager.update(
            self,
            batch_idx,
            group_name="training_state",
            loss=loss,
            learning_rate=safe_lr,
        )
        self.metric_manager.log(self, phase="train", batch_idx=batch_idx)
        return loss
    
    def save(self, data, filename_base, extension, save_dir, **kwargs):
        return super().save(data, filename_base, extension, save_dir, **kwargs)

    def validation_step(self, batch: OldPtModelInput, batch_idx: int):
        mel = batch.mel_data
        acc = batch.acc_data
        pitch_shift = batch.pitch_shift
        loss = self.loss(mel, acc, pitch_shift)
        self.metric_manager.update(
            self,
            batch_idx,
            group_name="training_state",
            loss=loss,
        )
        self.metric_manager.log(
            self,
            phase="val",
            batch_idx=batch_idx,
        )
        return loss

    def global_sampling(self, x, x_mel_gt=None, max_seq_len=384, temperature=1.0):
        batch_size, seq_len, subseq_len = x.shape
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2 == 0).long()  # → [seq_len], 1 at even idx (acc), 0 at odd idx (mel)
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)
        h, _ = self.local_encode(x, token_type_ids)
        h = h.view(batch_size, seq_len, -1)
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list a0,m0,a1,m1,a_to_be_2
        if x_mel_gt is not None:
            _, seq_len_gt, _ = x_mel_gt.shape
            h_mel, _ = self.local_encode(
                x_mel_gt,
                torch.zeros(*x_mel_gt.shape[:-1], x_mel_gt.shape[-1] + 1, device=x_mel_gt.device, dtype=x_mel_gt.dtype),
            )
            h_mel = h_mel.view(batch_size, seq_len_gt, -1)
            print("with gt!")
            for i in range(0, max_seq_len):
                if i % 10 == 0:
                    # print('Sampling', i, '/', max_seq_len)
                    ...
                if i % 2 == 0:
                    # generate accompaniment for this timestep and append its summary to global h
                    y_next, h = self._generate_local_and_update_h(
                        h, subseq_len=subseq_len, temperature=temperature, token_type_one=True
                    )
                    y.append(y_next)
                else:
                    # token_type_ids = torch.zeros((b, s, l+1), dtype=torch.long, device=y_next.device)
                    h_prev_mel = h_mel[:, i // 2, :].unsqueeze(1)  # [B, 1, H]
                    h = torch.cat([h, h_prev_mel], dim=1)  # [B, cur_len, H]
                    y.append(x_mel_gt[:, i // 2, :])
        else:
            for i in range(0, max_seq_len):
                # alternate between generating accompaniment (even i) and inserting melody summary (odd i)
                token_one = i % 2 == 0
                y_next, h = self._generate_local_and_update_h(
                    h, subseq_len=subseq_len, temperature=temperature, token_type_one=token_one
                )
                y.append(y_next)
        return y

    def global_sampling_from_scratch(self, x_mel: torch.LongTensor, temperature: float = 1.0, max_seq_len=384):
        B, S, L = x_mel.shape
        device = x_mel.device

        # Build program IDs = 0 for all melody tokens
        # token_type_ids = torch.zeros_like(x_mel, dtype=torch.long)  # [B, S, L]
        h_mel, _ = self.local_encode(x_mel, torch.zeros((B, S, L + 1), device=device, dtype=x_mel.dtype))
        h_mel = h_mel.view(B, S, self.hidden_size)  # [B, S, H]

        # Prepare SOS for global
        sos = self.global_sos.view(1, 1, -1).repeat(B, 1, 1)  # [B, 1, H]

        # Will store generated accompaniment frames
        y = []  # each entry: [B, L]

        # Start with just [SOS]
        h = sos  # [B, 1, H]
        for t in range(max_seq_len):
            # if t % 10 == 0:
            #         print('Sampling', t, '/', max_seq_len)
            if t > 0:
                # Append previous melody summary before generating new accompaniment
                h_prev_mel = h_mel[:, t - 1, :].unsqueeze(1)  # [B, 1, H]
                h = torch.cat([h, h_prev_mel], dim=1)  # [B, cur_len, H]
                y.append(x_mel[:, t - 1, :])

            # generate accompaniment for this melody timestep and append its summary to global h
            y_next, h = self._generate_local_and_update_h(h, subseq_len=L, temperature=temperature, token_type_one=True)
            y.append(y_next)
        return y  # list of S tensors [B, L]

    # ----- Inference helpers (generation + saving) -----
    def generate(
        self,
        x: torch.LongTensor = None,
        x_mel_gt: torch.LongTensor = None,
        prompt_length: int = 0,
        generation_length: int = 384,
        temperature: float = 1.0,
        n_samples: int = 1,
        from_scratch: bool = False,
        gt_mel: bool = True,
    ):
        """Unified generation entrypoint.

        - from_scratch=True: expects `x_mel_gt` (mel frames) and will call
          `global_sampling_from_scratch` (returns list of timesteps [B, L]).
        - from_scratch=False: expects `x` (stacked acc/mel frames) and optional
          `x_mel_gt` for teacher-forced mel frames; calls `global_sampling`.

        Returns the raw list-of-timesteps output (each entry is a Tensor [B, L]).
        """
        # Validate inputs
        if from_scratch:
            assert x_mel_gt is not None, "from_scratch requires x_mel_gt"

        # repeat batch n_samples times and call the appropriate sampler inside no_grad
        with torch.no_grad():
            if from_scratch:
                # x_mel_gt: [B, S, L]
                x_mel = x_mel_gt.repeat(n_samples, 1, 1)
                outputs = self.global_sampling_from_scratch(
                    x_mel, temperature=temperature, max_seq_len=generation_length
                )
            else:
                assert x is not None, "generate requires x when from_scratch=False"
                x_rep = x.repeat(n_samples, 1, 1)
                x_mel_gt_rep = None
                if x_mel_gt is not None:
                    x_mel_gt_rep = x_mel_gt.repeat(n_samples, 1, 1)
                outputs = self.global_sampling(
                    x_rep,
                    x_mel_gt=x_mel_gt_rep if gt_mel else None,
                    temperature=temperature,
                    max_seq_len=generation_length,
                )

        return outputs

    def tokens_to_prettymidi(self, outputs, tempo: float = 90.0, single: bool = False):
        """Convert model token outputs to a list of pretty_midi.PrettyMIDI objects.

        - outputs: list of timesteps, each a Tensor of shape [B, L] (batch first).
        - returns: list of PrettyMIDI objects, one per batch element.
        """
        # lazy imports so normal training doesn't require pretty_midi
        try:
            import pretty_midi
        except Exception:  # pretty_midi may be unavailable in some envs
            pretty_midi = None
        try:
            from StreamMUSE2.preprocess.preprocess_midi2pt_dataset import DURATION_TEMPLATES
        except Exception:
            # fallback: minimal durations (quarter-note)
            DURATION_TEMPLATES = [1.0]

        # outputs: list of T steps, each [B, L]
        T = len(outputs)
        if T == 0:
            return []

        # stack along time to shape [B, T, L]
        seqs = [o for o in outputs]
        if isinstance(seqs[0], torch.Tensor):
            # each seq tensor: [B, L] -> stack -> [B, T, L]
            stacked = torch.stack(seqs, dim=1)
        else:
            # fallback to tensor conversion
            stacked = torch.tensor(seqs)

        # stacked: [B, T, L]
        midis = []
        time_step_length = 60.0 / tempo / 4
        if pretty_midi is None:
            raise RuntimeError("pretty_midi is required to convert tokens to MIDI; install pretty_midi")

        for b in range(stacked.shape[0]):
            midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
            instrument_map = {}
            for t in range(stacked.shape[1]):
                content = stacked[b, t]
                time_step = t if single else t // 2
                start_time = time_step * time_step_length
                for i in range(0, content.shape[-1], 2):
                    program = int(content[i].item())
                    if program == EOS_TOKEN:
                        break
                    if i + 1 >= content.shape[-1]:
                        break
                    pitch_duration = int(content[i + 1].item()) - 2
                    pitch = pitch_duration % 128
                    duration = pitch_duration // 128
                    if program not in (0, 1):
                        # skip invalid program
                        continue
                    if pitch < 0 or pitch >= 128:
                        continue
                    if duration < 0 or duration >= len(DURATION_TEMPLATES):
                        continue
                    end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time
                    if program not in instrument_map:
                        if program == 0:
                            inst = pretty_midi.Instrument(program=24, name="Guitar")
                        else:
                            inst = pretty_midi.Instrument(program=0, name="Piano")
                        instrument_map[program] = inst
                        midi.instruments.append(inst)
                    inst = instrument_map[program]
                    inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time))
            midis.append(midi)
        return midis

    def save_tokens_as_midi(self, outputs, save_path, tempo: float = 90.0, single: bool = False):
        """Save model token outputs to disk as a MIDI file or multiple files.

        - outputs: list of timesteps (each [B, L]). If batch size >1, this will save
          one file per batch element by appending an index to `save_path`.
        - save_path: if batch==1, exact file path; if batch>1, treated as a directory or
          a template (e.g. 'out/sample' -> 'out/sample_0.mid').
        """
        import os

        midis = self.tokens_to_prettymidi(outputs, tempo=tempo, single=single)
        if len(midis) == 0:
            return []
        saved = []
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if len(midis) == 1:
            path = save_path if save_path.endswith(".mid") else save_path + ".mid"
            midis[0].write(path)
            saved.append(path)
        else:
            base, ext = (save_path, ".mid") if save_path.endswith(".mid") else (save_path, ".mid")
            for i, midi in enumerate(midis):
                path = f"{base}_{i}{ext}"
                midi.write(path)
                saved.append(path)
        return saved

    # def configure_optimizers(self):
    #     max_lr = 1e-4
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr)
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer, max_lr=max_lr, total_steps=MAX_STEPS, pct_start=0.005
    #     )
    #     return [optimizer], [scheduler]

    def _move_to_device(self, batch: OldPtModelInput) -> OldPtModelInput:
        """将 batch 中的张量移动到模型的 device 上。

        这个函数对 batch 的每个属性调用 .to(self.device)（如果存在），并返回
        与原始 batch 同类型的对象（通过 type(batch) 重建）。这对多设备训练/推理很有用。
        """
        return type(batch)(**{k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.__dict__.items()})
