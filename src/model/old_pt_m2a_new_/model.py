from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.model import BaseModel
from .config import OldPtM2ANewConfig
from ...dataset.old_pt.model_input import OldPtModelInput

# 设置调试日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
DEBUG = False

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


class OldPtM2ANew(BaseModel):
    """基于 RoFormer 的分层局部/全局编码器模型（旧版 Pt 变体）。

    设计要点：
    - 使用局部 encoder/decoder 对每个子序列（subseq）进行编码/解码。
    - 使用全局 encoder 将局部编码的表示按时间序列合并并进行跨子序列的上下文建模。
    - 提供采样、预处理、损失计算等工具函数以便训练和推理。
    """

    def __init__(self, config: OldPtM2ANewConfig):
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
        from ...network.customized_roformer.network import CustomizedRoFormerEncoder

        # Use the provided RoFormerConfig instances from the config params. These
        # were supplied via `CustomizedRoFormerEncoderParams.config`.
        main_roformer_config = global_params.config
        self.model = CustomizedRoFormerEncoder(main_roformer_config)
        local_encoder_config = local_enc_params.config
        local_decoder_config = local_dec_params.config

        self.local_embedding = nn.Embedding(N_TOKENS, self.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        with torch.no_grad():
            self.token_type_embeddings.weight.mul_(2.0)

        self.local_encoder = CustomizedRoFormerEncoder(local_encoder_config)
        self.local_decoder = CustomizedRoFormerEncoder(local_decoder_config)
        self.final_decoder = nn.Linear(self.hidden_size, N_TOKENS)
        self.global_sos = nn.Parameter(torch.randn(self.hidden_size))
        self._future_mask = torch.empty(0)

    # --- Small helpers to improve readability ---
    def _prepare_local_inputs(self, x: torch.LongTensor, token_type_ids: torch.LongTensor):
        """为局部编码器准备输入：展平、添加SOS、构建mask和embeddings。

        处理流程：
        1. 将 [batch, seq, subseq] 展平为 [batch*seq, subseq]
        2. 在每个子序列前添加 SOS_TOKEN
        3. 构建 attention mask（非PAD位置为True）
        4. 计算 word embedding 和 type embedding 并相加

        Args:
            x: 输入张量，形状 [batch, seq, subseq]
            token_type_ids: token类型ID，形状 [batch, seq, subseq]

        Returns:
            tuple: (mask, emb, batch_size, seq_len)
            - mask: attention mask，形状 [batch*seq, subseq+1]
            - emb: 组合embedding，形状 [batch*seq, subseq+1, hidden_size]
            - batch_size: 原始batch大小
            - seq_len: 原始序列长度

        潜在问题排查：
            如果出现维度不匹配错误，检查：
            1. token_type_ids 的形状是否为 [batch, seq, subseq+1]
            2. word_emb 和 type_emb 的序列长度维度是否一致
        """
        batch_size, seq_len, subseq_len = x.shape
        DEBUG and logger.debug(f"x.shape = {x.shape}")
        DEBUG and logger.debug(f"token_type_ids.shape = {token_type_ids.shape}")

        # 1. 展平 batch 和 seq 维度：[batch, seq, subseq] -> [batch*seq, subseq]
        x_flat = x.view(-1, subseq_len)
        DEBUG and logger.debug(f"x_flat.shape = {x_flat.shape}")

        # 2. 在每个子序列前添加 SOS_TOKEN：[batch*seq, subseq] -> [batch*seq, subseq+1]
        # x_flat = torch.cat(
        #     [torch.full((x_flat.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=x_flat.device), x_flat],
        #     dim=-1,
        # )
        DEBUG and logger.debug(f"x_flat after SOS.shape = {x_flat.shape}")

        # 3. 构建 attention mask：非PAD位置为True
        mask = x_flat != PAD_TOKEN
        DEBUG and logger.debug(f"mask.shape = {mask.shape}")

        # 4. 计算 word embedding
        word_emb = self.local_embedding(x_flat)
        DEBUG and logger.debug(f"word_emb.shape = {word_emb.shape}")

        # 5. 重塑 token_type_ids 并计算 type embedding
        # 关键：token_type_ids 应该已经包含 SOS 位置的维度
        # token_type_ids 原形状: [batch, seq, subseq]
        # 需要重塑为: [batch*seq, subseq]，最后一维应该是1（表示每个位置的type）
        DEBUG and logger.debug(f"token_type_ids.shape before reshape = {token_type_ids.shape}")
        DEBUG and logger.debug(f"x_flat.shape = {x_flat.shape}")
        DEBUG and logger.debug(f"x.shape = {x.shape}")
        type_emb_input = token_type_ids.reshape(batch_size * seq_len, -1)
        DEBUG and logger.debug(f"type_emb_input.shape = {type_emb_input.shape}")

        type_emb = self.token_type_embeddings(type_emb_input)
        DEBUG and logger.debug(f"type_emb.shape = {type_emb.shape}")

        # 6. 组合两种 embedding
        # 这里如果出现维度不匹配，说明 word_emb 和 type_emb 的序列长度不一致
        emb = word_emb + type_emb
        DEBUG and logger.debug(f"final emb.shape = {emb.shape}")

        return mask, emb, batch_size, seq_len

    def _process_triplet_tensor(self, t: torch.LongTensor, pitch_shift: torch.LongTensor, type_value: int):
        """将三元组表示的音乐数据转换为模型可用的 token 格式。

        数据转换流程：
        输入: [batch, seq, subseq] 其中每3个连续元素为一个三元组 (program, pitch, duration)
        输出: [batch, seq, subseq//3*2] 其中每个三元组转换为2个 token (type, combined_value)

        Args:
            t: 原始音乐数据张量，形状 [batch, seq, subseq]
               - 每3个连续元素组成一个音乐事件：[program, pitch, duration]
               - program: 乐器类型 (0-127, 254表示EOS, 127表示鼓)
               - pitch: 音高 (0-127, 255表示PAD)
               - duration: 时长 (0-24, 表示不同的时值)
            pitch_shift: 音高偏移量，形状 [batch]，用于数据增强
            type_value: token 类型标识符
                       - 0: 输入数据（如旋律）
                       - 1: 目标数据（如伴奏）

        Returns:
            处理后的张量，形状 [batch, seq, subseq//3*2]
            每个原始三元组 (program, pitch, duration) 转换为两个 token:
            - token1: type_value (用于区分数据类型)
            - token2: combined_value (编码了 pitch + duration*128 + offset + pitch_shift)

        特殊值处理：
            - PAD_TOKEN: 用于填充序列到固定长度
            - EOS_TOKEN: 标记序列结束
            - 鼓声不应用 pitch_shift（因为鼓的音高概念不同）
        """
        batch_size, seq_length, subseq_length = t.shape

        # 将输入重塑为三元组格式：[batch, seq, num_events, 3]
        # 其中 num_events = subseq_length // 3，每个 event 包含 [program, pitch, duration]
        t = t.long().view(batch_size, seq_length, subseq_length // 3, 3)

        # 创建输出张量：每个三元组将转换为2个 token
        out = torch.zeros(
            batch_size,
            seq_length,
            subseq_length // 3,  # 事件数量
            2,  # 每个事件2个 token
            dtype=torch.long,
            device=t.device,
        )

        # 识别特殊标记位置
        pad_indices = t[:, :, :, 1] == 255  # pitch=255 表示 PAD
        eos_indices = t[:, :, :, 0] == 254  # program=254 表示 EOS（序列结束）
        is_not_drum = t[:, :, :, 0] != 127  # program=127 表示鼓，鼓不应用音高偏移

        # 设置第一个 token：数据类型标识符
        # type_value=0 表示输入数据（旋律），type_value=1 表示目标数据（伴奏）
        out[:, :, :, 0] = type_value

        # 设置第二个 token：组合编码值
        # 公式: pitch + duration*128 + 2 + pitch_shift*is_not_drum
        # - pitch: 原始音高 (0-127)
        # - duration*128: 时长左移7位，为 pitch 留出空间
        # - +2: 基础偏移，避免与特殊 token 冲突
        # - pitch_shift: 只对非鼓声应用音高偏移（数据增强）
        out[:, :, :, 1] = (
            t[:, :, :, 1]  # pitch
            + (t[:, :, :, 2]) * 128  # duration * 128
            + 2  # base offset
            + pitch_shift[:, None, None] * is_not_drum  # conditional pitch shift
        )

        # 处理特殊位置
        out[pad_indices] = PAD_TOKEN  # PAD 位置设为 PAD_TOKEN
        out[:, :, :, 0][eos_indices] = EOS_TOKEN  # EOS 位置的第一个 token 设为 EOS_TOKEN

        # 重塑为最终输出格式：[batch, seq, subseq//3*2]
        return out.view(batch_size, seq_length, subseq_length // 3 * 2)

    def local_encode(self, x: torch.LongTensor, token_type_ids: torch.LongTensor) -> torch.Tensor:
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
            - h: 每个子序列的表示 [batch*seq, subseq, hidden]
        """
        mask, emb, batch_size, seq_len = self._prepare_local_inputs(x, token_type_ids)
        h = self.local_encoder(emb)[0]
        return h

    def local_decode(self, h: torch.Tensor, reduced_global_h: torch.Tensor):
        """局部解码器：将局部隐藏表示与 embeddings 拼接并解码为 token logits。

        - h: 局部聚合表示，形状 [batch*seq, subseq, hidden]
        - reduced_global_h: 全局上下文表示，形状 [batch, seq, hidden]
        返回 logits，形状与解码器输出对应（用于后续 softmax/采样）。
        - logits: [batch*seq, subseq, N_TOKENS]
        """
        batch_size, seq_len, hidden_size = reduced_global_h.shape
        reduced_global_h = reduced_global_h.view(batch_size * seq_len, 1, -1)  # [B*S, 1, H]

        h = h + reduced_global_h  # [B*S, sub_S, H]
        hidden = self.local_decoder(h)[0]
        return self.final_decoder(hidden)

    def local_sampling(self, h: torch.Tensor, reduced_global_h: torch.Tensor, temperature: float = 1.0):
        """基于局部 decoder 进行自回归采样。

        - h: 局部聚合向量，作为采样初始上下文 [batch*seq, subseq, hidden]
        - reduced_global_h: 全局上下文表示，形状 [batch*seq, hidden]
        - temperature: 采样温度，0 表示贪心（取 argmax）
        过程：迭代地通过 local_decoder 预测下一个 token，直到所有样本触发 EOS 或达到最大长度。
        返回采样到的 token id 序列（不含 SOS）。
        - y: [batch*seq, subseq, 1]
        """
        dec_h = self.local_decoder(h)[0]
        logits = self.final_decoder(dec_h)  # [B*S, sub_S, N_TOKENS]
        if temperature == 0:
            next_probs = F.one_hot(logits.argmax(dim=-1), N_TOKENS).float()  # [B*S, sub_S, N_TOKENS]
        else:
            next_probs = F.softmax(logits / temperature, dim=-1)  # [B*S, sub_S, N_TOKENS]
            
        next_probs = next_probs.reshape(-1, N_TOKENS)  # [B*S*sub_S, N_TOKENS]
        y_next = torch.multinomial(next_probs, 1)  # [B*S*sub_S, 1]
        y_next = y_next.view(h.size(0), h.size(1), 1)  # [B*S, sub_S, 1]
        return y_next

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

    def forward(self, x: torch.LongTensor):
        """主前向函数：执行局部编码 -> 合并 -> 全局编码 -> 局部解码 的流程。

        输入 x 的形状为 [batch, seq, subseq]：seq 是帧序列数量，subseq 是每帧内部的 token 数。
        输出 logits 形状为 [batch*seq, subseq, N_TOKENS]，用于后续的 softmax 或采样。
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
        # sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        # token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)

        # 局部编码
        local_h = self.local_encode(x, token_type_ids)  # [B*S, sub_S, H]
        reduced_global_h = local_h[:, -1, :]  # 取每个子序列的最后一个 token 作为聚合向量 [B*S, H]
        reduced_global_h = reduced_global_h.view(batch_size, seq_len, -1)  # [B, S, H]
        # 在全局序列前加入 SOS 并把前一时刻的隐藏拼接进来（自回归偏移）
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        reduced_global_h = torch.cat([sos, reduced_global_h[:, :-1, :]], dim=1)  # [B, S, H] -> [B, S+1, H]

        # 全局 encoder（使用 buffered_future_mask 保证自回归）
        reduced_global_h = self.model(
            reduced_global_h, attention_mask=self.buffered_future_mask(reduced_global_h), interleave_pos=True
        )[0]  # [B, S+1, H]
        return self.local_decode(local_h, reduced_global_h)

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
        # 处理旋律数据 x (type_value=0 表示输入)
        x_processed = self._process_triplet_tensor(x, pitch_shift, type_value=0)

        if y is None:
            return x_processed
        else:
            # 处理伴奏数据 y (type_value=1 表示目标)
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

    def global_sampling(self, x: torch.Tensor, x_mel_gt: torch.Tensor = None, max_seq_len=384, temperature=1.0):
        batch_size, seq_len, subseq_len = x.shape
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2 == 0).long()  # → [seq_len], 1 at even idx (acc), 0 at odd idx (mel)
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        local_h = self.local_encode(x, token_type_ids)  # [B*S, sub_S, H]
        global_h = local_h.view(batch_size, seq_len, subseq_len, -1)  # [B, S, sub_S, H]
        sos = self.global_sos.view(1, 1, 1, -1).repeat(batch_size, seq_len, subseq_len, 1)
        global_h = torch.cat([sos, global_h], dim=1)  # [B, S+1, sub_S, H]

        reduced_global_h = global_h[:, :, -1, :]  # [B, S+1, H]

        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list a0,m0,a1,m1,a_to_be_2
        if x_mel_gt is not None:
            _, seq_len_gt, _ = x_mel_gt.shape
            local_h_mel = self.local_encode(
                x_mel_gt,
                torch.zeros_like(x_mel_gt, dtype=torch.long, device=x_mel_gt.device),
            )  # [B*S_mel, sub_S, H]
            global_h_mel = local_h_mel.view(batch_size, seq_len_gt, subseq_len, -1)  # [B, S_mel, sub_S, H]
           
            # SOS mel 不该存在
            # sos_mel = self.global_sos.view(1, 1, 1, -1).repeat(batch_size, 1, 1, 1)
            # global_h_mel = torch.cat([sos_mel, global_h_mel[:, :-1, :, :]], dim=1)  # [B, S_mel -1 +1, sub_S, H]
            reduced_global_h_mel = global_h_mel[:, :, -1, :]  # [B, S_mel, H]
            print("with gt!")
            for i in range(0, max_seq_len):
                print(f"Sampling {i}/{max_seq_len}")
                if i % 10 == 0:
                    # print('Sampling', i, '/', max_seq_len)
                    ...
                if i % 2 == 0:
                    reduced_global_h_out = self.model(
                        reduced_global_h,
                        attention_mask=self.buffered_future_mask(reduced_global_h),
                        interleave_pos=True,
                    )[0]  # [B, cur_len, H]
                    print(f"reduced_ golobal h out shape: {reduced_global_h_out.shape}")
                    y_next = self.local_sampling(
                        local_h, reduced_global_h=reduced_global_h_out, temperature=temperature
                    )  # [B*S, sub_S, 1]
                    y_next = y_next.view(batch_size, -1, subseq_len)  # [B, S, sub_S]
                    print(f"y_next shape: {y_next.shape}")
                    y.append(y_next[:, -1, :])  # [B, sub_S]
                    # next mel frame: 0
                    token_type_ids = torch.ones_like(y_next[:, -1:, :], dtype=torch.long, device=y_next.device)  # [B,1, sub_S]
                    
                    local_h = self.local_encode( y_next[:, -1:, :], token_type_ids=token_type_ids)  # [B*S, sub_S, H]
                    print(f"local h shape: {local_h.shape}")
                    reduced_global_h = torch.cat(
                        [
                            reduced_global_h,
                            local_h.view(
                                batch_size, -1, subseq_len, self.hidden_size
                            )[:, :, -1, :],
                        ],
                        dim=1,
                    )  # [B, cur_len + 1, H]
                elif i % 2 == 1:  # use gt melody
                    global_h_prev_mel = reduced_global_h_mel[:, i // 2, :].unsqueeze(1)  # [B, 1, H]
                    reduced_global_h = torch.cat([reduced_global_h, global_h_prev_mel], dim=1)  # [B, cur_len + 1, H]
                    y.append(x_mel_gt[:, i // 2, :])  # [B, S_mel//2 ,sub_S]
        else:
            for i in range(0, max_seq_len):
                # if i % 10 == 0:
                #     print('Sampling', i, '/', max_seq_len)
                reduced_global_h_out = self.model(
                    reduced_global_h,
                    attention_mask=self.buffered_future_mask(reduced_global_h),
                    interleave_pos=True,
                )[0]  # [B, cur_len, H]
                y_next = self.local_sampling(
                    local_h,
                    reduced_global_h_out,
                    temperature=temperature,
                )  # [B*S, sub_S, 1]
                y.append(y_next)
                if i % 2 == 0:  # next acc frame: 1 ; next mel frame: 0
                    token_type_ids = torch.ones_like(y_next, dtype=torch.long, device=y_next.device)  # [B, S, sub_S]
                else:
                    token_type_ids = torch.zeros_like(y_next, dtype=torch.long, device=y_next.device)  # [B, S, sub_S]

                reduced_global_h = torch.cat(
                    [
                        reduced_global_h,
                        self.local_encode(y_next, token_type_ids=token_type_ids)[0].view(
                            batch_size, -1, subseq_len, self.hidden_size
                        )[:, -1, :, :],
                    ],
                    dim=1,
                )  # [B, cur_len + 1, H]
        return y

    def global_sampling_from_scratch(self, x_mel: torch.LongTensor, temperature: float = 1.0, max_seq_len=384):
        batch_size, seq_len, subseq_len = x_mel.shape

        device = x_mel.device

        # Build program IDs = 0 for all melody tokens
        token_type_ids = torch.zeros_like(x_mel, dtype=torch.long)  # [B, S, sub_S]
        local_h_mel = self.local_encode(x_mel, token_type_ids=token_type_ids)  # [B*S, sub_S, H]
        global_h_mel = local_h_mel.view(batch_size, seq_len, subseq_len, -1)  # [B, S, sub_S, H]
        local_h = local_h_mel.clone()
        reduced_global_h_mel = global_h_mel[:, :, -1, :]
        reduced_global_h = reduced_global_h_mel.clone()  # [B, S, H]

        # local_h_mel = local_h_mel.view(batch_size, seq_len, self.hidden_size)  # [B, S, H]

        # Prepare SOS for global
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)  # [B, 1, H]

        # Will store generated accompaniment frames
        y = []  # each entry: [B, L]

        # Start with just [SOS]
        h = sos  # [B, 1, H]
        for t in range(max_seq_len):
            if t > 0:
                # Append previous melody summary before generating new accompaniment
                global_h_prev_mel = reduced_global_h_mel[:, t - 1, :].unsqueeze(1)  # [B, 1, H]
                reduced_global_h = torch.cat([reduced_global_h, global_h_prev_mel], dim=1)  # [B, cur_len + 1, H]
                y.append(x_mel[:, t - 1, :])  # [B, S_mel//2 ,sub_S]

            reduced_global_h_out = self.model(
                reduced_global_h, attention_mask=self.buffered_future_mask(reduced_global_h), interleave_pos=True
            )[0]
            y_next = self.local_sampling(local_h, reduced_global_h_out, temperature=temperature)
            y.append(y_next)

            # next acc frame: 1 ; next mel frame: 0
            token_type_ids = torch.ones_like(y_next, dtype=torch.long, device=y_next.device)  # [B, S, sub_S]

            reduced_global_h = torch.cat(
                [
                    reduced_global_h,
                    self.local_encode(y_next, token_type_ids=token_type_ids)[0].view(
                        batch_size, -1, subseq_len, self.hidden_size
                    )[:, -1, :, :],
                ],
                dim=1,
            )  # [B, cur_len + 1, H]
        return y  # list of S tensors [B, L]

    def loss(self, x_mel, x_acc, pitch_shift):
        """计算模型的交叉熵损失。

        该方法实现了分层模型的损失计算：
        1. 对输入的旋律和伴奏数据进行预处理
        2. 将伴奏和旋律交替排列构成训练序列
        3. 构建目标序列，其中旋律位置被掩码（设为 PAD_TOKEN）
        4. 通过前向传播得到 logits 并计算交叉熵损失

        Args:
            x_mel: 旋律数据，形状 [batch, seq, subseq]
            x_acc: 伴奏数据，形状 [batch, seq, subseq]
            pitch_shift: 音高偏移量，形状 [batch]

        Returns:
            交叉熵损失值
        """
        # 预处理旋律和伴奏数据
        x_mel_proc, x_acc_proc = self.preprocess(x_mel, pitch_shift, y=x_acc)
        batch_size, seq_len, subseq_len = x_mel_proc.shape

        # 将伴奏和旋律交替排列：[acc0, mel0, acc1, mel1, ...]
        stacked = torch.stack([x_acc_proc, x_mel_proc], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)

        # 构建目标序列（与输入相同）
        x_target = x.clone()

        # 构建掩码：在奇数位置（旋律位置）设置为 True
        idx = torch.arange(seq_len * 2, device=x.device)
        mel_mask = (idx % 2 == 1).unsqueeze(0).unsqueeze(-1)  # [1, 2*S, 1]
        mel_mask = mel_mask.expand(batch_size, seq_len * 2, subseq_len)  # [B, 2*S, L]

        # 将旋律位置的目标设为 PAD_TOKEN（模型只需要预测伴奏）
        x_target[mel_mask] = PAD_TOKEN

        # 前向传播得到 logits
        logits = self(x)

        # 计算交叉熵损失
        return F.cross_entropy(logits.view(-1, N_TOKENS), x_target.view(-1), ignore_index=PAD_TOKEN)
