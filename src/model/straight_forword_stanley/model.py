from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.model import BaseModel
from .config import OldPtM2ATransformerConfig
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
N_TOKENS = N_NORMAL_TOKENS + 7
SOS_TOKEN = N_NORMAL_TOKENS
EOS_TOKEN = N_NORMAL_TOKENS + 1
PAD_TOKEN = N_NORMAL_TOKENS + 2
SOM_TOKEN = N_NORMAL_TOKENS + 3  # Start of Melody
EOM_TOKEN = N_NORMAL_TOKENS + 4  # End of Melody
SOA_TOKEN = N_NORMAL_TOKENS + 5  # Start of Accompaniment
EOA_TOKEN = N_NORMAL_TOKENS + 6  # End of Accompaniment


def fill_with_neg_inf(t: torch.Tensor) -> torch.Tensor:
    """FP16 兼容的工具函数：将张量填充为 -inf。

    这个函数用于构造 attention mask（未来位置填充为 -inf），
    在混合精度下先转换为 float 再填充，然后转换回原始类型以保证数值稳定性。
    输入: 任意 shape 的张量 t，通常是 zeros([L, L])；返回: 相同 shape、类型的 -inf 填充值张量。
    """
    return t.float().fill_(float("-inf")).type_as(t)


class OldPtM2ATransformer(BaseModel):
    """基于 RoFormer 的分层局部/全局编码器模型，直接使用的是原版的 RoFormer，因为不需要 interleaving。

    设计要点：
    - 使用局部 encoder/decoder 对每个子序列（subseq）进行编码/解码。
    - 使用全局 encoder 将局部编码的表示按时间序列合并并进行跨子序列的上下文建模。
    - 采用 [SOM, mel, EOM, SOA, acc, EOA] 的序列结构。
    - Loss 函数只在伴奏部分计算。
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
        from transformers import RoFormerEncoder

        # Use the configs to instantiate the three RoFormerEncoder networks.
        self.model = RoFormerEncoder(global_params)

        self.local_embedding = nn.Embedding(N_TOKENS, self.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        with torch.no_grad():
            self.token_type_embeddings.weight.mul_(2.0)

        self.local_encoder = RoFormerEncoder(local_enc_params)
        self.local_decoder = RoFormerEncoder(local_dec_params)
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
            token_type_ids: token类型ID，形状 [batch, seq, subseq+1]
                          注意：比x多一维，因为包含了SOS位置的type

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
        x_flat = torch.cat(
            [torch.full((x_flat.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=x_flat.device), x_flat],
            dim=-1,
        )
        DEBUG and logger.debug(f"x_flat after SOS.shape = {x_flat.shape}")

        # 3. 构建 attention mask：非PAD位置为True
        mask = x_flat != PAD_TOKEN
        DEBUG and logger.debug(f"mask.shape = {mask.shape}")

        # 4. 计算 word embedding
        word_emb = self.local_embedding(x_flat)
        DEBUG and logger.debug(f"word_emb.shape = {word_emb.shape}")

        # 5. 重塑 token_type_ids 并计算 type embedding
        # 关键：token_type_ids 应该已经包含 SOS 位置的维度
        # token_type_ids 原形状: [batch, seq, subseq+1]
        # 需要重塑为: [batch*seq, subseq+1]，最后一维应该是1（表示每个位置的type）
        type_emb_input = token_type_ids.view(batch_size * seq_len, -1)
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

    def forward(self, x: torch.LongTensor, token_type_ids: torch.LongTensor):
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

        # 构造用于局部编码器的 token type ids
        # 传入的 token_type_ids 是 [batch, seq], 需要扩展
        local_token_type_ids = token_type_ids.unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        sos_type = token_type_ids.unsqueeze(-1).expand(batch_size, seq_len, 1)
        local_token_type_ids = torch.cat([sos_type, local_token_type_ids], dim=-1)

        # 局部编码
        h, emb = self.local_encode(x, local_token_type_ids)
        h = h.view(batch_size, seq_len, -1)

        # 构造全局编码器的输入 (关键的自回归结构)
        # 在全局序列前加入 SOS，并使用前一时刻的真实输出来预测当前时刻
        # h[:, :-1] 创建了一个向右平移一位的序列，这是标准的 Teacher Forcing
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)

        # 为全局编码器准备 token_type_ids
        # 注意：这里的 token_type_ids 也需要与 h_shifted 对齐
        global_token_type_ids = torch.cat(
            [
                torch.zeros(batch_size, 1, device=x.device, dtype=torch.long),  # SOS type
                token_type_ids[:, :-1],  # 与 h[:, :-1] 对应
            ],
            dim=1,
        )

        # 全局 encoder（使用 buffered_future_mask 保证自回归）
        h_out = self.model(h, attention_mask=self.buffered_future_mask(h), token_type_ids=global_token_type_ids)[0]

        return self.local_decode(h_out, emb)

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

    # def training_step(self, batch: OldPtModelInput, batch_idx: int):
    #     # Expect batch to have attributes mel_data, acc_data, pitch_shift
    #     mel = batch.mel_data
    #     acc = batch.acc_data
    #     pitch_shift = batch.pitch_shift
    #     loss = self.loss(mel, acc, pitch_shift)
    #     self.log(
    #         "train_loss",
    #         loss,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         sync_dist=True,
    #     )
    #     scheduler = self.lr_schedulers()
    #     if scheduler:
    #         scheduler.step()
    #         self.log(
    #             "training/lr",
    #             scheduler.get_last_lr()[0],
    #             on_step=True,
    #             on_epoch=True,
    #             prog_bar=True,
    #             logger=True,
    #             sync_dist=True,
    #         )
    #     return loss

    # def validation_step(self, batch: OldPtModelInput, batch_idx: int):
    #     mel = batch.mel_data
    #     acc = batch.acc_data
    #     pitch_shift = batch.pitch_shift
    #     loss = self.loss(mel, acc, pitch_shift)
    #     self.log(
    #         "val_loss",
    #         loss,
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #         sync_dist=True,
    #     )
    #     return loss

    def global_sampling(self, x_mel: torch.LongTensor, max_len: int = 384, temperature: float = 1.0):
        """根据给定的旋律 x_mel，自回归地生成伴奏。需要注意一点就是，在当前模型中，frame 和 ticks 是一样的。

        Args:
            x_mel (torch.LongTensor): 输入的旋律数据，形状 [B, S, L]。
            max_len (int): 生成伴奏的最大帧数。
            temperature (float): 采样温度。

        Returns:
            torch.LongTensor: 生成的完整伴奏序列。
        """
        self.eval()  # 切换到评估模式
        batch_size, _, subseq_len = x_mel.shape
        device = x_mel.device

        with torch.no_grad():
            # 1. 预处理旋律并构建初始序列 [SOM, mel, EOM, SOA]
            x_mel_proc = self.preprocess(x_mel, torch.zeros(batch_size, device=device, dtype=torch.long))

            som_token = torch.full((batch_size, 1, subseq_len), SOM_TOKEN, device=device, dtype=torch.long)
            eom_token = torch.full((batch_size, 1, subseq_len), EOM_TOKEN, device=device, dtype=torch.long)
            soa_token = torch.full((batch_size, 1, subseq_len), SOA_TOKEN, device=device, dtype=torch.long)

            prompt_seq = torch.cat([som_token, x_mel_proc, eom_token, soa_token], dim=1)

            # 2. 为初始序列构建 token_type_ids
            mel_len = 1 + x_mel_proc.shape[1] + 1
            acc_len = 1
            prompt_types = torch.cat(
                [
                    torch.zeros((batch_size, mel_len), device=device, dtype=torch.long),
                    torch.ones((batch_size, acc_len), device=device, dtype=torch.long),
                ],
                dim=1,
            )

            # 3. 编码初始序列，得到上下文 h
            # 注意：这里我们不使用 forward，而是手动执行编码步骤来获取 h
            local_token_type_ids = prompt_types.unsqueeze(-1).expand(
                prompt_seq.shape[0], prompt_seq.shape[1], subseq_len
            )
            sos_type = prompt_types.unsqueeze(-1).expand(prompt_seq.shape[0], prompt_seq.shape[1], 1)
            local_token_type_ids = torch.cat([sos_type, local_token_type_ids], dim=-1)

            h, _ = self.local_encode(prompt_seq, local_token_type_ids)
            h = h.view(batch_size, -1, self.hidden_size)  # [B, S_prompt, H]

            # 准备全局编码器的输入
            sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
            h_global_input = torch.cat([sos, h], dim=1)

            generated_acc = []
            eos_triggered = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # 4. 自回归生成循环
            for i in range(max_len):
                # a. 全局编码
                global_token_types_input = torch.cat(
                    [torch.zeros(batch_size, 1, device=device, dtype=torch.long), prompt_types], dim=1
                )

                h_out = self.model(
                    h_global_input,
                    attention_mask=self.buffered_future_mask(h_global_input),
                    token_type_ids=global_token_types_input,
                )[0]

                # b. 局部采样，只使用最后一个时间步的输出
                next_acc_frame = self.local_sampling(h_out[:, -1], max_subseq_len=subseq_len, temperature=temperature)

                # 如果某个样本已生成 EOS，则后续用 PAD 填充
                next_acc_frame[eos_triggered] = PAD_TOKEN
                eos_triggered = eos_triggered | (next_acc_frame == EOS_TOKEN).any(dim=-1)

                generated_acc.append(next_acc_frame)
                if torch.all(eos_triggered):
                    break

                # c. 准备下一轮的输入
                next_acc_frame_proc = next_acc_frame.unsqueeze(1)  # [B, 1, L]

                # d. 编码新生成的帧
                next_h, _ = self.local_encode(
                    next_acc_frame_proc, torch.ones((batch_size, 1, subseq_len + 1), device=device, dtype=torch.long)
                )
                next_h = next_h.view(batch_size, 1, -1)

                # e. 更新全局上下文 h 和 token_types
                h_global_input = torch.cat([h_global_input, next_h], dim=1)
                prompt_types = torch.cat(
                    [prompt_types, torch.ones((batch_size, 1), device=device, dtype=torch.long)], dim=1
                )

        return (
            torch.cat(generated_acc, dim=1) if generated_acc else torch.empty(batch_size, 0, subseq_len, device=device)
        )

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

    def save(self, data, filename_base, extension, save_dir, **kwargs):
        return super().save(data, filename_base, extension, save_dir, **kwargs)

    def loss(self, x_mel, x_acc, pitch_shift):
        """计算模型的交叉熵损失，采用新的序列构建和数据增强策略

        该方法实现了新的序列构建和损失计算逻辑：
        1. 随机选择一个切分点 `t` 作为数据增强。
        2. 构建序列 `x = [SOM, mel[0:t], EOM, SOA, acc[t:], EOA]`。
        3. 构建对应的 `token_type_ids` (0 for mel part, 1 for acc part)。
        4. 构建目标 `labels`，其中只有伴奏部分 `acc[t:]` 和 `EOA` 有值，
           其余部分（旋律、SOM、EOM、SOA）被 `ignore_index` (PAD_TOKEN) 掩码。
        5. 调用 `forward` 得到 `logits` 并计算交叉熵损失。

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
        device = x_mel_proc.device

        # 数据增强：在整个 batch 中使用同一个随机切分点 t
        # 这样做可以保持 batch 内所有序列长度一致，避免了复杂的填充操作。
        cut_start = self.config.cut_point_start
        # 如果未指定结束点，则默认为序列最大长度。
        # torch.randint(low, high) 生成的 t 在 [low, high-1] 之间。
        # 为确保 acc_part 至少有一个 token (即 t_max = seq_len - 1)，
        # effective_cut_end (即 high) 必须是 seq_len。
        effective_cut_end = self.config.cut_point_end if self.config.cut_point_end is not None else seq_len
        # 确保切分点范围有效
        # 严格验证参数范围，如果无效则抛出异常
        if not (0 <= cut_start < seq_len):
            raise ValueError(f"配置错误: 'cut_point_start' ({cut_start}) 必须在 [0, {seq_len - 1}] 范围内。")

        if not (0 < effective_cut_end <= seq_len):
            raise ValueError(f"配置错误: 'cut_point_end' ({effective_cut_end}) 必须在 (0, {seq_len - 1}] 范围内。")

        if cut_start >= effective_cut_end:
            raise ValueError(
                f"配置错误: 'cut_point_start' ({cut_start}) 必须严格小于 'cut_point_end' ({effective_cut_end})。"
            )
        # 在验证通过的范围内随机选择切分点 t
        t = torch.randint(cut_start, effective_cut_end, (1,), device=device).item()

        # 3. 构建输入序列 x 和 token_type_ids
        # 准备特殊 token (形状为 [B, 1, L])
        som_token = torch.full((batch_size, 1, subseq_len), SOM_TOKEN, device=device, dtype=torch.long)
        eom_token = torch.full((batch_size, 1, subseq_len), EOM_TOKEN, device=device, dtype=torch.long)
        soa_token = torch.full((batch_size, 1, subseq_len), SOA_TOKEN, device=device, dtype=torch.long)
        eoa_token = torch.full((batch_size, 1, subseq_len), EOA_TOKEN, device=device, dtype=torch.long)

        # 提取旋律和伴奏部分
        mel_part = x_mel_proc[:, :t, :]  # [0, t-1]
        acc_part = x_acc_proc[:, t:, :]  # [t, end]

        # 拼接成最终输入序列 x: [SOM, mel, EOM, SOA, acc, EOA]
        x = torch.cat([som_token, mel_part, eom_token, soa_token, acc_part, eoa_token], dim=1)

        # 构建对应的 token_type_ids (0 for mel, 1 for acc)
        # 旋律部分长度: 1 (SOM) + t (mel) + 1 (EOM) = t + 2
        mel_type_len = t + 2
        # 伴奏部分长度: 1 (SOA) + (seq_len - t) (acc) + 1 (EOA) = seq_len - t + 2
        acc_type_len = seq_len - t + 2

        mel_types = torch.zeros((batch_size, mel_type_len), device=device, dtype=torch.long)
        acc_types = torch.ones((batch_size, acc_type_len), device=device, dtype=torch.long)
        token_types = torch.cat([mel_types, acc_types], dim=1)

        # 构建目标序列（与输入相同）
        labels = x.clone()

        # 掩码掉所有不需要预测的部分 (设置为 PAD_TOKEN, 即 ignore_index)
        # 我们需要预测从 SOA 开始的所有 token，所以掩码掉 SOA 之前的所有内容
        # 掩码长度 = 1 (SOM) + t (mel) + 1 (EOM) + 1 (SOA) = t + 3
        mask_len = t + 3
        labels[:, :mask_len, :] = PAD_TOKEN

        # 前向传播得到 logits
        # forward 方法接收 x 和 token_types，并执行自回归预测
        logits = self(x, token_types)

        # 计算交叉熵损失
        return F.cross_entropy(logits.view(-1, N_TOKENS), labels.view(-1), ignore_index=PAD_TOKEN)
