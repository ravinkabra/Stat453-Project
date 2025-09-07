from typing import Optional, Any

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
        self.hidden_size = model_schema.hidden_size
        self.num_layers = model_schema.num_layers
        self.num_attention_heads = model_schema.num_attention_heads
        self.intermediate_size = model_schema.intermediate_size
        self.local_model_num_layers = model_schema.local_model_num_layers
        self.local_model_num_attention_heads = (
            model_schema.local_model_num_attention_heads
        )
        self.local_model_intermediate_size = model_schema.local_model_intermediate_size

        # Lazy import of transformers RoFormer to avoid heavy import at module load
        from transformers.models.roformer.modeling_roformer import (
            RoFormerConfig,
            RoFormerEncoder,
        )

        main_roformer_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.model = RoFormerEncoder(main_roformer_config)
        local_encoder_config = local_decoder_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.local_embedding = nn.Embedding(N_TOKENS, self.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        with torch.no_grad():
            self.token_type_embeddings.weight.mul_(2.0)

        self.local_encoder = RoFormerEncoder(local_encoder_config)
        self.local_decoder = RoFormerEncoder(local_decoder_config)
        self.final_decoder = nn.Linear(self.hidden_size, N_TOKENS)
        self.global_sos = nn.Parameter(torch.randn(self.hidden_size))
        self._future_mask = torch.empty(0)

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
        batch_size, seq_len, subseq_len = x.shape
        x = x.view(-1, subseq_len)

        # prepend SOS: 在每个子序列头部插入特殊开始符
        x = torch.cat(
            [
                torch.full(
                    (x.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=x.device
                ),
                x,
            ],
            dim=-1,
        )

        # mask 用于告诉 encoder 哪些位置是 PAD
        mask = x != PAD_TOKEN
        word_emb = self.local_embedding(x)

        # token type embedding 用于指示 frame 类型（比如 mel/acc）
        type_emb = self.token_type_embeddings(token_type_ids)
        type_emb = type_emb.view(batch_size * seq_len, word_emb.shape[1], -1)

        emb = word_emb + type_emb
        h = self.local_encoder(emb, encoder_attention_mask=mask)[0]

        # 返回每段序列的聚合向量（第 0 个位置为 SOS 的输出）以及用于解码器的 embeddings
        return h[:, 0], emb[:, :-1]

    def local_decode(self, h: torch.Tensor, emb: torch.Tensor):
        """局部解码器：将局部隐藏表示与 embeddings 拼接并解码为 token logits。

        - h: 局部聚合表示，形状 [batch*seq, hidden]
        - emb: 局部 embeddings，形状 [batch*seq, subseq_len, hidden]
        返回 logits，形状与解码器输出对应（用于后续 softmax/采样）。
        """
        batch_size, subseq_len, _ = emb.shape
        # 将 h 放回序列头位置并与 emb 拼接（保持长度一致）
        h = h.view(batch_size, 1, -1)
        emb = torch.cat([h, emb[:, 1:]], dim=1)
        h = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
        return self.final_decoder(h)

    def local_sampling(
        self, h: torch.Tensor, max_subseq_len: int = 32, temperature: float = 1.0
    ):
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

        for i in range(max_subseq_len):
            h_ = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[
                0
            ]
            if temperature == 0:
                # 贪心解码
                p = F.one_hot(self.final_decoder(h_).argmax(dim=-1), N_TOKENS).float()
            else:
                # 概率采样
                p = F.softmax(self.final_decoder(h_[:, -1]) / temperature, dim=-1)
            y_next = torch.multinomial(p, 1)
            # 已触发 EOS 的位置用 PAD 填充（不再生成有效 token）
            y_next[eos_triggered, :] = PAD_TOKEN
            eos_triggered = eos_triggered | (y_next.squeeze(1) == EOS_TOKEN)
            y = torch.cat([y, y_next], dim=1)
            if torch.all(eos_triggered):
                break

            # 更新 emb，用新生成的 token 的 embedding 以及 type embedding
            emb = torch.cat(
                [
                    emb,
                    self.local_embedding(y_next)
                    + self.token_type_embeddings(torch.ones_like(y_next)),
                ],
                dim=1,
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
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

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
        token_type_ids = (
            frame_type.unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, seq_len, subseq_len)
        )
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)

        # 局部编码
        h, emb = self.local_encode(x, token_type_ids)
        h = h.view(batch_size, seq_len, -1)

        # 在全局序列前加入 SOS 并把前一时刻的隐藏拼接进来（自回归偏移）
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)

        # 全局 encoder（使用 buffered_future_mask 保证自回归）
        h = self.model(
            h, attention_mask=self.buffered_future_mask(h), interleave_pos=True
        )[0]
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
        x_processed[:, :, :, 1] = (
            x[:, :, :, 1]
            + (x[:, :, :, 2]) * 128
            + 2
            + pitch_shift[:, None, None] * is_not_drum
        )
        x_processed[pad_indices] = PAD_TOKEN
        x_processed[:, :, :, 0][eos_indices] = EOS_TOKEN

        if y is None:
            return x_processed.view(batch_size, seq_length, subseq_length // 3 * 2)
        else:
            # 如果有目标 y，则对 y 做相同的处理，type 字段设置为 1（表示目标）
            batch_size_y, seq_length_y, subseq_length_y = y.shape
            y = y.long().view(batch_size_y, seq_length_y, subseq_length_y // 3, 3)
            y_processed = torch.zeros(
                batch_size_y,
                seq_length_y,
                subseq_length_y // 3,
                2,
                dtype=torch.long,
                device=y.device,
            )
            pad_indices_y = y[:, :, :, 1] == 255
            eos_indices_y = y[:, :, :, 0] == 254
            is_not_drum_y = y[:, :, :, 0] != 127
            y_processed[:, :, :, 0] = 1
            y_processed[:, :, :, 1] = (
                y[:, :, :, 1]
                + (y[:, :, :, 2]) * 128
                + 2
                + pitch_shift[:, None, None] * is_not_drum_y
            )
            y_processed[pad_indices_y] = PAD_TOKEN
            y_processed[:, :, :, 0][eos_indices_y] = EOS_TOKEN

            return x_processed.view(
                batch_size, seq_length, subseq_length // 3 * 2
            ), y_processed.view(batch_size_y, seq_length_y, subseq_length_y // 3 * 2)

    def loss(
        self,
        x_mel: torch.LongTensor,
        x_acc: torch.LongTensor,
        pitch_shift: torch.LongTensor,
    ):
        """计算训练时的交叉熵损失。

        过程：
            1. 对 mel/acc 进行 preprocess 合并为模型输入 x
            2. 构造 x_target（将 mel 的位置掩为 PAD，因为 mel 是输入而非目标）
            3. 前向得到 logits y，然后对 y 与 x_target 做交叉熵（忽略 PAD）
        返回标量 loss。
        """
        x_mel, x_acc = self.preprocess(x_mel, pitch_shift, y=x_acc)
        batch_size, seq_len, subseq_len = x_mel.shape
        stacked = torch.stack([x_acc, x_mel], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)

        # 构造目标：将 mel 位置（偶数/奇数交替）设为 PAD（不作为目标）
        x_target = x.clone()
        idx = torch.arange(seq_len * 2, device=x.device)
        mel_mask = (idx % 2 == 1).unsqueeze(0).unsqueeze(-1)
        mel_mask = mel_mask.expand(batch_size, seq_len * 2, subseq_len)
        x_target[mel_mask] = PAD_TOKEN

        # 前向计算 logits
        y = self(x)

        # 计算交叉熵，忽略 PAD
        return F.cross_entropy(
            y.view(-1, N_TOKENS), x_target.view(-1), ignore_index=PAD_TOKEN
        )

    def training_step(self, batch: OldPtModelInput, batch_idx: int):
        # Expect batch to have attributes mel_data, acc_data, pitch_shift
        mel = batch.mel_data
        acc = batch.acc_data
        pitch_shift = batch.pitch_shift
        loss = self.loss(mel, acc, pitch_shift)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        scheduler = self.lr_schedulers()
        if scheduler:
            scheduler.step()
            self.log(
                "training/lr",
                scheduler.get_last_lr()[0],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch: OldPtModelInput, batch_idx: int):
        mel = batch.mel_data
        acc = batch.acc_data
        pitch_shift = batch.pitch_shift
        loss = self.loss(mel, acc, pitch_shift)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

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
        return type(batch)(
            **{
                k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in batch.__dict__.items()
            }
        )
