#!/bin/bash

echo "--- 启动实验序列 ---"

# ===============================================
# 实验 1: 运行配置 A 的训练
# ===============================================
echo ""
echo "--- 正在运行实验 1 (配置 A) ---"

# 切换到第一个实验所需的目录
cd /home/ubuntu/stanleyz/Training-Framework

# 运行第一个训练命令
# 注意：即使这个命令失败，脚本也会继续执行后续命令
uv run python -m src.model.baseline.m2a_transformer_inference --model_path "/home/ubuntu/stanleyz/shared_models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt" --prompt_len 75 --model_size 0.12B

echo "实验 1.1 执行完毕 (无论成功或失败)。"

uv run python -m src.model.baseline.m2a_transformer_inference --model_path "/home/ubuntu/stanleyz/shared_models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt" --prompt_len 128 --model_size 0.12B

echo "实验 1.2 执行完毕 (无论成功或失败)。"

uv run python -m src.model.baseline.m2a_transformer_inference --model_path "/home/ubuntu/stanleyz/shared_models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt" --prompt_len 150 --model_size 0.12B

echo "实验 1.3 执行完毕 (无论成功或失败)。"

# ===============================================
# 实验 2: 运行配置 B 的训练
# ===============================================
echo ""
echo "--- 正在运行实验 2 (配置 B) ---"

# 切换到第二个实验所需的目录
# cd /path/to/experiment_B_repo 

# 运行第二个训练命令
uv run python -m src.model.baseline.m2a_transformer_inference --model_path "/home/ubuntu/stanleyz/shared_models/ModelAiraDeduped0.25/1.5.5/epoch=64-val_loss=0.73.ckpt" --prompt_len 75 --model_size 0.25B

echo "实验 2.1 执行完毕 (无论成功或失败)。"

uv run python -m src.model.baseline.m2a_transformer_inference --model_path "/home/ubuntu/stanleyz/shared_models/ModelAiraDeduped0.25/1.5.5/epoch=64-val_loss=0.73.ckpt" --prompt_len 128 --model_size 0.25B

echo "实验 2.2 执行完毕 (无论成功或失败)。"

uv run python -m src.model.baseline.m2a_transformer_inference --model_path "/home/ubuntu/stanleyz/shared_models/ModelAiraDeduped0.25/1.5.5/epoch=64-val_loss=0.73.ckpt" --prompt_len 150 --model_size 0.25B

echo "实验 2.3 执行完毕 (无论成功或失败)。"

# ===============================================
# 实验 3: 运行不同模型的训练
# ===============================================
echo ""
echo "--- 正在运行实验 3 (新模型) ---"

# 切换到第三个实验所需的目录
# cd /path/to/new_model_repo 

# 运行第三个训练命令
uv run python -m src.model.old_pt_m2a_transformer_with_att.inference --prompt_len 75

echo "实验 3.1 执行完毕 (无论成功或失败)。"

uv run python -m src.model.old_pt_m2a_transformer_with_att.inference --prompt_len 128

echo "实验 3.2 执行完毕 (无论成功或失败)。"

uv run python -m src.model.old_pt_m2a_transformer_with_att.inference --prompt_len 150

echo "实验 3.3 执行完毕 (无论成功或失败)。"

# ===============================================
# 最终清理或返回主目录
# ===============================================
echo ""
echo "所有实验任务已完成。"
cd $HOME 
echo "已返回主目录。"

exit 0