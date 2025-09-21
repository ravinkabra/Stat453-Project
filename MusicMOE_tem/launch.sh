export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/ephemeral/tmp
torchrun --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    cp_transformer.py   --batch-size 10 --model-size large --moe --expert_num 2  --topk 1
    