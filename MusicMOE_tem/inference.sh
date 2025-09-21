# python cp_transformer_inference.py /ephemeral/ckpt/cp_transformer_smalldense_small_batch_20/cp_transformer_smalldense_small_batch_20.epoch=00.val_loss=0.52616.ckpt  false false
# Usage: python cp_transformer_inference.py <model_path> <large> <moe>
python cp_transformer_inference.py /mnt/jiahe_new_volume/cp_transformer_moe_large_batch_10_expert_2_topk_1_moe_hidden4096_swiglu_new/cp_transformer_moe_large_batch_10_expert_2_topk_1_moe_hidden4096_swiglu_new.epoch=00.val_loss=0.52240.ckpt true true 2 1
