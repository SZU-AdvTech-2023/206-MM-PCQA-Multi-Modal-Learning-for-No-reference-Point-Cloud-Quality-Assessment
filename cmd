
CUDA_VISIBLE_DEVICES=0 python -u train.py \
--learning_rate 0.00005 \
--model MM_PCQA \
--batch_size  8 \
--database SJTU  \
--G:\database\SJTU\sjtu_projections path_to_sjtu_projections/ \
--G:\database\SJTU\sjtu_patch_2048 path_to_sjtu_patch_2048/ \
--loss l2rank \
--num_epochs 50 \
--k_fold_num 9 \
>> logs/sjtu_mmpcqa.log