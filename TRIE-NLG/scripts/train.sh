#!/bin/bash
cd ..
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source activate ../as_py38

export NCCL_DEBUG=INFO
export NPROC_PER_NODE=8
export PARENT=`/bin/hostname -s`
export MPORT=12343

#input and output directories
export BASE_DIR='.'
export input_dir="../../as_data/"
export output_dir="outputs/exp_65"

#model details
export model_type="bart"  #t5
export model_chkpt="facebook/bart-base"  #t5-base
export cache_dir='../cache_dir'

python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr="$PARENT" \
    --master_port="$MPORT" "train.py" \
    --train_data ${input_dir}/both_as_ts/both_as_ts_ms_v4_train_new.tsv  \
    --val_data ${input_dir}/both_as_ts/both_as_ts_ms_v4_val_new.tsv  \
    --unseen_dict ${input_dir}/database/ms_v4_suffix_MPC.json \
    --output_dir ${output_dir} \
    --model_type $model_type \
    --model_chkpt $model_chkpt \
    --max_source_length 220 \
    --max_target_length 32 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --lr_scheduler_type "linear" \
    --logging_steps 100 \
    --save_steps 5000 \
    --eval_steps 5000 \
    --cache_dir ${cache_dir} \
    --read_n_data_obj -1 \
    --do_train \
    --enable_trie_context \
    --synth_contx_for_prefix_len 1000 \