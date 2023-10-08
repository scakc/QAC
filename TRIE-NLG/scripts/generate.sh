#!/bin/bash
cd ..
#stting GPUs to use
export CUDA_VISIBLE_DEVICES=0
#conda enviourment activate
source activate ../as_py38

# misc. settings
export seed=1234

# model settings
export model_type="bart" #bart #t5
export model_chkpt="outputs/exp_64/checkpoint-118000"

#input and output directories
export input_dir="../../as_data"
export output_dir="outputs/exp_64"
export cache_dir='../cache_dir'

python train.py \
    --test_data ${input_dir}/test/both_as_ts_yes_trie_ms_v4_test_new.tsv  ${input_dir}/test/both_as_ts_no_trie_ms_v4_test_new.tsv \
    --unseen_dict ${input_dir}/database/ms_v4_suffix_MPC.json \
    --output_dir ${output_dir} \
    --model_type ${model_type} \
    --model_chkpt ${model_chkpt} \
    --test_batch_size 300 \
    --max_source_length 220  \
    --max_target_length 32 \
    --length_penalty 0.6 \
    --beam_size 8 \
    --early_stopping \
    --num_of_return_seq 8 \
    --max_generated_seq_len 16 \
    --min_generated_seq_len 1 \
    --cache_dir ${cache_dir} \
    --do_test \
    --read_n_data_obj -1 \
    --top_n_for_eval 8 \
    --enable_trie_context \
    --synth_contx_for_prefix_len 1000 \