#!/bin/bash
cd ..
# activate Conda environment
source activate ../as_py38

# MRR and BLEU weighted RR metric, --evalFile : path to the test/generated file, --top: consider tok k suggestions
python -u mrr_rrbleu.py \
    --evalFile 'outputs/mpc/ms_v4_seen_cand_with_trie.tsv' 'outputs/mpc/ms_v4_unseen_cand_with_prefix_dict.tsv' \
    --top 8 \
    --group1 1 2 3 \
    --group2 4 5 6 7 8 \
    --th_prefix 8 \

# APPG metric, --evalFile : path to the test/generated file, --top: consider tok k suggestions, --verbose: enable instance score
python -u appg_our.py \
    --evalFile 'outputs/mpc/ms_v4_seen_cand_with_trie.tsv' 'outputs/mpc/ms_v4_unseen_cand_with_prefix_dict.tsv' \
    --top 8 \
    --group1 1 2 3 \
    --group2 4 5 6 7 8 \
    --th_prefix 8 \

# Multilingaul BLEU and ROUGE metrics, --evalFile : path to the test/generated file, --lang: text languge name
python -u mbleu_mrouge.py \
    --evalFile 'outputs/mpc/ms_v4_seen_cand_with_trie.tsv' 'outputs/mpc/ms_v4_unseen_cand_with_prefix_dict.tsv' \
    --lang english \
    --group1 1 2 3 \
    --group2 4 5 6 7 8 \
    --th_prefix 8 \

#    --report_with_prefix_length \
#    --evalFile 'outputs/exp_33/pred_both_as_ts_yes_trie_ms_v4_test.tsv' 'outputs/exp_33/ms_v4_unseen_cand_with_suffix_dict.tsv'\