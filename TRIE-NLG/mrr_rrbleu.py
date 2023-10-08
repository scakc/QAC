import argparse
import logging
from re import L
import numpy as np
from pathlib import Path
import json
import sys
import os
import csv
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm
import pandas as pd

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

def calc_mrr_bleu(args, gt_data, pred_data, prefix_data, report_with_prefix_length):
    args.group1 = [ int(item) for item in args.group1]
    args.group2 = [ int(item) for item in args.group2]
    all_mrr,all_bleu = [],[]
    mrr_13, mrr_46, mrr_g6 = [], [], []
    rrbl_13, rrbl_46, rrbl_g6 = [], [], [] 
    for j, (gt_line, preds, prefix) in tqdm(enumerate(zip(gt_data, pred_data, prefix_data)), total=len(gt_data)):
      if len(preds) != 0 and len(gt_line) !=0 :
        gt_label = gt_line.lower()  # ground truth next_query
        preds = [p.lower() for p in preds]
        mrr = 0.0
        for i, curr_pred in enumerate(preds):
            if gt_label == curr_pred:
                mrr = 1.0 / (i + 1)
                break
        all_mrr.append(mrr)

        wgtd_bleu_score = 0.0
        normalizer = 0.0
        for i, curr_pred in enumerate(preds):
            wgtd_bleu_score += (
                bleu_score.sentence_bleu(
                    [gt_label.split()],
                    curr_pred.split(),
                    smoothing_function=SmoothingFunction().method1,
                )
                / (i + 1)
            )
            normalizer += 1.0 / (i + 1)
        wgtd_bleu_score = wgtd_bleu_score / normalizer if normalizer > 0 else wgtd_bleu_score
        all_bleu.append(wgtd_bleu_score)
        
        if report_with_prefix_length:
            prfix_len = len(prefix.strip())
            if prfix_len in args.group1:
               mrr_13.append(mrr)
               rrbl_13.append(wgtd_bleu_score)
            elif prfix_len in args.group2:
               mrr_46.append(mrr)
               rrbl_46.append(wgtd_bleu_score)
            else:
               mrr_g6.append(mrr)
               rrbl_g6.append(wgtd_bleu_score)

    if report_with_prefix_length:    
        assert len(all_mrr) == len(mrr_13 + mrr_46 + mrr_g6), 'All prefix and distributed prfix list length for MRR should be same'
        assert len(all_bleu) == len(rrbl_13 + rrbl_46 + rrbl_g6), 'All prefix and distributed prfix list length for RR_BLEU should be same'

        LOGGER.info("Evalution Scores for Prefix Length Range {} are: ".format(str(args.group1)))
        LOGGER.info({"MRR": np.mean(mrr_13), "RR_bleu(up-to 4-gram)": np.mean(rrbl_13), "num_samples": len(mrr_13)})
        LOGGER.info("Evalution Scores for Prefix Length Range {} are: ".format(str(args.group2)))
        LOGGER.info({"MRR": np.mean(mrr_46), "RR_bleu(up-to 4-gram)": np.mean(rrbl_46), "num_samples": len(mrr_46)})
        LOGGER.info("Evalution Scores for Prefix Length Range > {} are: ".format(str(args.th_prefix)))
        LOGGER.info({"MRR": np.mean(mrr_g6), "RR_bleu(up-to 4-gram)": np.mean(rrbl_g6), "num_samples": len(mrr_g6)})

    return {"MRR": np.mean(all_mrr), "RR_bleu(up-to 4-gram)": np.mean(all_bleu), "num_samples": len(all_mrr)}

def main():
    """ Code are partialy taken from https://github.com/amzn/pecos/blob/mainline/examples/qp2q/eval/eval_pred_data.py"""
    parser = argparse.ArgumentParser(description="MRR and RR_bleu metric evaluation script")
    parser.add_argument('--evalFile', action='append', nargs='+', help='path to testing datasets')
    #parser.add_argument("--evalFile", type=str, required=True, help="Path to model genertaed file")
    parser.add_argument("--top",help="top k generation to consider", type=int, default=8)
    parser.add_argument('--report_with_prefix_length', help='report the scores for different prefix length', action='store_true')
    parser.add_argument('--group1', nargs="+", default=[1,2,3])
    parser.add_argument('--group2', nargs="+", default=[4,5,6])
    parser.add_argument("--th_prefix",help="all prefixes greater then lenghts", type=int, default=8)

    args = parser.parse_args()
    LOGGER.info(args)

    evalfiles = [(pd.read_csv(fileName, sep='\t', header=None).values.tolist(), fileName) for fileName in args.evalFile[0]]
    
    all_ref, all_pred, all_prefix= [], [], []
    for read_evalfile, fName in evalfiles:
        ref, pred, prefix= [], [], []
        for instance in read_evalfile:
            ref.append(json.loads(instance[0])['reference'])
            pred.append(json.loads(instance[0])['predictions'])
            prefix.append(json.loads(instance[0])['prefix'])
        pred = [ instance[:args.top] for instance in pred]
        assert len(ref) == len(pred) == len(prefix), 'reference, prediction and prefix list lengths should be same'
        LOGGER.info("Evalution Score for File %s is :", fName)
        results = calc_mrr_bleu(args, gt_data=ref, pred_data=pred, prefix_data=prefix, report_with_prefix_length=args.report_with_prefix_length)
        LOGGER.info(results)
        print("*"*100)
        all_ref.extend(ref)
        all_pred.extend(pred)
        all_prefix.extend(prefix)
    assert len(all_ref) == len(all_pred), 'reference and prediction lengths should be same'
    LOGGER.info("Overall Evalution Score : %s", fName)
    all_results = calc_mrr_bleu(args, gt_data=all_ref, pred_data=all_pred, prefix_data=all_prefix, report_with_prefix_length=args.report_with_prefix_length)
    LOGGER.info(all_results)


if __name__ == "__main__":
    main()