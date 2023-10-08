import os
from random import shuffle
import sys
import argparse
import json
import numpy as np
from numpy.lib.npyio import save
import copy
from tqdm import tqdm

import opts as opts

import torch
import torch.nn as nn
from torch.optim import AdamW

from transformers import (
    set_seed,
    Trainer,
    TrainingArguments,
)
from utils import (
    freeze_embeds, 
    freeze_params,
    assert_all_frozen,
)
from model import load_model_tokenizer
from datasets import load_the_dataset
from mrr_rrbleu import calc_mrr_bleu
from appg_our import calculate_appg
from mbleu_mrouge import calculate_bleu, calculate_bleu_score


#import apex
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

#import socket
#os.environ['MASTER_ADDR'] = socket.gethostbyname(socket.gethostname())
#os.environ['TOKENIZERS_PARALLELISM'] = 'False'

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    opts.add_md_help_argument(parser)
    opts.train_opts(parser)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        force=True,
    )
    logger.info("Training/evaluation parameters %s", args)

    #Create and write config file in output directory
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f_out:
        json.dump(vars(args), f_out)
        f_out.close()
    
    #Set the random seed for deterministic nature
    set_seed(args.seed)

    #loading model and tokenizer
    model, tokenizer = load_model_tokenizer(args)

    #Freezing model components 
    logger.info("Total Number of parameters: %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.freeze_embeds:
        freeze_embeds(model)
    if args.freeze_encoder:
        #freeze_params(model.get_encoder())
        #assert_all_frozen(model.get_encoder())
        for i, m in enumerate(model.encoder.block): 
            if i > 5:
                for parameter in m.parameters():
                    parameter.requires_grad = False 
    if args.freeze_embeds_and_decoder:
        if args.model_type == 't5':
            #freeze_embeds(model)
            for i, m in enumerate(model.decoder.block): 
                if i < 6:
                    for parameter in m.parameters():
                        parameter.requires_grad = False 
        elif args.model_type == 'bart':
            for i, dec_layer in enumerate(model.get_decoder().layers):
                if i < 6:
                    for dec_param in dec_layer.parameters():
                        dec_param.requires_grad = False
    logger.info("Total Number of parameters after FREEZE (if any): %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    #print train and validation data size:
    if args.train_data: train_dataset = load_the_dataset(args, args.train_data, tokenizer, "train")
    if args.val_data: val_dataset = load_the_dataset(args, args.val_data, tokenizer, "val") 
    if args.train_data: logger.info("Training Data Size : %s", len(train_dataset))
    if args.val_data: logger.info("Validation Data Size : %s", len(val_dataset))


    # Save training 
    if args.do_train:
        with open(os.path.join(args.output_dir, "debug_train_examples.txt"), 'w', encoding='utf8') as debug_fp:
            for example in train_dataset[:50]:
                example_instance = {}
                example_instance['input'] = example['input_ids'].tolist()
                example_instance['attention_mask'] = example['attention_mask'].tolist()
                example_instance['labels'] = example['labels'].tolist()
                example_instance['input_text'] = tokenizer.decode(example['input_ids'], skip_special_tokens=False)
                example['labels'].masked_fill_(example['labels'] == -100, tokenizer.pad_token_id)
                example_instance['label_text'] = tokenizer.decode(example['labels'], skip_special_tokens=False)
                debug_fp.write(json.dumps(example_instance) + '\n')
            debug_fp.close()
  
        hf_args = TrainingArguments(
            output_dir = args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size, 
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            warmup_ratio=args.warmup_ratio,
            learning_rate= args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            dataloader_num_workers=args.num_workers,
            fp16=False,
            #logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=args.logging_steps,
            logging_first_step=True,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            seed=args.seed,
            sharded_ddp=True,
            report_to="tensorboard",
            local_rank=args.local_rank,
            overwrite_output_dir="False",
            adafactor=False,
            #resume_from_checkpoint=args.resume_from_checkpoint,
        )

        trainer = Trainer(
            model=model, 
            args=hf_args,
            train_dataset=train_dataset, 
            eval_dataset=val_dataset, 
            tokenizer=tokenizer,
        )

        trainer.train()
        trainer.save_model()

    if args.do_test:
        #cheking dataset size
        test_dataset = load_the_dataset(args, args.test_data, tokenizer, "test")
        logger.info("Total Number of Test Files are : %s", len(test_dataset))
        for testFileObject, testFileName in test_dataset:
            logger.info("Size of %s File is : %d", testFileName, len(testFileObject))

        #tranfering model and tensor to Device
        device= 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        #Evaluation score reporting function
        def report_eval_score(args, _prefixes, _predictions, _references, report_with_prefix_length):
            evaluation_score={}
            top_n_predictions = [ instance[:args.top_n_for_eval] for instance in _predictions]
            assert len(_predictions)  == len(_references) == len(_prefixes), "number of  predictions, refereces and prefixes should be same"
            #logger.info("*****  Running MRR and RR_BLEU Evalution Script *****")
            mrr_rrbleu_score = calc_mrr_bleu(args, gt_data=_references, pred_data=_predictions, prefix_data=_prefixes, report_with_prefix_length=args.report_with_prefix_length)
            logger.info(mrr_rrbleu_score)
            #logger.info("*****  Running APPG Evalution Script *****")
            appg_score = calculate_appg(args, prefix_data=_prefixes, gt_data=_references, pred_data=_predictions, report_with_prefix_length=args.report_with_prefix_length, verboseFlag='False')
            #logger.info("*****  Running BLEU (English) Evalution Script *****")
            one_preds= [ pred[0] for pred in _predictions] 
            bleu_score = calculate_bleu_score(args, one_preds, _references, _prefixes,  report_with_prefix_length, bleu_lang='english')
            #logger.info("*"*100)
            evaluation_score["MRR"] = mrr_rrbleu_score['MRR']
            evaluation_score["RR_BLEU"] = mrr_rrbleu_score['RR_bleu(up-to 4-gram)']
            evaluation_score["APPG"] = appg_score
            evaluation_score["BLEU"] = str(bleu_score).split(" = ")[1:]
            return evaluation_score 
    
        all_evalution_score=[]
        all_eval_score_file_name = ''
        all_references, all_predictions, all_prefixes = [], [], []
        for test_data_object, testFileName in test_dataset:
            logger.info("Running Generation Script for File: %s", testFileName)
            test_datasets= torch.utils.data.DataLoader(
                dataset=test_data_object,
                batch_size=args.test_batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                num_workers=args.num_workers
            )
            save_pred_file_name= "pred_"+str(testFileName.strip().split('/')[-1])
            all_eval_score_file_name = all_eval_score_file_name+"_"+str(save_pred_file_name.split('.')[0])
            with torch.no_grad(), open(os.path.join(args.output_dir, save_pred_file_name), 'w', encoding='utf8') as f_gen:
                all_reference, all_prediction, all_prefix = [], [], []
                for test_idx, test_instance in tqdm(enumerate(test_datasets), total=len(test_datasets)):
                    outputs = model.generate(
                        input_ids=test_instance['input_ids'].to(device),
                        attention_mask=test_instance['attention_mask'].to(device),
                        max_length=args.max_generated_seq_len,
                        min_length=args.min_generated_seq_len,
                        num_return_sequences=args.num_of_return_seq,
                        early_stopping=args.early_stopping,
                        pad_token_id=tokenizer.pad_token_id,
                        num_beams=args.beam_size,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        length_penalty=args.length_penalty,
                    ) 
                    batch_predictions=tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    batch_references =tokenizer.batch_decode(test_instance['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    batch_predictions = [batch_predictions[i:i + args.num_of_return_seq] for i in range(0, len(batch_predictions), args.num_of_return_seq)]          
                    assert len(batch_predictions) == len(batch_references) == len(test_instance['prefix']) == len(test_instance['session']), "Predictions and  reference lists are different size"
                    
                    for current_idex, (ref, pred, prefix, session) in enumerate(zip(batch_references, batch_predictions, test_instance['prefix'], test_instance['session'])):
                        f_gen.write(json.dumps({"instance_id": test_idx*args.test_batch_size + (current_idex +1), "prefix": prefix, "session": session, "reference": ref, "predictions":pred}, ensure_ascii=False) + "\n")
                    all_reference.extend(batch_references)
                    all_prediction.extend(batch_predictions)
                    all_prefix.extend(test_instance['prefix'])
                logger.info("Running Evaluation Script for File: %s", testFileName)
                dict_eval_score = report_eval_score(args, all_prefix, all_prediction, all_reference, args.report_with_prefix_length)
                dict_eval_score.update({"FileName" : testFileName, "FileSize" : len(all_reference)})
                all_evalution_score.append(dict_eval_score)
                all_references.extend(all_reference)
                all_predictions.extend(all_prediction)
                all_prefixes.extend(all_prefix)
            f_gen.close()
        logger.info("Running Evaluation Script for all the file")
        overall_eval_score = report_eval_score(args, all_prefixes, all_predictions, all_references, args.report_with_prefix_length)
        overall_eval_score.update({"FileName" : "All Files", "FileSize" : len(all_references)})
        all_evalution_score.append(overall_eval_score)
        logger.info("Saving All Evalution Scores in || %s || File", 'all_score'+str(all_eval_score_file_name)+'.txt')
        with  open(os.path.join(args.output_dir, 'all_score'+str(all_eval_score_file_name)+'.txt'), 'w') as feval:
            for item in all_evalution_score:
                feval.write(json.dumps(item)+ "\n")
        feval.close()
        logger.info("********Generation And Evalution are Completed!!!********")

if __name__ == '__main__':
    main()



