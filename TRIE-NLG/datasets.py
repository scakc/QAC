import os
import glob
import csv
from re import S
from torch.utils import data
import tqdm
import numpy as np
import json

import torch
from torch.utils.data import Dataset

import transformers

def load_the_dataset(args, data_path, tokenizer, mode):
    if mode == 'test':
        files= data_path[0]
        datasets = [(PrepareDataset(args, file, tokenizer, mode), file) for file in files]
        return datasets
    else:
        files = [data_path]
        return PrepareDataset(args, files, tokenizer, mode)

class PrepareDataset(Dataset):
    def __init__(self, args, files, tokenizer, mode):
        self.args = args
        self.enable_trie_context = args.enable_trie_context
        self.files = files
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length
        self.data = []
        if isinstance(self.files, list):
            for file in self.files:
                if self.args.read_n_data_obj != -1: 
                    self.data += list(csv.reader((i.replace('\x00', '').rstrip('\n').lower() for \
                        count, i in enumerate(open(self.files[0], 'r', encoding='utf8')) if \
                            count < args.read_n_data_obj ), delimiter='\t', quoting=csv.QUOTE_NONE))
                else:
                    self.data += list(csv.reader((i.replace('\x00', '').rstrip('\n').lower() for \
                        i in open(self.files[0], 'r', encoding='utf8')), delimiter='\t', quoting=csv.QUOTE_NONE))
        else:
            if self.args.read_n_data_obj != -1: 
                self.data += list(csv.reader((i.replace('\x00', '').rstrip('\n').lower() for \
                    count, i in enumerate(open(self.files, 'r', encoding='utf8')) if \
                        count < args.read_n_data_obj ), delimiter='\t', quoting=csv.QUOTE_NONE))
            else:
                self.data += list(csv.reader((i.replace('\x00', '').rstrip('\n').lower() for \
                    i in open(self.files, 'r', encoding='utf8')), delimiter='\t', quoting=csv.QUOTE_NONE))
    
        with open(args.unseen_dict, 'r') as fp:
            self.unseen_dict = json.load(fp)
    
        if args.print_stats:
            if mode == 'train':
                print("Statistics for TRAIN dataset:")
            elif mode == 'val':
                print("Statistics for VALIDATION dataset:")
            elif mode == 'test':
                print("Statistics for TEST dataset:")
            source_length,  target_length = [], []
            sl_8, sl_16, sl_32, sl_64, sl_128, sl_256 = [],[],[],[],[],[]
            tl_2, tl_4, tl_8, tl_16, tl_32, tl_64 = [],[],[],[],[],[] 
            for i, example in enumerate(self.data):
                session, label, prefix = example[0], example[2], example[1]
                session = ",".join(session.split('||')[:-1])
                source_len = len(self.tokenizer.tokenize(session)) + len(self.tokenizer.tokenize(prefix))
                source_length.append(source_len)
                sl_8.append(int(source_len > 8))
                sl_16.append(int(source_len > 16))
                sl_32.append(int(source_len > 32))
                sl_64.append(int(source_len > 64))
                sl_128.append(int(source_len > 128))
                sl_256.append(int(source_len > 256))

                target_len = len(self.tokenizer.tokenize(label))
                target_length.append(target_len)
                tl_2.append(int(target_len > 2))
                tl_4.append(int(target_len > 4))
                tl_8.append(int(target_len > 8))
                tl_16.append(int(target_len > 16))
                tl_32.append(int(target_len > 32))
                tl_64.append(int(target_len > 64))

            #if i%99000 == 0  and i !=0:
            #print("Current Example :", i)
            print("Maximum Length of Input :", max(source_length))                          
            print("Minimum Length of Input :", min(source_length))
            print("Avarage Length of Input :", sum(source_length)/len(source_length))
            print("% source > 8: ", sum(sl_8)/len(sl_8))
            print("% source > 16: ", sum(sl_16)/len(sl_16))
            print("% source > 32: ", sum(sl_32)/len(sl_32))
            print("% source > 64: ", sum(sl_64)/len(sl_64))
            print("% source > 128: ", sum(sl_128)/len(sl_128))
            print("% source > 256: ", sum(sl_256)/len(sl_256))
            print("+"*100)
            print("Maximum Length of Target :", max(target_length))
            print("Minimum Length of Target :", min(target_length))
            print("Avarage Length of Target :", sum(target_length)/len(target_length))
            print("% target > 2: ", sum(tl_2)/len(tl_2))
            print("% target > 4: ", sum(tl_4)/len(tl_4))
            print("% target > 8: ", sum(tl_8)/len(tl_8))
            print("% target > 16: ", sum(tl_16)/len(tl_16))
            print("% target > 32: ", sum(tl_32)/len(tl_32))
            print("% target > 64: ", sum(tl_64)/len(tl_64))
            print("*"*100)
            print("*"*100)
            exit(0)

    def __len__(self):
        return len(self.data)
    
    def create_input_instance(self, encoided_ids, max_sequences_length):
        encoided_ids +=[-100 for _ in range(max_sequences_length - len(encoided_ids))]
        encoided_ids = torch.LongTensor(encoided_ids)
        input_mask = (encoided_ids != -100).long()
        encoided_ids.masked_fill_(encoided_ids == -100, self.tokenizer.pad_token_id)
        return {
            'input_ids': encoided_ids,
            'attention_mask' : input_mask,
        }
    
    def get_trie_suggestions(self, example):
        prefix = example[1].strip()
        trie_suggs = list(dict.fromkeys([example[3], example[5], example[7], example[9], example[11], example[13], example[15], example[17]]))   #change 1
        trie_suggs = [ item for item in trie_suggs if item != 'd##']                                          
        if self.args.synth_contx_for_prefix_len == -1:
            synthatic_suggs=[]
        elif self.args.synth_contx_for_prefix_len != -1 and prefix in self.unseen_dict and len(prefix) <= self.args.synth_contx_for_prefix_len:
            synthatic_suggs =  [ item[0] for item in self.unseen_dict[prefix]][:8]   #change 2
        else:
            synthatic_suggs=[]
        
        if len(trie_suggs) == 8:                      ##change 3
           return ','.join(trie_suggs[::-1])
        elif len(trie_suggs) == 0 and len(synthatic_suggs) != 0:
           return ','.join(synthatic_suggs[::-1])
        elif len(trie_suggs) == 0 and len(synthatic_suggs) == 0:
           return '' 
        else:
            all_suggest = trie_suggs + synthatic_suggs
            return ','.join(all_suggest[:8][::-1])     #change 4

    def features_bin(self, lines):
        if not isinstance(lines[0], list):
            examples = [lines]
        else:
            examples = lines
        data_instances = []
        for line in examples:
            #encdoing session and prefix
            if self.enable_trie_context:
                session, label, prefix = line[0], line[2], line[1]
                session = ",".join(session.split('||')[:-1])
                trie_context = self.get_trie_suggestions(line)
                #print(session)
                #print(prefix)
                #print(trie_context)
                #print(label)
                #print("*"*100)
                if len(trie_context) != 0:
                    encoded = self.tokenizer.encode( session + self.tokenizer.eos_token + trie_context + self.tokenizer.eos_token + prefix + self.tokenizer.eos_token, add_special_tokens=False)
                else:
                    encoded = self.tokenizer.encode(session + self.tokenizer.eos_token + prefix + self.tokenizer.eos_token, add_special_tokens=False)
            else: 
                session, label, prefix = line[0], line[2], line[1]
                session = ",".join(session.split('||')[:-1])
                #session = " " 
                encoded = self.tokenizer.encode(session + self.tokenizer.eos_token + prefix + self.tokenizer.eos_token, add_special_tokens=False)
            encoded = encoded[::-1][:self.max_source_length][::-1]
            data_instance = self.create_input_instance(encoded, self.max_source_length)
            
            #encoding label
            encoded_label = self.tokenizer.encode(label + self.tokenizer.eos_token, add_special_tokens=False)
            encoded_label = encoded_label[::-1][:self.max_target_length][::-1]
            data_instance_label = self.create_input_instance(encoded_label, self.max_target_length)
            data_instance['labels'] = data_instance_label['input_ids']
            #data_instance['decoder_attention_mask'] = data_instance_label['attention_mask']
            
            if self.mode == 'test':
                data_instance['prefix'] = prefix.strip()
                data_instance['session'] = line[0]

            if not isinstance(lines[0], list):
                return data_instance
            else:
                data_instances.append(data_instance)
        return data_instances

    def set_epoch(self, epoch):
        self.random_state= np.random.RandomState(epoch)

    def __getitem__(self, indx):
        return self.features_bin(self.data[indx])
