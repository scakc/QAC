import os
import torch
from torch import nn

from transformers import (
    T5Config, 
    T5ForConditionalGeneration, 
    T5Tokenizer,
    BartConfig,
    BartForConditionalGeneration, 
    BartTokenizer,
)

def load_model_tokenizer(args):
    config_kwargs = {}
    if args.max_generated_seq_len:
        config_kwargs.update({'max_length': args.max_generated_seq_len})
    if args.beam_size:
        config_kwargs.update({'num_beams': args.beam_size})
    if args.length_penalty:
        config_kwargs.update({'length_penalty': args.length_penalty})
    if args.no_repeat_ngram_size:
        config_kwargs.update({'no_repeat_ngram_size': args.no_repeat_ngram_size})
    
    if args.model_type == "t5":
        def copy_layers(src_layers, dest_layers, layers_to_copy):
            layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
            assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
            dest_layers.load_state_dict(layers_to_copy.state_dict()) 

        config = T5Config.from_pretrained(
            args.model_chkpt,
            cache_dir=args.cache_dir, **config_kwargs,
        )

        tokenizer = T5Tokenizer.from_pretrained(
            args.model_chkpt,
            use_fast=False, cache_dir=args.cache_dir,
        )
        if args.without_pre_train_weights:
            model = T5ForConditionalGeneration(config=config)
        else:
            old_model = T5ForConditionalGeneration.from_pretrained(
                args.model_chkpt,
                from_tf=".ckpt" in args.model_chkpt,
                config=config,
                cache_dir=args.cache_dir,
            ) 
            new_config = T5Config.from_pretrained(
                args.model_chkpt,
                num_layers=6,
                num_decoder_layers=6,
                cache_dir=args.cache_dir, **config_kwargs,
            )

            model = T5ForConditionalGeneration(new_config)
            layers_to_copy = [0,1,2,3,4,5]

            copy_layers(old_model.encoder.block, model.encoder.block, layers_to_copy)
            copy_layers(old_model.decoder.block, model.decoder.block, layers_to_copy)

            #model.save_pretrained("small_model")
            #model = T5ForConditionalGeneration.from_pretrained("small_model")



    elif args.model_type == "bart":
        config = BartConfig.from_pretrained(
            args.model_chkpt,
            cache_dir=args.cache_dir, **config_kwargs,
        )
        tokenizer = BartTokenizer.from_pretrained(
            args.model_chkpt,
            use_fast=False, cache_dir=args.cache_dir,
        )
        if args.without_pre_train_weights:
            model = BartForConditionalGeneration(config=config)
        else:
            model = BartForConditionalGeneration.from_pretrained(
                args.model_chkpt,
                from_tf=".ckpt" in args.model_chkpt,
                config=config,
                cache_dir=args.cache_dir,
            )
    return model, tokenizer