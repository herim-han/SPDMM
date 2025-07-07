from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch.distributed
import argparse
from pathlib import Path
from transformers import BertTokenizer, WordpieceTokenizer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import CSVLogger
import time
import pickle
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import mmap
import joblib
from joblib import Parallel, delayed
from dataset import SMILESDataset_pretrain, collate_fn
from SPMM_debug_models import SPMM
torch.set_float32_matmul_precision('medium')

def main(args, config):
    #CUDA_LAUNCH_BLOCKING=1
    #ngpu=1
    print("Creating dataset")

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False, add_special_tokens=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)
    if (args.pkl is None):
        st = time.time()
        file_names = os.listdir('./Dataset')[:2]
        tmp_dataset = []
        for name in file_names:
            print(f'Start load {name}')
            tmp_st = time.time()
            data = pickle.load(open(f'./Dataset/{name}', 'rb'))
            tmp_et = time.time()
            print(f'End load {name}, {tmp_et-tmp_st:.3f}')
            tmp_dataset.append(data) 
        all_dataset = torch.utils.data.ConcatDataset(tmp_dataset)
        data_loader = DataLoader(all_dataset, batch_size=config['batch_size'], num_workers=8, shuffle=False, pin_memory=True, drop_last=True, collate_fn=collate_fn)#num_workers = IterableDataset
        et = time.time()
        print(f'time for dataloading: {et-st:.3f}, # dataset: {len(all_dataset)}')
    else:#
#        dataset = SMILESDataset_pretrain(args.data_path)
        all_dataset = pickle.load(open(args.pkl, 'rb'))
        data_loader = DataLoader(all_dataset, batch_size=config['batch_size'], num_workers=8, shuffle=False, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    if args.debugging:
        print('Turn on debugging mode')
        model = SPMM(config=config, tokenizer=tokenizer, loader_len=len(all_dataset) // torch.cuda.device_count(), debugging=True)
        et2 = time.time()
    else: 
        model = SPMM(config=config, tokenizer=tokenizer, loader_len=len(all_dataset) // torch.cuda.device_count(), debugging=False)
#    else:
#        from bk_SPMM_models import SPMM
#        from bk_dataset import SMILESDataset_pretrain
#        st = time.time()
#        if (args.pkl is None):
#            dataset = SMILESDataset_pretrain(args.data_path)
#            with open('SMILESDataset.pkl', 'wb') as f:
#                pickle.dump(dataset, f)
#        else:
#            dataset = pickle.load(open(args.pkl, 'rb'))
#        et = time.time()
#        print('time for dataset', et-st)
#        print('#data:', len(dataset), torch.cuda.is_available())
#        print('turn off debugging')
#
#        data_loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=0, shuffle=False, pin_memory=True, drop_last=True)
#        model = SPMM(config=config, tokenizer=tokenizer, loader_len=len(data_loader) // torch.cuda.device_count())

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        _ = model.load_state_dict(checkpoint['state_dict'], strict=False)
#    tb_logger = pl_loggers.TensorBoardLogger(save_dir='lightning_csv_logs', name=f'spmm-bs-{config["batch_size"]}-ddim-{config["embed_dim"]}')
    csv_logger = CSVLogger('lightning_csv_logs', name=f'spmm-bs-{config["batch_size"]}-ddim-{config["embed_dim"]}')
    # training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, 
                                                       filename='checkpoint_{epoch}',
#                                                       every_n_train_steps=10000,
                                                       every_n_train_steps=1,
                                                       )
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=args.gpus, 
                         precision='16-mixed', 
                         max_epochs=config['schedular']['epochs'],
                         callbacks=[checkpoint_callback], 
                         strategy=DDPStrategy(find_unused_parameters=True), 
                         limit_val_batches=0.,
#                         logger=True
                         )
    trainer.logger = csv_logger
    print('start model.fit')
    trainer.fit(model, data_loader, None, ckpt_path=args.checkpoint if args.checkpoint else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    # parser.add_argument('--data_path', default='./data/1_Pretrain/pretrain_20m.txt')
    parser.add_argument('--data_path', default='./data/chemformer_parsed2_shuffle.txt')
    parser.add_argument('--pkl', default=None)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='./Pretrain')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debugging', action='store_true', default=False)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='auto')

    args = parser.parse_args()

    pretrain_config = {
        'property_width': 256,#768
        'embed_dim': 128,
        'batch_size': 24,
        #'batch_size': 96,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'queue_size': 24576, #36864,
        'momentum': 0.995,
        'alpha': 0.4,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
        'schedular': {'sched': 'cosine', 'lr': 5e-5, 'epochs': 30, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 5e-5, 'weight_decay': 0.02},
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, pretrain_config)
