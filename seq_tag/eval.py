import os
import pickle
import logging
from argparse import Namespace,ArgumentParser
from typing import Tuple, Dict
from pathlib import Path
import json
from typing import Iterable
from tqdm import tqdm
from utils import Tokenizer, Embedding

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.autograd import Variable
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import SeqTaggingDataset
from train import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()


##stored test.jsonl to test.pkl
def main(path):
    with open(path) as f:
        test = [json.loads(line) for line in f]


    with open("./seq_tag/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    logging.info('Creating test dataset...')
    create_seq_tag_dataset(
        process_seq_tag_samples(tokenizer, test),
        './seq_tag/test.pkl'
    )

def process_seq_tag_samples(tokenizer, samples):
    processeds = []
    for sample in tqdm(samples):
        if not sample['sent_bounds']:
            continue
        processed = {
            'id': sample['id'],
            'text': tokenizer.encode(sample['text']),
            'sent_range': get_tokens_range(tokenizer, sample)
        }

        if 'extractive_summary' in sample:
            label_start, label_end = processed['sent_range'][sample['extractive_summary']]
            processed['label'] = [
                1 if label_start <= i < label_end else 0
                for i in range(len(processed['text']))
            ]
            assert len(processed['label']) == len(processed['text'])
        processeds.append(processed)
    return processeds


def get_tokens_range(tokenizer,
                     sample) -> Iterable:
    ranges = []
    token_start = 0
    for char_start, char_end in sample['sent_bounds']:
        sent = sample['text'][char_start:char_end]
        tokens_in_sent = tokenizer.tokenize(sent)
        token_end = token_start + len(tokens_in_sent)
        ranges.append((token_start, token_end))
        token_start = token_end
    return ranges


def create_seq_tag_dataset(samples, save_path, padding=0):
    dataset = SeqTaggingDataset(
        samples, padding=padding,
        max_text_len=300
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

main(args.test_data_path)


hparams = Namespace(**{
        'embedding_path': "./seq_tag/embedding.pkl",
        'embed_size': 300,

        'train_dataset_path': "./seq_tag/train.pkl",
        'valid_dataset_path': "./seq_tag/valid.pkl",

        'test_dataset_path': args.test_data_path,####
        'ignore_idx': -100,

        'batch_size': 16,
        'pos_weight': 5, # TODO
    #     When you increase the pos_weight, 
    #     the number of false negatives will artificially increase.

        'rnn_hidden_size': 300,
    })

seq_tagger = SeqTagger(hparams)
seq_tagger = seq_tagger.load_from_checkpoint("./seq_tag/seq_tag.ckpt")
trainer = pl.Trainer()
trainer.test(seq_tagger)


with open("./seq_tag/test.pkl", "rb") as f:
    test = pickle.load(f)


test_dataset = SeqTaggingDataset(
        test, padding=0,
        max_text_len=300,)


sent_range = [d['sent_range'] for d in test_dataset.data]
final_ans = []

for p, i, r in zip(predict, id, sent_range):
    sent_probs = np.array([np.array(p[start:end]).mean() for start, end in r])
    final_ans.append({'id': i, 'predict_sentence_index': [int(x) for x in sent_probs.argsort()[::-1][0:2]]})

Path(args.output_path).write_text('\n'.join([json.dumps(ans) for ans in final_ans])+'\n')
