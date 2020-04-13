import json
import pickle
from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.nn as nn
from utils import Embedding,Tokenizer
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm

##########################################
from train import (Seq2Seq, Attention, Encoder, Decoder)
##########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()

ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5


def pad_to_len(seqs, to_len, padding=0):
    paddeds = []
    for seq in seqs:
        paddeds.append(
            seq[:to_len] + [padding] * max(0, to_len - len(seq))
        )

    return paddeds

class Seq2SeqDataset(Dataset):
    def __init__(self, data, padding=0,
                 max_text_len=300, max_summary_len=80):
        self.data = data
        self.padding = padding
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            'id': sample['id'],
            'text': sample['text'][:self.max_text_len],
            'len_text': len(sample['text']),
            'attention_mask': [True] * min(len(sample['text']),
                                           self.max_text_len)
        }
        if 'summary' in sample:
            sample['summary'] = sample['summary'][:self.max_summary_len]
            sample['len_summary'] = len(sample['summary'])
        return instance

    def collate_fn(self, samples):
        batch = {}
        for key in ['id', 'len_text', 'len_summary']:
            if any(key not in sample for sample in samples):
                continue
            batch[key] = [sample[key] for sample in samples]

        for key in ['text', 'summary', 'attention_mask']:
            if any(key not in sample for sample in samples):
                continue
            to_len = max([len(sample[key]) for sample in samples])
            padded = pad_to_len(
                [sample[key] for sample in samples], to_len, self.padding
            )
            batch[key] = torch.tensor(padded)

        return batch


class SeqTaggingDataset(Seq2SeqDataset):
    ignore_idx = -100

    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            'id': sample['id'],
            'text': sample['text'][:self.max_text_len],
            'sent_range': sample['sent_range']
        }
        if 'label' in sample:
            instance['label'] = sample['label'][:self.max_text_len]
        return instance

    def collate_fn(self, samples):
        batch = {}
        for key in ['id', 'sent_range']:
            batch[key] = [sample[key] for sample in samples]

        for key in ['text', 'label']:
            if any(key not in sample for sample in samples):
                continue
            to_len = max([len(sample[key]) for sample in samples])
            padded = pad_to_len(
                [sample[key] for sample in samples], to_len,
                self.padding if key != 'label' else SeqTaggingDataset.ignore_idx
            )
            batch[key] = torch.tensor(padded)

        return batch

        
def process_samples(tokenizer, samples):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    processeds = []
    for sample in tqdm(samples):
        processed = {
            'id': sample['id'],
            'text': tokenizer.encode(sample['text']) + [eos_id],
        }
        if 'summary' in sample:
            processed['summary'] = (
                [bos_id]
                + tokenizer.encode(sample['summary'])
                + [eos_id]
            )
        processeds.append(processed)

    return processeds


def testing(model, iterator):
    
    model.eval()
    
    ans = []
    ids = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            src = batch['text'].permute(1,0)
            output = model(src, None, 0).to(device) #turn off teacher forcing
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output = output[1:].argmax(-1) # 300 16 90000
            ans += [tokenizer.decode(line) for line in output]
            ids += batch['id']
    return ans, ids

with open(args.test_data_path) as f:
    test = [json.loads(line) for line in f]
    
with open('./seq2seq/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

test_dataset = Seq2SeqDataset(process_samples(tokenizer, test), padding=0)
testloader = DataLoader(test_dataset, batch_size=8, collate_fn=test_dataset.collate_fn)


attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load('./seq2seq/seq2seq_model.pt'))

model.eval()

predictions, ids = testing(model, testloader)


output_file=Path(args.output_path)

#import ipdb
#ipdb.set_trace()
final_ans = []
for i, p in zip(ids, predictions):
    #sent_probs = np.array([np.array(p[start:end]).mean() for start, end in r])
    #print(sent_probs)
    final_ans.append({'id': i, 'predict': p})

Path('predict.jsonl').write_text('\n'.join([json.dumps(ans) for ans in final_ans])+'\n')

