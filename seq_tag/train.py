import os
import logging

import pickle
from argparse import Namespace
from typing import Tuple, Dict
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.autograd import Variable
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import SeqTaggingDataset
from utils import Tokenizer, Embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predict = []
id = []



class Encoder(nn.Module):
    def __init__(self,
                 embedding_path,
                 embed_size,
                 rnn_hidden_size) -> None:
        super(Encoder, self).__init__()
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.rnn = nn.LSTM(embed_size, rnn_hidden_size, num_layers = 2, batch_first=True) # TODO
        # init a LSTM/RNN
        self.embed_size = embed_size
        self.hidden_size = rnn_hidden_size
        

    def forward(self, idxs) -> Tuple[torch.tensor, torch.tensor]:
        embed = self.embedding(idxs) #idx is pretrained weight
#         print(embed.size())
        output, state = self.rnn(embed)################
        return output, state
    
class SeqTagger(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super(SeqTagger, self).__init__()
        self.hparams = hparams
        self.criterion = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=torch.tensor(hparams.pos_weight))
        self.encoder = Encoder(hparams.embedding_path, hparams.embed_size,
                               hparams.rnn_hidden_size)
        self.proj = nn.Linear(hparams.rnn_hidden_size, 1)
        
        self.threshold = 0.5

    def forward(self, idxs) -> torch.tensor:
        # TODO
        # take the output of encoder
        # project it to 1 dimensional tensor
        output, state = self.encoder(idxs)
#         print("idx",idxs.size())
#         print("from",output.size())
        logit = self.proj(output).squeeze()
#         print("to",logit.size())
        return logit #model to fit

    
    

    def _unpack_batch(self, batch) -> Tuple[torch.tensor, torch.tensor]:
        return batch['text'], batch['label'].float()

    def _calculate_loss(self, y_hat, y) -> torch.tensor:
        # TODO
        # calculate the logits
        # plz use BCEWithLogit
        # adjust pos_weight!
        # MASK OUT PADDINGS' LOSSES!
        
#         print("from",y_hat.size())

#         print("to",y_hat.size())
        loss = self.criterion(y_hat,y)
        
        mask = y.ne(-100)
        loss = torch.masked_select(loss,mask)
        
        return loss.mean()

    def training_step(self, batch, batch_nb) -> Dict:
        x, y = self._unpack_batch(batch)
        logit = self.forward(x)
        
#         print("yo", logit.size(),y.size())
        loss = self._calculate_loss(logit, y)
        tensorboard_logs = {'train_loss': loss}
        
#         print(loss,"!!!")
        
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb) -> Dict:
        x, y = self._unpack_batch(batch)
        logit = self.forward(x)
        loss = self._calculate_loss(logit, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs) -> Dict:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())

    def _load_dataset(self, dataset_path: str) -> SeqTaggingDataset:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    def train_dataloader(self):
        dataset = self._load_dataset(self.hparams.train_dataset_path)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          shuffle=True,
                          collate_fn=dataset.collate_fn)

    def val_dataloader(self):
        dataset = self._load_dataset(self.hparams.valid_dataset_path)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          collate_fn=dataset.collate_fn)
    ####
    def test_step(self, batch, batch_idx):
        x = batch['text']##error
        pred = self.forward(x).sigmoid().cpu().numpy()
        
        return {'predict': pred, 'id': batch['id']}

    def test_epoch_end(self, outputs):
        global predict, id
        
        for x in outputs:
            predict += x['predict'].tolist()
            id += x['id']
        return {'predict': predict, 'id': id}

    def test_dataloader(self):
        dataset = self._load_dataset(self.hparams.test_dataset_path)
        return DataLoader(dataset, 
                          self.hparams.batch_size, 
                          collate_fn=dataset.collate_fn)

hparams = Namespace(**{
    'embedding_path': "./seq_tag/embedding.pkl",
    'embed_size': 300,

    'train_dataset_path': "./seq_tag/train.pkl",
    'valid_dataset_path': "./seq_tag/valid.pkl",
    'test_dataset_path': "./seq_tag/valid.pkl",####
    'ignore_idx': -100,

    'batch_size': 16,
    'pos_weight': 5, # TODO
#     When you increase the pos_weight, 
#     the number of false negatives will artificially increase.

    'rnn_hidden_size': 300,
})
if __name__ == "__main__":

    seq_tagger = SeqTagger(hparams)
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(seq_tagger)
    trainer.save_checkpoint("./seq_tag/checkpt_test.ckpt")         #save ckpt
    trainer.test()



    test_dataset_path = './seq_tag/valid.pkl'
    with open(test_dataset_path, 'rb') as f:
        test_dataset = pickle.load(f)
    sent_range = [d['sent_range'] for d in test_dataset.data]

    final_ans = []
    for p, i, r in zip(predict, id, sent_range):
        sent_probs = np.array([np.array(p[start:end]).mean() for start, end in r])

        final_ans.append({'id': i, 'predict_sentence_index': [int(x) for x in sent_probs.argsort()[::-1][0:2]]})

    Path('predict.jsonl').write_text('\n'.join([json.dumps(ans) for ans in final_ans])+'\n')
