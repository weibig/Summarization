import os
import pickle
from argparse import Namespace
from typing import Tuple, Dict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl
from torch.autograd import Variable
# import tensorflow as tf
from torch import optim
import torch.nn.functional as F
import math
import time
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 8

SOS_token = 0
EOS_token = 1

ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

EMBEDDING_PATH="./seq2seq/embedding.pkl"


class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        with open(EMBEDDING_PATH, 'rb') as f:
            embedding = pickle.load(f)
            self.input_dim=len(embedding.vocab)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [src len, batch size]

        embedded = self.dropout(self.embedding(src)).to(device)


        #embedded = [src len, batch size, emb dim] torch.Size([300, 16, 300])
        ##########

        outputs, hidden = self.rnn(embedded)


        #outputs = [src len, batch size, hid dim * num directions] torch.Size([300, 16, 512])


        #hidden = [n layers * num directions, batch size, hid dim] torch.Size([2, 300, 256])

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer

        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN

        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):

        #hidden = [batch size, dec hid dim] torch.Size([300, 256])
        #encoder_outputs = [src len, batch size, enc hid dim * 2] torch.Size([300, 16, 512])

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        #print(hidden.size(),encoder_outputs.size())

        encoder_outputs = encoder_outputs.permute(1, 0, 2)


        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))

        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        #attention= [batch size, src len]

        return F.softmax(attention, dim=1)



class Decoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()


        self.attention = attention

        with open(EMBEDDING_PATH, 'rb') as f:
            embedding = pickle.load(f)
            self.output_dim = len(embedding.vocab)
        ####
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        ####

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, self.output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):

        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)

        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input))


        #embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)

        #a = [batch size, src len]

        a = a.unsqueeze(1)

        #a = [batch size, 1, src len]

        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs) # batch matrix multiply

        #weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        #weighted = [1, batch size, enc hid dim * 2]
        #print(weighted.size(),embedded.size())



        rnn_input = torch.cat((embedded, weighted), dim = 2) ###error


        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]

        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        #prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0),a.squeeze(1) ########## revise for plotting


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, teacher_forcing_ratio = 0.5):

        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0] if trg!=None else 80

        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        attention_weight=torch.zeros(trg_len, batch_size) ########## revise for plotting

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer

        encoder_outputs, hidden = self.encoder(src.to(self.device))

        #first input to the decoder is the <sos> tokens

        
        input = trg[0,:] if trg!=None else torch.ones(batch_size).to(self.device).long()
#         print("content",input,"size",input.size())

        for t in range(1,trg_len):

            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            
            #attention = [batch size, src len]
            output, hidden , attention= self.decoder(input, hidden, encoder_outputs)
            
            #attention_weight[t]=attention ########## revise for plotting

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output # output= [batch size, output dim]

            #get the highest predicted token from our predictions
            top1 = output.argmax(1).long()

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if trg!=None else top1

        return outputs#,attention_weight ########## revise for plotting


def _load_dataset(dataset_path: str):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def train_dataloader():
    dataset = _load_dataset(TRAIN_DATA_PATH)
    return DataLoader(dataset, 
                      BATCH_SIZE, 
                      shuffle=True,
                      collate_fn=dataset.collate_fn,pin_memory=True)

def val_dataloader():
    dataset = _load_dataset(VALID_DATA_PATH)
    return DataLoader(dataset, 
                      BATCH_SIZE, 
                      collate_fn=dataset.collate_fn,pin_memory=True)        

def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(tqdm(iterator)):

        # print(batch)
        src = batch['text'].permute(1,0).to(device)
        trg = batch['summary'].permute(1,0).to(device)

        #print(src.device,trg.device)

        optimizer.zero_grad()
        output = model(src, trg)

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        #print(output)
        #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]

        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(tqdm(iterator)):

            
            src = batch['text'].permute(1,0).to(device)
            trg = batch['summary'].permute(1,0).to(device)


            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
        
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    TRAIN_DATA_PATH="./seq2seq/train.pkl"
    VALID_DATA_PATH="./seq2seq/valid.pkl"
    TOKENIZER_PATH='./seq2seq/tokenizer.pkl'

    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)


# train model here

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM).to(device)
    enc = Encoder(ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT).to(device)
    dec = Decoder(DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn).to(device)

    model = Seq2Seq(enc, dec, device)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index = 0)


    N_EPOCHS = 3

    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_dataloader(), optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_dataloader(), criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './seq2seq/seq2seq_model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')













