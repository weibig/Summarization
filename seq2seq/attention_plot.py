# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import os
import json
import random
import pickle
import math
from pathlib import Path
from utils import Tokenizer, Embedding
from torch.nn.utils import clip_grad_norm
from dataset import Seq2SeqDataset
from tqdm import tqdm
from tqdm import tnrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from argparse import ArgumentParser
from train import Encoder, Attention, Seq2Seq, Decoder ####


# %%
TEST_DATA_PATH = './seq2seq/valid.pkl'    ###
with open(TEST_DATA_PATH, "rb") as f: 
    test = pickle.load(f)
with open("./seq2seq/embedding.pkl", "rb") as f:   ###
    embedding = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
en_size = len(embedding.vocab)
ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5


attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load('./seq2seq/seq2seq_model.pt'))  ###checkpoint


# %%
SOS_token = 1
EOS_token = 2

def generate_pair(data):
    pairs = []
    input_length = len(data.data)
    for i in range(input_length):
        input = data.__getitem__(i)['text']
        id = data.__getitem__(i)['id']
        pairs.append((input, id))
    return pairs


class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc[idx.item()] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())

def decode(decoder, decoder_hidden, input_len, encoder_outputs, voc, max_length=40):

    decoder_input = torch.LongTensor([SOS_token])
    decoder_input = decoder_input.to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, input_len) #TODO: or (MAX_LEN+1, MAX_LEN+1)
#     print(decoder_hidden.size())
#     all_attention = []
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data # 1 * input_len
        _, topi = decoder_output.topk(3)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc[ni.item()])

        decoder_input = torch.LongTensor([ni])
        decoder_input = decoder_input.to(device)

    return decoded_words, decoder_attentions[:di + 1]
#     return decoded_words, decoder_attentions


def evaluate(encoder, decoder, voc, sentence , max_length=40):
    indexes_batch = [sentence] #[1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    batch_size = input_batch.size(1)
    encoder_outputs, encoder_hidden = encoder(input_batch)

    decoder_hidden = encoder_hidden[:1]
#     decoder_hidden = encoder_hidden.view(encoder.n_layers, batch_size, -1)
    return decode(decoder, decoder_hidden, lengths[0], encoder_outputs, voc)


def evaluateRandomly(encoder, decoder, tokenizer, voc, pairs, reverse, beam_size, n=10):
    for _ in range(n):
        pair = random.choice(pairs)
        print("=============================================================")
        if reverse:
            print('>', " ".join(reversed(pair[0].split())))
        else:
            print('>', tokenizer.decode(pair[0]))
            print('=', tokenizer.decode(pair[1]))
        if beam_size == 1:
            output_words, _ = evaluate(encoder, decoder, voc, pair[0], beam_size)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
        else:
            output_words_list = evaluate(encoder, decoder, voc, pair[0], beam_size)
            for output_words, score in output_words_list:
                output_sentence = ' '.join(output_words)
                print("{:.3f} < {}".format(score, output_sentence))
                
def predict_out(list_dict, file_path):
    with open(file_path , 'w') as outfile:
        for entry in list_dict:
            json.dump(entry, outfile)
            outfile.write('\n')
            
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    plt.rcParams['figure.figsize'] = [20, 10]
    fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    print(input_sentence.split())
    print(output_words.split())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split() +
                       ['</s>'], rotation=90)
    ax.set_yticklabels([''] + output_words.split())

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def predict(encoder, decoder, voc, pairs):
    n = len(pairs)
    out = []
    show_per = 100
    for i in tnrange(n):
        predict = {}
        pair = pairs[i]
        output_words, attention = evaluate(encoder, decoder, voc, pair[0])
        atten = torch.Tensor.cpu(attention.detach()).squeeze(0)
        atten = atten
        output_sentence = ' '.join(output_words)
        if(len(pair[0]) - len(output_sentence.split()) < 10):
            showAttention(tokenizer.decode(pair[0]), output_sentence, atten)
        predict['id'] = pair[1]
        predict['predict'] = output_sentence[:-6]
        out.append(predict)
    return out


# %%
tokenizer = Tokenizer(embedding.vocab, lower=False)
pairs = generate_pair(test)
out = predict(enc, dec, embedding.vocab, pairs)
# predict_out(out, OUT_DATA_PATH)


