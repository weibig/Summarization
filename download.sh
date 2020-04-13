#!/usr/bin/env bash

#pkl
wget https://www.dropbox.com/s/e8pzmgf742m9gof/embedding.pkl?dl=1 -O ./seq_tag/embedding.pkl
wget https://www.dropbox.com/s/hdtdrhgde3g0ryl/tokenizer.pkl?dl=1 -O ./seq_tag/tokenizer.pkl

wget https://www.dropbox.com/s/v3wcwdlfzmtq2m7/embedding.pkl?dl=1 -O ./seq2seq/embedding.pkl
wget https://www.dropbox.com/s/i0chavnwse890h9/tokenizer.pkl?dl=1 -O ./seq2seq/tokenizer.pkl


#ckpt
extractive=https://www.dropbox.com/s/jqicathh8nzpc82/seq_tag.ckpt?dl=1  #checkpoint
abstractive=https://www.dropbox.com/s/1xuzlomxc9fkr29/seq2seq_model_new.pt?dl=1 #checkpoint
attention=https://www.dropbox.com/s/1xuzlomxc9fkr29/seq2seq_model_new.pt?dl=1 #checkpoint

wget "${extractive}" -O ./seq_tag/seq_tag.ckpt #第x個checkpoint
wget "${abstractive}" -O ./seq2seq/seq2seq_model.pt #第x個checkpoint
# wget "${attention}" -O ./seq2seq/seq2seq_model.pt #第x個checkpoint ##Same as abstractive