# ADL HW1

## Implement Prediction
### Extractive Summary
```bash
bash download.sh
bash extractive.sh "${TEST_FILE}" "${EXT_FILE}"
```


### Abstractive Summary
```bash
bash download.sh
bash seq2seq.sh "${TEST_FILE}" "${S2S_FILE}"
```


### Abstractive (with attention) Summary
```bash
bash download.sh
bash attention.sh "${TEST_FILE}" "${ATT_FILE}"
```


## Training Model
### Extractive Summary
```bash
python3.6 ./seq_tag/train.py
```


### Abstractive Summary
```bash
python3.6 ./seq2seq/train.py
```

### Abstractive (with attention) Summary
```bash
python3.6 ./seq2seq/train.py
```


## Plotting Figures
### <HW IV.> Plot the distribution of relative location

```bash
pip3 install matplotlib.pyplot
pip3 install math

python3.6 ./seq_tag/distribution_plot.py
```


### <HW V.> Visualize the attention weights

```bash
pip3 install matplotlib.pyplot
pip3 install matplotlib.ticker

python3.6 ./seq2seq/attention_plot.py
```

