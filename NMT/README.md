# Pytorch Attention Neural Machine Translation

Paper implementation of [Neural Machine Translation by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473.pdf).

## Requirements

- Python 3.6
- PyTorch 0.3.0
- torchvision
- numpy
- matplotlib
- floyd-cli


## Installation

```
git clone https://github.com/selenewang/Neural-Machine-Translation.git
cd Neural-Machine-Translation
```

## Usage

```
python trainer.py language_in language_out dataset_path
```

### Example

```
python trainer.py en_3000 ar_3000 data
```
Since the total datasets are extremly large, to test the model, we could run with 3000 pairs of sentences.


## Data


United Nations Parallel Corpus, tokenized by Sentencepiece: https://github.com/google/sentencepiece.git

