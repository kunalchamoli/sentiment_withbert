# Simple Implementation Of Sentiment Classification Using Bert and Pytorch-Lightning

The goal of this repo is to make a simple sentiment classifier using [BERT](https://arxiv.org/pdf/1810.04805.pdf)

The Libraries Mainly Used are:
- [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [Transformers](https://huggingface.co/transformers/index.html)
- [PyTorch-NLP](https://pytorchnlp.readthedocs.io/en/latest/index.html)

## Requirements
This Project uses Python 3.7.7 installed using miniconda

Create a virtual environment using conda 
``` bash
conda create -n yourvirtualenv python=3.7.7
```
Install the requirements (inside the project folder):
```bash
pip install -r requirements.txt
```

## Getting Started:

### Train:
``` bash 
python f.py
```
Available commands:
```bash
    --debug         Work only with a Single Batch, will help in faster debugging of code
```