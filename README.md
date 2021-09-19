# High-Order Pair-Wise Aspect and Opinion Terms Extraction With Edge-Enhanced Syntactic Graph Convolution
This repository implements the dependency parser described in the paper [High-Order Pair-Wise Aspect and Opinion Terms Extraction With Edge-Enhanced Syntactic Graph Convolution](https://ieeexplore.ieee.org/document/9478183)
## Prerequisite
* [pytorch Library](https://pytorch.org/)
* [transformers](https://huggingface.co/transformers/model_doc/bert.html)
* [corenlp](https://stanfordnlp.github.io/CoreNLP/)

## Usage (by examples)
### Data
Orignal data comes from [TOWE](https://github.com/NJUNLP/TOWE/tree/master/data).


### Preprocessing
We need to obtain the dependency sturcture for each data, and save as json format.
Pay attention to the file path and modify as needed

#### Get Dependency
To parse the dependency structure, we employ the [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) provided by stanfordnlp.
So please download relavant files first.
```
  cd data/datasets/orignal 
  python preprocess.py
```

#### Save
```
  cd data/datasets/towe 
  python preprocess.py
```

We also provide some preprocessed examples. 
If you want to use other datasets to train the model, please refer to the above steps.

### Train
We use embedding bert-cased by [bert-base-cased](https://huggingface.co/bert-base-cased) (768d)

```
  python aop.py train --config configs/16res_train.conf
```
### Test
```
  python aop.py test --config configs/16res_test.conf
```

## Notes
This code refers to [SpERT](https://github.com/lavis-nlp/spert), Thanks a lot.
