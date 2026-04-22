# NLP Experiments

This folder contains Akkadian -> English translation experiments built on the same core MT dataset, plus one ongoing MLM-pretraining-to-MT notebook that also uses a larger Akkadian-only pretraining corpus.

The goal of this README is to capture, in one place:

- which datasets each experiment used
- what the dataset columns looked like
- the train / validation / test split used
- preprocessing choices
- model and optimisation hyperparameters
- any recorded test results
- the **geometric mean of BLEU and chrF++** on test data, which is the evaluation metric we care about operationally

## Evaluation Metric

For completed MT experiments, we report:

- BLEU
- chrF++
- geometric mean of BLEU and chrF++

Formula:

```text
GM(BLEU, chrF++) = sqrt(BLEU * chrF++)
```

This README reports that value on **test data** where test metrics are available in the current artifacts.

## Datasets

### 1. `data.csv` — machine translation dataset

Raw file:

- path: `data.csv`
- raw line count: `62,207` including header

Raw columns:

- `translation`
- `transliteration`
- `source`
- `language`

Example row shape:

- `translation`: English translation text
- `transliteration`: Akkadian transliteration text
- `source`: split/source label in the CSV itself, often values like `train`
- `language`: language code, e.g. `en`

How experiments use it:

- `bilstm_fixed.ipynb` and `bilstm_attention.ipynb` read `['source', 'translation']`
  - note: this means those early notebooks are using the CSV column literally named `source`, not `transliteration`
- `bilstm_attention_bpe.ipynb`, `transformer_akkadian_english.ipynb`, and `transformer_akkadian_mlm_to_mt.ipynb` read `['transliteration', 'translation']` and rename `transliteration -> source`

Cleaned MT size used by the later experiments:

- after dropping null `transliteration/translation` pairs: `62,196`

Common MT preprocessing:

- lowercase text
- remove double quotes `"`
- replace `<gap>` with ` <sep> `

### 2. `pretrain_large.csv` — Akkadian-only pretraining corpus

Raw file:

- path: `pretrain_large.csv`
- raw line count: `349,491` including header

Raw columns:

- `transliteration`
- `clean_transliteration`
- `translation`
- `clean_translation`

How the MLM notebook uses it:

- only the `transliteration` column is used for pretraining
- text is lowercased
- double quotes are removed
- `<gap>` is replaced with ` <sep> `
- strings shorter than 4 characters are filtered out

Cleaned pretraining size used by the MLM notebook:

- `325,818` Akkadian strings

## Shared Split Conventions

### MT split used by the later serious runs

For:

- `bilstm_attention_bpe.ipynb`
- `transformer_akkadian_english.ipynb`
- `transformer_akkadian_mlm_to_mt.ipynb` stage 2

Split:

- train: `49,756`
- val: `6,220`
- test: `6,220`
- strategy: `80 / 10 / 10`, with split done before tokeniser training to avoid leakage

### MLM pretraining split

For:

- `transformer_akkadian_mlm_to_mt.ipynb` stage 1

Split:

- train: `309,527`
- val: `16,291`
- strategy: `95 / 5`

## Experiment Summary

### 1. `bilstm_fixed.ipynb` — BiLSTM seq2seq baseline

Purpose:

- first cleaner seq2seq baseline
- encoder-decoder setup without attention

Dataset used:

- `data.csv`
- columns used: `['source', 'translation']`

Important caveat:

- this notebook uses the CSV column literally named `source`, not `transliteration`
- based on the current CSV, that `source` field looks like a split/source tag such as `train`
- the resulting printed source vocabulary size in the notebook is only `9`
- so this notebook should be treated as an early baseline attempt with a likely dataset-column mismatch

Preprocessing:

- lowercase
- remove `"`
- replace `<gap>` -> ` <sep> `
- wrap targets with `<sos>` and `<eos>`
- whitespace tokenisation
- vocabulary built from training split only
- vocabulary minimum frequency: `2`

Split:

- documented as `80 / 10 / 10`
- later-cell DataLoaders use train / val / test DataFrames produced from that split

Data loading:

- batch size: `32`
- train loader shuffle: `True`
- val/test shuffle: `False`

Model:

- encoder: bidirectional LSTM
- decoder: unidirectional LSTM
- embeddings: `256`
- hidden size: `512`
- dropout: `0.5`
- trainable params: `32,415,561`

Optimisation:

- optimiser: `Adam` with default learning rate
- loss: `CrossEntropyLoss(ignore_index=0)`
- teacher forcing ratio during training: `0.5`
- teacher forcing ratio during validation: `0.0`
- gradient clipping: `1.0`
- planned epochs: `30`
- checkpoint: `bilstm_best.pt`

Observed status:

- notebook output shows it was interrupted during epoch 1
- no recorded final test metrics are available in the checked-in notebook/artifacts

Test metrics:

- unavailable from current artifacts

Geometric mean on test:

- unavailable

### 2. `bilstm_attention.ipynb` — early attention prototype

Purpose:

- quick prototype of BiLSTM + attention
- appears exploratory rather than a fully tracked experiment

Dataset used:

- `data.csv`
- columns used: `['source', 'translation']`

Important caveat:

- same as the fixed BiLSTM notebook, this reads the CSV column named `source`
- no explicit train/val/test split is created in the notebook

Preprocessing:

- lowercase
- remove `"`
- replace `<gap>` -> ` <sep> `
- wrap targets with `<sos>` and `<eos>`
- whitespace tokenisation
- vocabulary built across the full loaded dataframe
- minimum token frequency effectively `2`

Split:

- none recorded
- training is done on the full dataframe
- evaluation is run on `df.sample(100)`, not on a held-out test set

Data loading:

- batch size: `32`
- shuffle: `True`

Model:

- attention hidden sizes: `256`
- encoder embedding dim: `128`
- encoder hidden dim: `256`
- decoder embedding dim: `128`
- decoder hidden dim: `256`

Optimisation:

- optimiser: `Adam` default settings
- loss: `CrossEntropyLoss(ignore_index=0)`
- epochs: `10`
- gradient clipping: `1.0`

Observed status:

- this notebook is best viewed as an exploratory prototype
- it does not define a proper held-out test evaluation pipeline

Test metrics:

- not applicable / not recorded as a real test-set result

Geometric mean on test:

- unavailable

### 3. `bilstm_attention_bpe.ipynb` — BiLSTM + Bahdanau attention + BPE

Purpose:

- stronger recurrent baseline with attention, subword tokenisation, label smoothing, LR scheduling, and beam search

Dataset used:

- `data.csv`
- columns used: `['transliteration', 'translation']`
- `transliteration` renamed to `source`

Preprocessing:

- lowercase
- remove `"`
- replace `<gap>` -> ` <sep> `
- split before tokeniser training
- source BPE vocab size: `4000`
- target BPE vocab size: `6000`
- special tokens: `<pad>`, `<unk>`, `<sos>`, `<eos>`, `<sep>`

Split:

- `80 / 10 / 10`
- train: `49,756`
- val: `6,220`
- test: `6,220`

Data loading:

- batch size: `32`
- train shuffle: `False`
- val/test shuffle: `False`
- recorded batches:
  - train: `1167`
  - val: `147`
  - test: `146`

Model:

- encoder: stacked BiLSTM with Bahdanau attention
- embedding dim: `128`
- hidden dim: `256`
- encoder layers: `2`
- dropout: `0.5`
- label smoothing: `0.1`
- trainable params: `9,640,560`

Optimisation:

- optimiser: `Adam(lr=1e-3)`
- scheduler: `ReduceLROnPlateau(mode='min', factor=0.5, patience=3)`
- loss: `CrossEntropyLoss(ignore_index=TGT_PAD_IDX, label_smoothing=0.1)`
- epochs planned: `100`
- early stopping patience: `10`
- gradient clipping: `1.0`
- beam size at inference: `4`
- beam length penalty: `0.7`
- checkpoint: `bilstm_attention_bpe_best.pt`

Observed status:

- notebook contains early training output
- no committed best checkpoint or final test output was found in this folder
- final test metrics are not recoverable from current checked-in artifacts

Test metrics:

- unavailable from current artifacts

Geometric mean on test:

- unavailable

### 4. `transformer_akkadian_english.ipynb` + `transformer_akkadian_english_runner.py` — vanilla Transformer + BPE

Purpose:

- full Transformer encoder-decoder baseline trained directly on MT data

Dataset used:

- `data.csv`
- columns used: `['transliteration', 'translation']`
- `transliteration` renamed to `source`

Preprocessing:

- lowercase
- remove `"`
- replace `<gap>` -> ` <sep> `
- split before tokeniser training
- source BPE vocab size: `4000`
- target BPE vocab size: `6000`
- target sequences wrapped with `<sos>` and `<eos>`
- later standalone runner also filters sequences to max length `80`

Split:

- `80 / 10 / 10`
- train: `49,756`
- val: `6,220`
- test: `6,220`
- after length filtering in the runner:
  - train: `37,325`
  - val: `4,687`
  - test: `4,653`

Data loading:

- batch size: `32`
- train shuffle: `False`
- val/test shuffle: `False`
- batches after filtering:
  - train: `1167`
  - val: `147`
  - test: `146`

Model:

- model dim: `256`
- attention heads: `8`
- encoder layers: `3`
- decoder layers: `3`
- feedforward dim: `512`
- dropout: `0.1`
- label smoothing: `0.1`
- warmup steps: `4000`
- trainable params: `8,056,688`

Optimisation:

- optimiser: `Adam(lr=1.0, betas=(0.9, 0.98), eps=1e-9)`
- scheduler: Noam warmup + inverse square root decay
- loss: `CrossEntropyLoss(ignore_index=TGT_PAD_IDX, label_smoothing=0.1)`
- planned epochs: `300`
- early stopping patience: `10`
- gradient clipping: `1.0`
- decoding:
  - greedy
  - beam search with beam size `4`
  - length penalty `0.7`

Final recorded run:

- run completed through the standalone script `transformer_akkadian_english_runner.py`
- early stopped after epoch `160`
- best validation loss epoch: `150`
- best validation BLEU epoch: `159`
- best validation chrF++ epoch: `160`

Final test metrics:

- greedy:
  - BLEU: `20.60`
  - chrF++: `47.80`
  - geometric mean: `31.38`
- beam-4:
  - BLEU: `25.95`
  - chrF++: `49.93`
  - geometric mean: `36.00`

Artifacts:

- `transformer_akkadian_best.pt`
- `transformer_history.csv`
- `transformer_runner_nohup.log`
- `transformer_akkadian_english_runner.py`

### 5. `transformer_akkadian_mlm_to_mt.ipynb` — MLM pretraining + MT fine-tuning

Purpose:

- two-stage Transformer setup
- stage 1: masked language modelling on a larger Akkadian-only corpus
- stage 2: initialise the MT encoder from the pretrained MLM encoder and fine-tune for translation

Status:

- currently ongoing / not yet finished
- no final test metrics are available yet

#### Stage 1: MLM pretraining

Dataset used:

- `pretrain_large.csv`
- column used: `transliteration`

Preprocessing:

- lowercase
- remove `"`
- replace `<gap>` -> ` <sep> `
- drop missing `transliteration`
- keep only strings with length >= `4`
- source BPE vocab size: `4000`
- special tokens:
  - `<pad>`
  - `<unk>`
  - `<sos>`
  - `<eos>`
  - `<sep>`
  - `<mask>`

Split:

- `95 / 5`
- train: `309,527`
- val: `16,291`

Masking / sequence setup:

- MLM max sequence length: `80`
- mask ratio: `0.15`
- of masked positions:
  - `80%` replaced with `<mask>`
  - `10%` replaced with a random token
  - `10%` kept unchanged
- ignore index for unmasked positions: `-100`

Data loading:

- batch size: `64`
- train shuffle: `True`
- val shuffle: `False`
- `num_workers=2`
- `pin_memory=True`

Model:

- Transformer encoder only
- model dim: `256`
- attention heads: `8`
- encoder layers: `3`
- feedforward dim: `512`
- dropout: `0.1`
- label smoothing: `0.1`

Optimisation:

- optimiser: `Adam(lr=1.0, betas=(0.9, 0.98), eps=1e-9)`
- scheduler: Noam warmup + inverse square root decay
- planned epochs: `100`
- early stopping patience: `5`
- gradient clipping: `1.0`
- checkpoint: `transformer_akkadian_mlm_best.pt`

#### Stage 2: MT fine-tuning

Dataset used:

- `data.csv`
- columns used: `['transliteration', 'translation']`

Preprocessing:

- same MT cleaning as the vanilla Transformer
- source tokenizer reused from MLM pretraining
- English target tokenizer trained only on MT training split
- target sequences wrapped with `<sos>` and `<eos>`

Split:

- `80 / 10 / 10`
- train: `49,756`
- val: `6,220`
- test: `6,220`

Data loading:

- MT batch size: `32`
- train shuffle: `False`
- val/test shuffle: `False`

Model:

- Transformer encoder-decoder MT model
- encoder weights loaded from pretrained MLM encoder
- decoder randomly initialised
- model dim: `256`
- attention heads: `8`
- encoder layers: `3`
- decoder layers: `3`
- feedforward dim: `512`
- dropout: `0.1`
- label smoothing: `0.1`

Fine-tuning strategy:

- encoder frozen for first `3` epochs
- then fully unfrozen

Optimisation:

- optimiser: `Adam(lr=1.0, betas=(0.9, 0.98), eps=1e-9)`
- scheduler: Noam warmup + inverse square root decay
- planned epochs: `300`
- early stopping patience: `20`
- gradient clipping: `1.0`
- checkpoint: `transformer_akkadian_mt_best.pt`
- decoding:
  - greedy
  - beam search with beam size `4`
  - length penalty `0.7`

Test metrics:

- not available yet because the run is still in progress / incomplete

Geometric mean on test:

- unavailable until final test BLEU and chrF++ are produced

## Current Artifact Inventory

Important saved files currently present in this folder:

- `transformer_akkadian_best.pt`
- `transformer_akkadian_mlm_best.pt`
- `transformer_history.csv`
- `transformer_runner_nohup.log`
- `transformer_akkadian_mlm_to_mt_nohup.log`
- `transformer_akkadian_english_runner.py`
- notebooks for all experiment variants

## Notes

- The strongest completed and fully logged result currently available in this folder is the **vanilla Transformer + BPE** run from `transformer_akkadian_english_runner.py`.
- Some earlier notebooks are exploratory and do not have fully recoverable final test metrics in the current committed artifacts.
- The ongoing **Transformer + MLM pretraining** notebook should be updated here once it finishes, especially with:
  - best validation epoch
  - test BLEU
  - test chrF++
  - geometric mean on test
