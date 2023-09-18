# Music Classification Research

This repository serves as a platform for exploring various facets of music classification. It brings together a diverse set of methods and approaches to tackle the problem, providing a rich ground for experimentation and learning.

- Explore supervised Classification using CNN/RNN and others
- Fined-tunning transformers pre trained for other audio tasks such as Wav2Vec2 and Humbert.
- Classic machine learning algoritms for tabular data.

### Resources

- [Great book on music classication](https://music-classification.github.io/tutorial/part3_supervised/tutorial.html)
- [Basic implementations of classsifcation models](https://github.com/minzwon/sota-music-tagging-models/blob/master/training/model.py)
- [Wav2Vec2 transformer for Classification](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [HuBERT transformer](https://huggingface.co/docs/transformers/model_doc/hubert)


### Installation

Using miniconda to manage env

- create envirioment from `envirioment.yml`

        conda env create -f environment.yml

- activate

        conda activate perc-gen

Linux only. Deps will may not install on OSX.

Quick breakdown of this repo. 

### Quick breakdown

- `utils`:  utility scripts 
  - `download_s3`: downloads from Better Problems s3 bucket. You must have aws cli configured to run this.
  - `split_audio`: splits audio files in smaller chunks
- `notebooks`: notebooks exploring the datasets and models
- `models`: model implementations
- `data`: both tabular and audio data folder. this data is not being pushed to this repo.

