# PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition

This repository is an official implementation based on paper [PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition](https://arxiv.org/abs/2402.04838) 

![overview](./padellm.png)

# Datasets
We provide processed datasets used in our paper at the **ner_data** directory, except ACE05 and Ontonotes 4 for copyright reasons.

# Training code and model weights

Since it is easy to re-implement the training procedure based on the paper, we will not provide the training code, instead, we provide the **fine-tuned models** online for replicating purpose.

All model weights are available at https://huggingface.co/JinghuiLuAstronaut

with names **PaDeLLM\_{model}\_{model_size}\_{dataset}**

# Inference code

You can run inference based on 
