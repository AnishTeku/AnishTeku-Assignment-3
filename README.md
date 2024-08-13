# LLM-MODEL-ASSIGNMENT
# HUMAN EMOTION DETECTION USING MACHINE LEARNING

---

This repository contains a project focused on building and fine-tuning a BERT-based model to classify emotions in text data. The project uses the dair-ai/emotion dataset and leverages the transformers library from Hugging Face along with TensorFlow for model training.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Architecture](#model-architecture)
- [Training and Fine-Tuning](#training-and-fine-tuning)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)
- [Citation](#citation)

## Overview
This project implements a text classification model to categorize emotions such as joy, sadness, anger, etc., in given text data. The model is based on the BERT (Bidirectional Encoder Representations from Transformers) architecture, which has proven effective in various NLP tasks.

## Dataset
The dataset used in this project is the dair-ai/emotion dataset from Hugging Face's datasets library. The dataset consists of text samples labeled with different emotions.

- *Training set*: 16,000 samples
- *Validation set*: 2,000 samples
- *Test set*: 2,000 samples


## Exploratory Data Analysis (EDA)
The project includes an EDA step to understand the distribution of emotions in the dataset. This involves plotting the frequency and proportion of each emotion category in the training, validation, and test sets.

## Model Architecture
The model uses a pre-trained BERT model from Hugging Face's transformers library. The architecture is as follows:

- *BERT Model*: Extracts contextual embeddings for the input text.
- *Dense Layer*: A fully connected layer that maps the [CLS] token's output from BERT to the emotion categories using a softmax activation function.

## Training and Fine-Tuning
The BERT model is fine-tuned on the emotion classification task using the following settings:

- *Optimizer*: Adam with a learning rate of 1e-5
- *Loss Function*: Sparse Categorical Crossentropy
- *Metrics*: Accuracy
- *Epochs*: 8

During training, the model's performance is monitored on the validation set to prevent overfitting.

## Evaluation
The model's performance is evaluated on the test set, achieving the following results:

- *Test Accuracy*: 92.20%
- *Test Loss*: 0.2231

Confusion matrices and classification reports are also generated to analyze the model's performance across different emotion categories.

## Results
The fine-tuned BERT model shows high accuracy in classifying emotions in text, with consistent performance across training, validation, and test datasets. The model's learning curves and confusion matrices are available in the results section.


## License
This project is licensed under the MIT License. See the [LICENSE](https://huggingface.co/datasets/dair-ai/emotion#licensing-information) file for details.

---
## Citation


bibtex
@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697",
    abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
}
