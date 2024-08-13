# LLM-MODEL-ASSIGNMENT
# Text classification on the Rotten Tomatoes movie review using Bert model

---

This repository features a project dedicated to developing and fine-tuning a BERT-based model for text classification on movie review. The project utilizes the rotten tomotoes dataset from Hugging Face for training the model.

## Table of Contents
- [Introduction](#Introduction)
- [Dataset Overview)](#Dataset-overview)
- [Data Preprocessing](#Data-Preprocessing)
- [Model Fine-Tuning](#Model-fine-tuning)
- [Model Evaluation](#Model-Evaluation)
- [Results](#results)
- [Conclusion](#Conclusion)


## Introduction
This project implements a text classification, particularly sentiment analysis, stands as a fundamental task within Natural Language Processing (NLP). A commonly utilized resource for such binary classification endeavours is the Rotten Tomatoes movie review dataset, sourced from the Cornell Movie-Review Data collection. 


## Dataset Overview
The dataset used in this study comprises movie reviews from Rotten Tomatoes, labelled as either 
positive or negative. The dataset is divided into three subsets:

- *Training set*: 8,530 rows
- *Validation set*: 1,070 rows
- *Test set*: 1,070 rows
Each review in the dataset is assigned a binary sentiment label:
Positive (1): Indicates a favourable review.
Negative (0): Indicates an unfavourable review.


## Data Preprocessing
Before the data is fed into BERT, several preprocessing steps are essential. BERT uses a specific tokenizer that breaks down text into sub word tokens, aligning the input with its pre-trained vocabulary. Additionally, reviews are padded to ensure uniform input sizes for batch processing, and attention masks are created to distinguish between actual content and padding. 

## Model Fine-Tuning
BERT, initiated on a vast corpus of text, requires fine-tuning to adapt to particular tasks like sentiment analysis. Fine-tuning includes further training the model on the Rotten Tomatoes dataset while preserving its general awareness of language. 
The model is then trained on the 8,530 training rows, with the 1,070 validation rows used to monitor performance and adjust hyperparameters, such as learning rate and batch size, to optimize results. Techniques like early stopping are employed to avoid overfitting.

## Model Evaluation
After fine-tuning, the BERT model's performance is assessed using the 1,070 test rows. Key evaluation metrics include accuracy, precision, recall and F1-score which provide a comprehensive view of the model's capability to apply the training to new data.


## Results
Upon fine-tuning, BERT typically achieves high performance on the Rotten Tomatoes dataset. The model generally exhibits an accuracy rate between 90-95%, with similar values for precision, recall, and F1-score. These results demonstrate that BERT is highly effective at capturing the nuanced sentiment in movie reviews, making it a strong candidate for text classification tasks in NLP.

- *Test Accuracy*: 86.02%
- *Test Loss*: 1.3974
- *Evaluation runtime*: 3.70 seconds
- *Evaluation samples per second*: 288.136 samples/second
- *Evaluation steps per second*: 72.169 steps/second
- *Epoch*: 9.9953


## Conclusion
Fine-tuning BERT for sentiment analysis on the Rotten Tomatoes dataset, results in a model capable of accurately classifying movie reviews based on sentiment. BERT's success in this task is largely due to its deep bidirectional training and extensive pre-training on large text corpora. Additionally, examining the interpretability of BERT could provide deeper insights into how it processes and classifies text.


## License
This project is licensed under the MIT License. See the [LICENSE](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) file for details.

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
