# TEXT_SUMMARIZATION
Text summarization is the process of generating a concise and condensed version of a given text while retaining its essential information. The goal of text summarization is to produce a shorter representation of the original document, preserving its key ideas, concepts, and main points. This can be useful for various purposes, such as quickly understanding the content of a document, extracting important information, or creating concise summaries for news articles, research papers, and other types of text.

# About
This repository contains code for evaluating different pre-trained models for text summarization using various evaluation metrics such as ROUGE and BLEU. The evaluation results are then ranked using the TOPSIS method.

# Prerequisites
Make sure you have the necessary Python libraries installed. You can install them using the following command:
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Usage
Clone the repository:
git clone https://github.com/Kashishsingla111/TEXT_SUMMARIZATION

# Models used
model_names = [ "facebook/bart-large-cnn", "google/pegasus-large", "t5-large", "sshleifer/distilbart-cnn-12-6", "microsoft/prophetnet-large-uncased" ]

# Evaluation Parameters
Evaluation Parameters used: ROGUE AND BLEU
ROGUE: Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in natural language processing.
BLEU: BiLingual Evaluation Understudy, is a metric for automatically evaluating machine-translated text. The BLEU score is a number between zero and one that measures the similarity of the machine-translated text to a set of high quality reference translations.
Rogue_1 -> for unigrams
Rogue_2 -> for bigrams
Rogue_l -> for longest common subsequence

 
# Results With Topsis Score
| Model                               | ROUGE-1            | ROUGE-2            | ROUGE-L            | BLEU               | TOPSIS Score      |
| ----------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ----------------- |
| facebook/bart-large-cnn             | 31.11              | 7.27               | 31.11              | 4.23e-153          | 89.62             |
| google/pegasus-large                | 11.11              | 0.00               | 11.11              | 6.19e-230          | 21.63             |
| t5-large                            | 32.56              | 8.70               | 32.56              | 4.71e-153          | 100.00            |
| sshleifer/distilbart-cnn-12-6       | 21.05              | 5.00               | 21.05              | 3.88e-153          | 66.16             |
| microsoft/prophetnet-large-uncased  | 0.00               | 0.00               | 0.00               | 0.00               | 0.00             |


# Analysis

Overall Model Performance:
Rouge1, Rouge2, RougeL: The evaluation results present a clear distinction in performance among the models. Model "t5-large" consistently outperforms others in capturing unigrams (ROUGE-1) and bigrams (ROUGE-2), facebook/bart-large-cnn" also performs well across all ROUGE metrics, Models "google/pegasus-large" and "microsoft/prophetnet-large-uncased" show relatively lower scores.
BLEU: All models, except "microsoft/prophetnet-large-uncased," receive extremely low BLEU scores, indicating a poor match with reference text.
TOPSIS: The TOPSIS scores reveal that "t5-large" is the top-performing model overall, followed by "facebook/bart-large-cnn" while "microsoft/prophetnet-large-uncased" receives a TOPSIS score of 0, indicating its underperformance across all metrics.

ROUGE metrics contribute significantly to the overall TOPSIS scores, emphasizing the importance of n-gram precision and recall in model ranking.
