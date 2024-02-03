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

# Results
---+------------------------------------+---------------------+----------------------+---------------------+-------------------------+
|   |               Model                |       ROUGE-1       |       ROUGE-2        |       ROUGE-L       |          BLEU          |
+---+------------------------------------+---------------------+----------------------+---------------------+------------------------+
| 0 |      facebook/bart-large-cnn       | 0.3111111070024692  | 0.07272726931570264  | 0.3111111070024692  | 4.231460607757819e-155 |
| 1 |        google/pegasus-large        | 0.11111110709876558 |         0.0          | 0.11111110709876558 | 6.190746313491463e-232 |
| 2 |              t5-large              | 0.3255813911303408  | 0.08695651788279792  | 0.3255813911303408  | 4.712028347038631e-155 |
| 3 |   sshleifer/distilbart-cnn-12-6    | 0.21052631128808874 | 0.049999995800000356 | 0.21052631128808874 | 3.884252021064659e-155 |
| 4 | microsoft/prophetnet-large-uncased |         0.0         |         0.0          |         0.0         |          0.0           |
+---+------------------------------------+---------------------+----------------------+---------------------+------------------------+

# With Topsis Score
-----------------------------------+--------------------+-------------------+--------------------+-------------------------+--------------------------+
|   |               Model                |      ROUGE-1       |      ROUGE-2      |      ROUGE-L       |          BLEU           |    TOPSIS Score    |
+---+------------------------------------+--------------------+-------------------+--------------------+-------------------------+--------------------+
| 0 |      facebook/bart-large-cnn       | 31.11111070024692  | 7.272726931570264 | 31.11111070024692  | 4.231460607757819e-153  |  89.6186986230545  |
| 1 |        google/pegasus-large        | 11.111110709876558 |        0.0        | 11.111110709876558 | 6.190746313491463e-230  | 21.629171084668883 |
| 2 |              t5-large              | 32.55813911303408  | 8.695651788279791 | 32.55813911303408  | 4.7120283470386307e-153 |       100.0        |
| 3 |   sshleifer/distilbart-cnn-12-6    | 21.052631128808873 | 4.999999580000035 | 21.052631128808873 | 3.884252021064659e-153  | 66.16399930420936  |
| 4 | microsoft/prophetnet-large-uncased |        0.0         |        0.0        |        0.0         |           0.0           |        0.0         |
+---+------------------------------------+--------------------+-------------------+--------------------+-------------------------+--------------------+

# Analysis

Overall Model Performance:
Rouge1, Rouge2, RougeL: The evaluation results present a clear distinction in performance among the models. Model "t5-large" consistently outperforms others in capturing unigrams (ROUGE-1) and bigrams (ROUGE-2), facebook/bart-large-cnn" also performs well across all ROUGE metrics, Models "google/pegasus-large" and "microsoft/prophetnet-large-uncased" show relatively lower scores.
BLEU: All models, except "microsoft/prophetnet-large-uncased," receive extremely low BLEU scores, indicating a poor match with reference text.
TOPSIS: The TOPSIS scores reveal that "t5-large" is the top-performing model overall, followed by "facebook/bart-large-cnn" while "microsoft/prophetnet-large-uncased" receives a TOPSIS score of 0, indicating its underperformance across all metrics.

ROUGE metrics contribute significantly to the overall TOPSIS scores, emphasizing the importance of n-gram precision and recall in model ranking.
