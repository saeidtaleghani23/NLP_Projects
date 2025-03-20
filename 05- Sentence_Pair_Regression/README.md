# Sentence Pair Regression with DistilBERT

Overview

This project fine-tunes the DistilBERT model for **sentence pair regression** using the Semantic Textual Similarity Benchmark (STS-B) dataset. The goal is to predict the semantic similarity score between two sentences on a scale of 0 to 5.

## 📚 Dataset
The dataset used is the STS-B dataset from the GLUE benchmark, which contains pairs of sentences along with human-annotated similarity scores.

## 🛠 Installation

### 1️⃣ Clone this repository:
```sh
git clone https://github.com/saeidtaleghani23/NLP_Projects.git
cd 05- Sentence_Pair_Regression
```

### 2️⃣ Install dependencies:
``` sh
conda env create -f env.yml
```

## 🏋️ Model and Training
- The project uses DistilBert tokenizer and DistilBert for score regression.
- Training is performed using Hugging Face's Trainer API with evaluation metrics like accuracy, precision, recall, and F1-score.

## 📊 Results

example1_1 = "The cat sat on the mat."

example1_2 = "The feline rest


Predicted similarity between example sentences: 0.07 (scale 0-5)

