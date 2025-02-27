# Sentiment Analysis with Transformer Models

## Overview
This project implements sentiment analysis using four transformer-based models: **BERT, DistilBERT, RoBERTa, and ALBERT**. It supports three different datasets: **IMDB, Yelp Polarity, and Amazon Polarity**. The models are trained and evaluated using standard classification metrics.

## Applications

- **Social Media Insights**  
  Monitor and interpret public sentiment on social platforms to understand trends and user opinions in real time.

- **Brand Reputation Tracking**  
  Analyze online discussions and reviews to assess how customers perceive a brand and detect potential crises early.

- **Customer Feedback Evaluation**  
  Process and categorize customer reviews to pinpoint strengths and areas needing improvement in products or services.

- **Sentiment Analysis Across Languages**  
  Identify and analyze emotions in text data from various languages, making sentiment analysis more inclusive and globally applicable.


## Project Structure
```
├── config/               # Configuration files
│   ├── config.yml        # User-defined training settings
├── env/                  # Environment setup files
│   ├── env.yml           # Conda environment file
├── logs/                 # logs the training process
├── results/              # Stores evaluation results
├── saved_models/         # Directory for saving trained models
├── train.py              # Training functions
├── README.md             # Project documentation
```

## Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/saeidtaleghani23/NLP_Projects.git
cd NLP_Projects/Sentiment_Analysis
```

### **2. Set Up the Environment**
Create a Conda environment using the provided `env.yml` file:
```bash
conda env create -f env/env.yml
conda activate sentiment_env
```

## Usage
### **Run the Training Script**
To train and evaluate models, run:
```bash
python train.py --dataset imdb --config config/config.yml
```
Replace `imdb` with `yelp_polarity` or `amazon_polarity` as needed.

## Configuration
Modify `config/config.yml` to adjust training settings, model selection, batch size, learning rate, etc.

Example configuration:
```yaml
DATASET:
  dataset: imdb

MODEL:
  BERT: bert-base-uncased
  DistilBERT: distilbert-base-uncased
  RoBERTa: roberta-base
  ALBERT: albert-base-v2

TRAIN:
  batch_size: 32
  num_epochs: 3
  learning_rate: 5e-5
  output_dir: ./saved_models
  logging_dir: ./logs
  results_dir: ./results
```

## Results
After training, the models achieved the following evaluation results:
```
==================================================
FINAL EVALUATION RESULTS
==================================================
Model: RoBERTa (roberta-base)
  - Accuracy: 0.9471
  - F1 Score: 0.9472
  - Training time: 50647.13 seconds
Model: BERT (bert-base-uncased)
  - Accuracy: 0.9328
  - F1 Score: 0.9325
  - Training time: 50693.49 seconds
Model: ALBERT (albert-base-v2)
  - Accuracy: 0.9272
  - F1 Score: 0.9254
  - Training time: 57387.47 seconds
Model: DistilBERT (distilbert-base-uncased)
  - Accuracy: 0.9251
  - F1 Score: 0.9253
  - Training time: 25787.94 seconds
```


## 🙋‍♂️ Contact

For any inquiries or contributions, feel free to reach out:

- **Name:** Saeid 
- **Email:** stalegha@uwaterloo.ca 
- **GitHub:** [saeidtaleghani23](https://github.com/saeidtaleghani23)