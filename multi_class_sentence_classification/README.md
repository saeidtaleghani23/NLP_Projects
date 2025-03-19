# Turkish Multi-Class Text Classification

This project implements a **multi-class text classification** model for the Turkish language using **BERT (dbmdz/bert-base-turkish-uncased)**. The model is fine-tuned on the **TTC4900 dataset**, which contains Turkish news articles categorized into multiple topics.

## 📚 Dataset

The dataset used is **TTC4900**, which contains Turkish text samples categorized into the following labels:

- 🖥 **teknoloji** (Technology)
- 💰 **ekonomi** (Economy)
- 🏥 **saglik** (Health)
- 🏛 **siyaset** (Politics)
- 🎭 **kultur** (Culture)
- ⚽ **spor** (Sports)
- 🌍 **dunya** (World News)

The dataset is automatically downloaded if not found in the project directory.


## 🛠 Installation

### 1️⃣ Clone this repository:
```sh
git clone https://github.com/saeidtaleghani23/NLP_Projects.git
cd multi_class_sentence_classification
```

### 2️⃣ Install dependencies:
``` sh
conda env create -f env.yml
```

## 🏋️ Model and Training
- The project uses BERT tokenizer and BERT for sequence classification.
- Datasets are tokenized and converted to PyTorch tensors.
- Training is performed using Hugging Face's Trainer API with evaluation metrics like accuracy, precision, recall, and F1-score.


## 🚀 Training
Run the training script:
```sh
## 5- Fine-tuning the model cell
```

Training arguments:

- Epochs: 15
- Batch Size: 16
- Metric for Best Model: F1-score
- Logging & Checkpoints: TensorBoard and step-based saving

## 🎯 Evaluation and Testing

After training, the model is evaluated using accuracy, precision, recall, and F1-score.

```sh
## 6- Evaluate on test set
```

## 📊 Results
- The best model is saved in multi_class_results_<timestamp>/
- Training logs are stored in multi_class_logs_<timestamp>/
- Performance metrics are printed after training.


