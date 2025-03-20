his project implements a **binary sentiment classification** model using **DistilBERT**. It is trained on the **IMDB dataset** and predicts whether a given text has a **positive** or **negative** sentiment. 

## 🚀 Features

- **Uses DistilBERT** for sequence classification
- **Trains on the IMDB dataset** with optimized preprocessing
- **Implements early stopping** and TensorBoard logging
- **Supports evaluation on a test set**
- **Predicts sentiment on custom text inputs**

## 🛠 Installation

### 1️⃣ Clone this repository:
```sh
git clone https://github.com/saeidtaleghani23/NLP_Projects.git
cd binary_sentence_classification
```

### 2️⃣ Install dependencies:
``` sh
conda env create -f env.yml
```

## 📊 Dataset
The project uses the IMDB dataset from the Hugging Face datasets library.

- Train Set: imdb_train
- Validation Set: imdb_val
- Test Set: imdb_test

## 📜 Data Preprocessing
- Tokenization: Uses DistilBertTokenizerFast with truncation & padding.
- Concatenation: Splits test data into test, val, and train subsets.
- Format Conversion: Converts dataset to PyTorch tensors.

## ✅ Sample Output

Text: This movie was absolutely fantastic! I loved every minute of it.
Sentiment: Positive (confidence: 0.9994)
--------------------------------------------------
Text: What a waste of time. Terrible acting and boring plot.
Sentiment: Negative (confidence: 0.9998)
--------------------------------------------------

