# install libraries
#pip install transformers datasets torch scikit-learn pandas numpy matplotlib seaborn

# import libraries
from transformers import AutoTokenizer # type:ignore
from datasets import load_dataset # type:ignore
from transformers import DataCollatorWithPadding # type:ignore
from torch.utils.data import DataLoader # type:ignore
import torch # type:ignore
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments # type:ignore
from sklearn.metrics import accuracy_score # type:ignore


# 1- import imdb, yelp, and amazon reviews datasets
dataset = load_dataset('imdb')
dataset = load_dataset('yelp_polarity')
dataset = load_dataset('amazon_polarity')
dataset = load_dataset(config['MODEL']['dataset'])
# 2- define the models
ModelNames= {"BERT": "bert-base-uncased", 
             "DistilBERT": "distilbert-base-uncased", 
             "RoBERTa": "roberta-base", 
             "ALBERT": "albert-base-v2",
             "XLNet": "xlnet-base-cased", 
             "GPT-2": "gpt2"}

# 3- define the tokenizer
# each model needs its own tokenizer
def tokenize_function(examples, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = {}

batch_size = config['TRAIN']['batch_size']  


for model_name in ModelNames.values():
    tokenized_datasets[model_name] = dataset.map(lambda x: tokenize_function(x, model_name), batched=True)
    # 4- Create Data Loaders

    
    data_collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(model_name))

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)



