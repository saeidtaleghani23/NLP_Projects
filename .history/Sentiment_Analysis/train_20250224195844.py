# install libraries
#pip install transformers datasets torch scikit-learn pandas numpy matplotlib seaborn

# import libraries
from transformers import AutoTokenizer # type:ignore
from datasets import load_dataset # type:ignore
from transformers import DataCollatorWithPadding # type:ignore
from torch.utils.data import DataLoader # type:ignore


# 1- import imdb, yelp, and amazon reviews datasets
dataset = load_dataset('imdb')
dataset = load_dataset('yelp_polarity')
dataset = load_dataset('amazon_polarity')

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

for model_name in ModelNames.values():
    tokenized_datasets[model_name] = dataset.map(lambda x: tokenize_function(x, model_name), batched=True)


# 4- Create Data Loaders

batch_size = 8  # Adjust as needed

data_collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"))

train_dataset = dataset["train"]
test_dataset = dataset["test"]

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)
