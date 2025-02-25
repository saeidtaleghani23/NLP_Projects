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
import yaml

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def tokenize_function(examples, tokenizer):
    """ Tokenizes input text for the model """
    return tokenizer(examples['text'], padding="max_length", truncation=True)


def train_model(model_name, dataset, batch_size, num_epochs):
    """
    Trains a transformer model for text classification.

    Args:
        model_name (str): Name of the transformer model.
        dataset (DatasetDict): Tokenized dataset for training and evaluation.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs for training.

    Returns:
        dict: Evaluation results.
    """
    print(f"\nTraining model: {model_name}")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification
    
    # Tokenize dataset using a lambda function to pass the tokenizer
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=2,  # Keep only last 2 checkpoints
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    # trainer.evaluate() method returns a dictionary with evaluation metrics, including accuracy.
    # Evaluate and return results
    eval_results = trainer.evaluate()
    return eval_results

# 0- load the configuration file
with open('./config/config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Extract training configurations
batch_size = config['TRAIN']['batch_size']
num_epochs = config['TRAIN']['num_epochs']

# 1- import imdb, yelp, and amazon reviews datasets
# dataset = load_dataset('imdb')
# dataset = load_dataset('yelp_polarity')
# dataset = load_dataset('amazon_polarity')
dataset = load_dataset(config['MODEL']['dataset'])


# 2- define the models
ModelNames= {"BERT": "bert-base-uncased", 
             "DistilBERT": "distilbert-base-uncased", 
             "RoBERTa": "roberta-base", 
             "ALBERT": "albert-base-v2",
             "XLNet": "xlnet-base-cased", 
}


batch_size = config['TRAIN']['batch_size']  
num_epochs = config['TRAIN']['num_epochs']
all_accuracy= {}
for key, model_name in ModelNames.items():
    eval_results = train_model(model_name, dataset, batch_size, num_epochs)
    all_accuracy[key] = eval_results.get("eval_accuracy", "N/A")  # Extract accuracy

print(f"\nEvaluation results: {all_accuracy}")





