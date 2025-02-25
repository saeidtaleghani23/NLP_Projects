# install libraries
#pip install transformers datasets torch scikit-learn pandas numpy matplotlib seaborn

# import libraries
from transformers import AutoTokenizer # type:ignore
from datasets import load_dataset # type:ignore
import torch # type:ignore
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments # type:ignore
from sklearn.metrics import accuracy_score # type:ignore
import yaml
import os
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    """_summary_

    Args:
        eval_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get the predicted class label
    return {"accuracy": accuracy_score(labels, predictions)}



def train_model(model_name, dataset, config):
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
    

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to GPU if available


    # Check dataset text key dynamically
    text_key = "text" if "text" in dataset["train"].column_names else dataset["train"].column_names[-1]

    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda x: tokenizer(x[text_key], padding="max_length", truncation=True), batched=True)

    # saving results
    output_dir_results = os.path.join(config['TRAIN']['output_dir'], model_name)
    os.makedirs(output_dir_results, exist_ok=True)  
    # saving logs
    logging_dir = os.path.join(config['TRAIN']['logging_dir'], model_name)
    os.makedirs(logging_dir, exist_ok=True)  


    training_args = TrainingArguments(
        output_dir=output_dir_results,
        eval_strategy="epoch", # eval_strategy evaluation_strategy
        save_strategy="epoch",
        logging_dir=logging_dir,
        logging_steps=config['TRAIN']['logging_steps'],  #   
        per_device_train_batch_size=config['TRAIN']['batch_size'] ,
        per_device_eval_batch_size= config['TRAIN']['batch_size'] ,
        num_train_epochs=config['TRAIN']['num_epochs']  ,
        weight_decay=config['TRAIN']['weight_decay'],  #   
        save_total_limit=config['TRAIN']['save_total_limit'],  # Keep only last 2 checkpoints  
        learning_rate=config['TRAIN']['learning_rate'],
        warmup_ratio=config['TRAIN']['warmup_ratio'],  # Warm-up for the first 10% of total training steps  
        lr_scheduler_type= config['TRAIN']['lr_scheduler_type'] ,  # Learning rate scheduler type
        max_grad_norm= config['TRAIN']['max_grad_norm']   # Clipping threshold (default: 1.0)
        
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

    # Save model & tokenizer
    output_dir_model = f"./saved_models/{model_name}"
    # Create the directory if it doesn't exist
    os.makedirs(output_dir_model, exist_ok=True)  # Recommended for saving trained models
    model.save_pretrained(output_dir_model)
    tokenizer.save_pretrained(output_dir_model)


    # trainer.evaluate() method returns a dictionary with evaluation metrics, including accuracy.
    # Evaluate and return results
    eval_results = trainer.evaluate()
    return eval_results


if __name__ == "__main__":
    # 0- load the configuration file
    with open('./config/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)


    #  Import imdb, yelp, and amazon reviews datasets
    # dataset = load_dataset('imdb')
    # dataset = load_dataset('yelp_polarity')
    # dataset = load_dataset('amazon_polarity')
    dataset = load_dataset(config['DATASET']['dataset'])


    # Define the models
    ModelNames= config['MODEL']

    # Train and evaluate models
    all_accuracy = {}
    for key, model_name in ModelNames.items():
        eval_results = train_model(model_name, dataset, config)
        all_accuracy[key] = eval_results.get("eval_accuracy", "N/A")

    print(f"\nEvaluation results: {all_accuracy}")





