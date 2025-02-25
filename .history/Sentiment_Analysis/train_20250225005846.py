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
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1).numpy()
    return {"accuracy": accuracy_score(labels, predictions)}



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
    
    # Check dataset text key dynamically
    text_key = "text" if "text" in dataset["train"].column_names else dataset["train"].column_names[-1]

    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda x: tokenizer(x[text_key], padding="max_length", truncation=True), batched=True)

    # Training arguments
    output_dir_results = f"./results/{model_name}"
    os.makedirs(output_dir_results, exist_ok=True)  # Recommended for saving results

    logging_dir = f"./logs/{model_name}"
    os.makedirs(logging_dir, exist_ok=True)  # Recommended for saving logs
    training_args = TrainingArguments(
        output_dir=output_dir_results,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=logging_dir,
        logging_steps=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=2,  # Keep only last 2 checkpoints
        warmup_ratio=0.1,  # Warm-up for the first 10% of total training steps
        lr_scheduler_type="cosine"  ,
        max_grad_norm=1.0  # Clipping threshold (default: 1.0)
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

    # Extract training configurations
    batch_size = config['TRAIN']['batch_size']  
    num_epochs = config['TRAIN']['num_epochs']
    # Train and evaluate models
    all_accuracy = {}
    for key, model_name in ModelNames.items():
        eval_results = train_model(model_name, dataset, batch_size, num_epochs)
        all_accuracy[key] = eval_results.get("eval_accuracy", "N/A")

    print(f"\nEvaluation results: {all_accuracy}")





