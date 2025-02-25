# install libraries
#pip install transformers datasets torch scikit-learn pandas numpy matplotlib seaborn
# Optional: pip install tensorboard

# import libraries
import os
import time
import yaml
import numpy as np
import torch  # type:ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments  # type:ignore
from datasets import load_dataset  # type:ignore
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # type:ignore
from typing import Dict, Any, Optional


def compute_metrics(eval_pred):
    """Calculate metrics for model evaluation.

    Args:
        eval_pred (tuple): Tuple containing logits and labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 metrics.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get the predicted class label
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate precision, recall, and F1 (for binary classification)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def tokenize_dataset(dataset, tokenizer, text_key=None, max_length=512):
    """Tokenize dataset with specified tokenizer.
    
    Args:
        dataset: Dataset to tokenize
        tokenizer: Tokenizer to use
        text_key: Column name containing text (auto-detected if None)
        max_length: Maximum sequence length
        
    Returns:
        Tokenized dataset
    """
    if text_key is None:
        text_key = "text" if "text" in dataset["train"].column_names else dataset["train"].column_names[-1]
    
    print(f"Tokenizing dataset using column: '{text_key}'")
    
    return dataset.map(
        lambda x: tokenizer(
            x[text_key],
            padding="max_length",
            truncation=True,
            max_length=max_length
        ), 
        batched=True
    )


def is_tensorboard_available():
    """Check if tensorboard is available."""
    try:
        import tensorboard
        return True
    except ImportError:
        try:
            import tensorboardX
            return True
        except ImportError:
            return False


def train_model(model_name: str, dataset, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trains a transformer model for text classification.

    Args:
        model_name (str): Name of the transformer model.
        dataset: Dataset for training and evaluation.
        config (Dict[str, Any]): Configuration dictionary with training parameters.

    Returns:
        dict: Evaluation results.
    """
    start_time = time.time()
    print(f"\nTraining model: {model_name}")

    try:
        # Load tokenizer & model
        print(f"Loading model and tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification
        
        # Check if GPU is available and set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)  # Move model to GPU if available

        # Check dataset text key dynamically and tokenize
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)
        print(f"Dataset tokenized. Train size: {len(tokenized_dataset['train'])}, Test size: {len(tokenized_dataset['test'])}")

        # Create output directories
        output_dir_results = os.path.join(config['TRAIN']['output_dir'], model_name)
        os.makedirs(output_dir_results, exist_ok=True)  
        
        logging_dir = os.path.join(config['TRAIN']['logging_dir'], model_name)
        os.makedirs(logging_dir, exist_ok=True)
        
        # Check if tensorboard is available
        tensorboard_available = is_tensorboard_available()
        if not tensorboard_available:
            print("TensorBoard not available. Disabling TensorBoard reporting.")
        
        # Set up training arguments
        print(f"Setting up training with batch size: {config['TRAIN']['batch_size']}, epochs: {config['TRAIN']['num_epochs']}")
        training_args = TrainingArguments(
            output_dir=output_dir_results,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=logging_dir,
            logging_steps=config['TRAIN']['logging_steps'],  
            per_device_train_batch_size=config['TRAIN']['batch_size'],
            per_device_eval_batch_size=config['TRAIN']['batch_size'],
            num_train_epochs=config['TRAIN']['num_epochs'],
            weight_decay=config['TRAIN']['weight_decay'],    
            save_total_limit=config['TRAIN']['save_total_limit'],
            learning_rate=float(config['TRAIN']['learning_rate']),
            warmup_ratio=config['TRAIN']['warmup_ratio'],
            lr_scheduler_type=config['TRAIN']['lr_scheduler_type'],
            max_grad_norm=config['TRAIN']['max_grad_norm'],
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            # Only enable tensorboard if available
            report_to=["tensorboard"] if tensorboard_available else []
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        # Train model
        print(f"Starting training for {model_name}...")
        training_result = trainer.train()
        
        # Log training stats
        train_runtime = round(training_result.metrics["train_runtime"], 2)
        print(f"Training completed in {train_runtime} seconds")

        # Save model & tokenizer
        model_save_dir = config['TRAIN'].get('model_save_dir', './saved_models')
        output_dir_model = os.path.join(model_save_dir, model_name)
        os.makedirs(output_dir_model, exist_ok=True)
        
        print(f"Saving model and tokenizer to {output_dir_model}")
        model.save_pretrained(output_dir_model)
        tokenizer.save_pretrained(output_dir_model)
        
        # Evaluate model
        print(f"Evaluating model: {model_name}")
        eval_results = trainer.evaluate()
        
        # Add training time and metadata to results
        training_time = time.time() - start_time
        eval_results['training_time'] = training_time
        eval_results['training_time_formatted'] = f"{training_time:.2f} seconds"
        eval_results['model_name'] = model_name
        eval_results['dataset'] = config['DATASET']['dataset']
        eval_results['batch_size'] = config['TRAIN']['batch_size']
        eval_results['epochs'] = config['TRAIN']['num_epochs']
        
        # Print evaluation results
        print(f"Evaluation results for {model_name}:")
        print(f"  - Accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"  - F1 Score: {eval_results.get('eval_f1', 'N/A'):.4f}")
        print(f"  - Training time: {eval_results['training_time_formatted']}")
        
        return eval_results
        
    except Exception as e:
        print(f"Error training model {model_name}: {str(e)}")
        return {
            "error": str(e),
            "model_name": model_name,
            "status": "failed"
        }


def main():
    """Main function to load config, datasets and train models."""
    try:
        # Load the configuration file
        config_path = './config/config.yml'
        print(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create directories if they don't exist
        os.makedirs(config['TRAIN']['output_dir'], exist_ok=True)
        os.makedirs(config['TRAIN']['logging_dir'], exist_ok=True)
        os.makedirs(config['TRAIN'].get('model_save_dir', './saved_models'), exist_ok=True)
        os.makedirs(config['TRAIN'].get('results_dir', './results'), exist_ok=True)

        # Load dataset
        dataset_name = config['DATASET']['dataset']
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        print(f"Dataset loaded successfully: {dataset_name}")
        print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

        # Train and evaluate models
        all_results = {}
        model_names = config['MODEL']
        
        # Print training configuration summary
        print(f"\n{'='*50}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"Batch size: {config['TRAIN']['batch_size']}")
        print(f"Epochs: {config['TRAIN']['num_epochs']}")
        print(f"Learning rate: {config['TRAIN']['learning_rate']}")
        print(f"Models to train: {', '.join(model_names.keys())}")
        print(f"{'='*50}\n")
        
        # Train each model
        for key, model_name in model_names.items():
            print(f"\n{'='*50}")
            print(f"Training model {key}: {model_name}")
            print(f"{'='*50}")
            
            eval_results = train_model(model_name, dataset, config)
            all_results[key] = eval_results
        
        # Print final comparison
        print(f"\n{'='*50}")
        print("FINAL EVALUATION RESULTS")
        print(f"{'='*50}")
        
        # Sort models by accuracy for better comparison
        sorted_results = sorted(
            [(key, results) for key, results in all_results.items() if "error" not in results],
            key=lambda x: x[1].get("eval_accuracy", 0),
            reverse=True
        )
        
        for key, results in sorted_results:
            model_name = model_names[key]
            accuracy = results.get("eval_accuracy", "N/A")
            f1 = results.get("eval_f1", "N/A")
            training_time = results.get("training_time_formatted", "N/A")
            
            print(f"Model: {key} ({model_name})")
            print(f"  - Accuracy: {accuracy:.4f}")
            print(f"  - F1 Score: {f1:.4f}")
            print(f"  - Training time: {training_time}")
        
        # Print failures if any
        failures = [(key, results) for key, results in all_results.items() if "error" in results]
        if failures:
            print(f"\nFAILED MODELS:")
            for key, results in failures:
                print(f"  - {key} ({model_names[key]}): {results['error']}")
        
        # Save results to file
        results_dir = config['TRAIN'].get('results_dir', './results')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(results_dir, f"model_comparison_{timestamp}.yml")
        
        with open(results_file, 'w') as f:
            yaml.dump(all_results, f)
        
        print(f"\nResults saved to {results_file}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()