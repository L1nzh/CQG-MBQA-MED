#!/usr/bin/env python3
"""
PubMedQA Three-Class Classification Evaluation
Evaluates embedding models on PubMedQA yes/no/maybe classification task.
Follows methodology from paper sections 2.4.2, 2.5, and 3.1-3.3.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from tqdm import tqdm
import random
from openai import OpenAI

sys.path.append(os.path.dirname(__file__))

from mbqa_model import MultiTaskClassifier, MBQAMTEBModelWrapper
from utils import BagOfTokenEncoder

class LinearClassifier(nn.Module):
    """Simple linear classifier for three-class classification."""
    def __init__(self, input_dim, num_classes=3, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)

class BERTModelWrapper:
    """Wrapper for BERT-type models to extract [CLS] tokens."""
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
    def encode_with_cls(self, texts, max_length=512):
        """Extract [CLS] token embeddings for classification."""
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, 
            max_length=max_length, return_tensors="pt"
        )
        
        with torch.no_grad():
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Extract [CLS] token (first token) embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        return cls_embeddings.cpu().numpy()

class SBERTEncodingModelGeneral:
    """General SBERT encoder wrapper."""
    def __init__(self, model, device="cuda"):
        self.model = SentenceTransformer(model, device=device)
        
    def encode(self, sentences, **kwargs):
        return self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False, **kwargs)

class OpenAIEncodingModelGeneralWrapper:
    """OpenAI embedding model wrapper."""
    def __init__(self, model, api_key=None):
        self.model = model
        if api_key:
            self.client = OpenAI(api_key=api_key)
        elif os.environ.get("OPENAI_API_KEY"):
            self.client = OpenAI()
        else:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        embeddings = []
        batch_size = 100
        for i in tqdm(range(0, len(sentences), batch_size), desc=f"OpenAI {self.model}"):
            response = self.client.embeddings.create(input=sentences[i:i+batch_size], model=self.model)
            embeddings_batch = [item.embedding for item in response.data]
            embeddings.extend(embeddings_batch)
        return np.array(embeddings)

def get_model(model_name):
    """Get model instance based on model name."""
    dirname = os.path.dirname(__file__)
    
    if model_name == "CQG-MBQA":
        with open(os.path.join(dirname, "../checkpoints/CQG-MBQA/questions.json"), "r") as f:
            linear_questions = json.load(f)
        
        model = MultiTaskClassifier(num_labels=len(linear_questions), backbone="WhereIsAI/UAE-Large-V1")
        model.load_state_dict(torch.load(os.path.join(dirname, "../checkpoints/CQG-MBQA/multi_task_classifier_uae_3000000.pt"), map_location="cuda:0"))
        model.to("cuda")
        model.eval()
        mteb_model = MBQAMTEBModelWrapper(model, linear_questions, use_sigmoid=True)
    
    elif model_name == "QAEmb-MBQA":
        with open(os.path.join(dirname, "../checkpoints/QAEmb-MBQA/questions.json"), "r") as f:
            linear_questions = json.load(f)
        
        model = MultiTaskClassifier(num_labels=len(linear_questions), backbone="WhereIsAI/UAE-Large-V1")
        model.load_state_dict(torch.load(os.path.join(dirname, "../checkpoints/QAEmb-MBQA/multi_task_classifier_uae_3000000.pt"), map_location="cuda:0"))
        model.to("cuda")
        model.eval()
        mteb_model = MBQAMTEBModelWrapper(model, linear_questions, use_sigmoid=True)
    
    elif model_name == "bag_of_words":
        mteb_model = BagOfTokenEncoder()
    
    elif model_name == "openai":
        mteb_model = OpenAIEncodingModelGeneralWrapper('text-embedding-3-large')
    
    else:
        # Check if it's a BERT-type model that should use [CLS] tokens
        bert_models = [
            'google-bert/bert-base-uncased',
            'dmis-lab/biobert-v1.1',
            'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
            'emilyalsentzer/Bio_ClinicalBERT',
            'medicalai/ClinicalBERT'
        ]
        
        if model_name in bert_models:
            mteb_model = BERTModelWrapper(model_name)
        else:
            # Use sentence-transformers for other models
            mteb_model = SBERTEncodingModelGeneral(model_name)
    
    return mteb_model

def prepare_pubmedqa_data(train_ratio=0.45):
    """Load and prepare PubMedQA dataset with train/test split."""
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    
    # Get all data
    all_data = dataset['train']
    
    def format_input(question, context):
        """Format question + context as single input text."""
        return f"Question: {question} Context: {context}"
    
    # Prepare all texts and labels
    all_texts = []
    all_labels = []
    
    # Map labels: yes=0, no=1, maybe=2
    label_map = {"yes": 0, "no": 1, "maybe": 2}
    
    for item in all_data:
        question = item['question']
        context = " ".join(item['context']['contexts'])
        text = format_input(question, context)
        label = label_map[item['final_decision']]
        
        all_texts.append(text)
        all_labels.append(label)
    
    # Split into train/test (450 train, 550 test as per paper methodology)
    split_idx = int(len(all_texts) * train_ratio)
    
    # Shuffle data first for random split
    import random
    random.seed(42)
    indices = list(range(len(all_texts)))
    random.shuffle(indices)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_texts = [all_texts[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    
    test_texts = [all_texts[i] for i in test_indices]
    test_labels = [all_labels[i] for i in test_indices]
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    return (train_texts, train_labels), (test_texts, test_labels)

def train_classifier(embeddings, labels, input_dim, num_epochs=3, learning_rate=5e-5, batch_size=16, device="cuda"):
    """Train linear classifier on embeddings."""
    # Ensure embeddings are detached numpy arrays
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Convert to tensors 
    X = torch.FloatTensor(embeddings.copy()).to(device)  # Use .copy() to avoid gradient issues
    y = torch.LongTensor(labels).to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize classifier
    classifier = LinearClassifier(input_dim, num_classes=3, dropout=0.1).to(device)
    optimizer = optim.AdamW(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    classifier.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = classifier(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return classifier

def evaluate_classifier(classifier, embeddings, labels, device="cuda"):
    """Evaluate classifier on test embeddings."""
    classifier.eval()
    
    # Ensure embeddings are detached numpy arrays
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    X = torch.FloatTensor(embeddings.copy()).to(device)  # Use .copy() to avoid gradient issues
    
    with torch.no_grad():
        outputs = classifier(X)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    accuracy = accuracy_score(labels, predictions)
    return accuracy, predictions

def evaluate_model_on_pubmedqa(model, model_name, train_data, test_data, num_runs=10):
    """Evaluate a single model on PubMedQA with 10-run averaging."""
    train_texts, train_labels = train_data
    test_texts, test_labels = test_data
    
    accuracies = []
    
    for run in range(num_runs):
        # Set random seed for reproducibility
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        random.seed(42 + run)
        
        # Extract embeddings
        if hasattr(model, 'encode_with_cls'):
            # BERT-type model using [CLS] tokens
            train_embeddings = model.encode_with_cls(train_texts)
            test_embeddings = model.encode_with_cls(test_texts)
        else:
            # Other models using average pooling
            train_embeddings = model.encode(train_texts)
            test_embeddings = model.encode(test_texts)
        
        # Ensure embeddings are regular numpy arrays
        if not isinstance(train_embeddings, np.ndarray):
            train_embeddings = np.array(train_embeddings)
        if not isinstance(test_embeddings, np.ndarray):
            test_embeddings = np.array(test_embeddings)
        
        input_dim = train_embeddings.shape[1]
        
        # Train classifier
        classifier = train_classifier(
            train_embeddings, train_labels, input_dim,
            num_epochs=3, learning_rate=5e-5, batch_size=16
        )
        
        # Evaluate on test set
        accuracy, _ = evaluate_classifier(classifier, test_embeddings, test_labels)
        accuracies.append(accuracy)
        
        # Clear GPU memory
        del classifier
        torch.cuda.empty_cache()
    
    # Calculate statistics
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    return {
        'model': model_name,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'all_accuracies': accuracies
    }

def main():
    """Main evaluation function."""
    # Load PubMedQA data
    train_data, test_data = prepare_pubmedqa_data()
    
    # Define models to evaluate
    models = [
        # BERT-type models (use [CLS] tokens)
        'google-bert/bert-base-uncased',
        'dmis-lab/biobert-v1.1',
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
        'emilyalsentzer/Bio_ClinicalBERT',
        'medicalai/ClinicalBERT',
        
        # Other embedding models (use average pooling)
        'sentence-transformers/average_word_embeddings_glove.6B.300d',
        'princeton-nlp/unsup-simcse-bert-base-uncased',
        'princeton-nlp/sup-simcse-bert-base-uncased',
        'sentence-transformers/all-MiniLM-L12-v2',
        'BAAI/bge-base-en-v1.5',
        'openai',
        'abhinand/MedEmbed-small-v0.1',
        'abhinand/MedEmbed-base-v0.1',
        'abhinand/MedEmbed-large-v0.1',
        
        # Interpretable models
        'bag_of_words',
        'CQG-MBQA',
        'QAEmb-MBQA',
    ]
    
    results = {}
    
    # Evaluate models
    for model_name in tqdm(models, desc="Evaluating models"):
        try:
            model = get_model(model_name)
            result = evaluate_model_on_pubmedqa(model, model_name, train_data, test_data, num_runs=10)
            results[model_name] = result
            
            # Clear GPU memory
            if hasattr(model, 'model') and hasattr(model.model, 'to'):
                model.model.to('cpu')
            torch.cuda.empty_cache()
            
        except Exception as e:
            results[model_name] = {
                'model': model_name,
                'mean_accuracy': None,
                'error': str(e)
            }
    
    # Sort results by mean accuracy
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v.get('mean_accuracy') is not None],
        key=lambda x: x[1]['mean_accuracy'],
        reverse=True
    )
    
    # Print results
    print(f"{'Rank':<4} {'Model':<50} {'Mean Acc':<12} {'Std':<8}")
    print("-" * 80)
    
    for rank, (model_name, result) in enumerate(sorted_results, 1):
        mean_acc = result['mean_accuracy']
        std_acc = result['std_accuracy']
        print(f"{rank:<4} {model_name:<50} {mean_acc:<12.4f} ±{std_acc:.4f}")
    
    # Save results
    os.makedirs("../results", exist_ok=True)
    output_file = "../results/pubmedqa_evaluation.json"
    
    json_results = {}
    for model_name, result in results.items():
        json_results[model_name] = {
            'mean_accuracy': float(result['mean_accuracy']) if result.get('mean_accuracy') is not None else None,
            'std_accuracy': float(result.get('std_accuracy', 0)) if result.get('mean_accuracy') is not None else None,
            'all_accuracies': [float(acc) for acc in result.get('all_accuracies', [])] if result.get('all_accuracies') else None,
            'error': result.get('error', None)
        }
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': str(datetime.now()),
            'dataset': 'PubMedQA',
            'train_size': len(train_data[0]),
            'test_size': len(test_data[0]),
            'num_runs': 10,
            'models_evaluated': len([r for r in results.values() if r.get('mean_accuracy')]),
            'results': json_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()