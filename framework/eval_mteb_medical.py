#!/usr/bin/env python3
"""
Medical MTEB Comprehensive Evaluation using proper MTEB framework
Follows the pattern from run-mteb-cqg-mbqa.py and run-mteb-qaemb-mbqa.py
"""

import os
import sys
import json
import torch
import mteb
import pandas as pd
import glob
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))

from mbqa_model import MultiTaskClassifier, MBQAMTEBModelWrapper
from utils import BagOfTokenEncoder

# Import model wrappers for other models
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np

class SBERTEncodingModelGeneral:
    """General SBERT encoder wrapper for MTEB."""
    def __init__(self, model, device="cuda"):
        self.model = SentenceTransformer(model, device=device)
        
    def encode(self, sentences, **kwargs):
        # Avoid duplicate parameters that might be passed by MTEB
        kwargs.setdefault('show_progress_bar', False)
        
        # Remove conflicting parameters if they exist
        convert_to_tensor = kwargs.pop('convert_to_tensor', True)
        convert_to_numpy = kwargs.pop('convert_to_numpy', False)
        
        # Always get tensor first for proper device handling
        embeddings = self.model.encode(sentences, convert_to_tensor=True, **kwargs)
        
        # Move to CPU if it's a CUDA tensor
        if hasattr(embeddings, 'cpu'):
            embeddings = embeddings.cpu()
        
        # Convert to numpy if needed
        if convert_to_numpy or not convert_to_tensor:
            return embeddings.numpy() if hasattr(embeddings, 'numpy') else embeddings
        else:
            return embeddings

class OpenAIEncodingModelGeneralWrapper:
    """OpenAI embedding model wrapper for MTEB."""
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

def get_model_wrapper(model_name):
    """Get model wrapper for MTEB evaluation."""
    dirname = os.path.dirname(__file__)
    
    if model_name == "CQG-MBQA":
        with open(os.path.join(dirname, "../checkpoints/CQG-MBQA/questions.json"), "r") as f:
            linear_questions = json.load(f)
        
        model = MultiTaskClassifier(num_labels=len(linear_questions), backbone="WhereIsAI/UAE-Large-V1")
        model.load_state_dict(torch.load(os.path.join(dirname, "../checkpoints/CQG-MBQA/multi_task_classifier_uae_3000000.pt"), map_location="cuda:0"))
        model.to("cuda")
        model.eval()
        return MBQAMTEBModelWrapper(model, linear_questions, is_binary=True, is_sparse=False, binary_threshold=0.5, use_sigmoid=True)
    
    elif model_name == "QAEmb-MBQA":
        with open(os.path.join(dirname, "../checkpoints/QAEmb-MBQA/questions.json"), "r") as f:
            linear_questions = json.load(f)
        
        model = MultiTaskClassifier(num_labels=len(linear_questions), backbone="WhereIsAI/UAE-Large-V1")
        model.load_state_dict(torch.load(os.path.join(dirname, "../checkpoints/QAEmb-MBQA/multi_task_classifier_uae_3000000.pt"), map_location="cuda:0"))
        model.to("cuda")
        model.eval()
        return MBQAMTEBModelWrapper(model, linear_questions, is_binary=True, is_sparse=False, binary_threshold=0.5, use_sigmoid=True)
    
    elif model_name == "bag_of_words":
        return BagOfTokenEncoder()
    
    elif model_name == "openai":
        return OpenAIEncodingModelGeneralWrapper('text-embedding-3-large')
    
    else:
        # Use sentence-transformers for other models
        return SBERTEncodingModelGeneral(model_name)

def get_medical_tasks():
    """Get medical MTEB tasks that are confirmed to work."""
    medical_tasks = [
        # STS tasks
        # mteb.get_task("BIOSSES", languages=["eng"]),
        
        # Retrieval tasks
        mteb.get_task("CUREv1", languages=["eng"]), #clinical
        mteb.get_task("R2MEDIIYiClinicalRetrieval", languages=["eng"]), #clinical
        mteb.get_task("R2MEDPMCClinicalRetrieval", languages=["eng"]), #clinical
        # mteb.get_task("NFCorpus", languages=["eng"]),
        mteb.get_task("CmedqaRetrieval", languages=["cmn"]), #chinese clinical
        # mteb.get_task("PublicHealthQA", languages=["eng"]),
        # mteb.get_task("MedicalQARetrieval", languages=["eng"]),
        # mteb.get_task("SciFact", languages=["eng"]),
        # mteb.get_task("ArguAna", languages=["eng"]),
        # mteb.get_task("TRECCOVID", languages=["eng"]),
        
        # Clustering tasks
        # mteb.get_task("BiorxivClusteringP2P", languages=["eng"]),
        # mteb.get_task("BiorxivClusteringS2S", languages=["eng"]),
        # mteb.get_task("MedrxivClusteringP2P", languages=["eng"]),
        # mteb.get_task("MedrxivClusteringS2S", languages=["eng"]),
        # mteb.get_task("ClusTREC-Covid", languages=["eng"]),
    ]
    
    return medical_tasks

def evaluate_single_model(model_name, tasks):
    """Evaluate a single model on medical MTEB tasks."""
    print(f"\n=== Evaluating {model_name} ===")
    
    try:
        # Get model wrapper
        model_wrapper = get_model_wrapper(model_name)
        
        # Create MTEB evaluation with medical tasks
        evaluation = mteb.MTEB(tasks=tasks)
        
        # Create output directory
        output_folder = os.path.join(os.path.dirname(__file__), f"../results/evaluation/model_outputs/mteb_medical_{model_name.replace('/', '_')}")
        os.makedirs(output_folder, exist_ok=True)
        
        # Run evaluation
        results = evaluation.run(model_wrapper, output_folder=output_folder)
        
        print(f"✅ {model_name}: Evaluation completed")
        
        # Clear GPU memory
        if hasattr(model_wrapper, 'model') and hasattr(model_wrapper.model, 'to'):
            model_wrapper.model.to('cpu')
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"❌ {model_name}: Error - {e}")
        return None

def check_evaluation_completeness(models, tasks):
    """Check if all models have been evaluated on all tasks."""
    base_dir = os.path.join(os.path.dirname(__file__), "../results/evaluation/model_outputs")
    
    missing_evaluations = []
    
    for model_name in models:
        model_dir_name = f"mteb_medical_{model_name.replace('/', '_')}"
        model_path = os.path.join(base_dir, model_dir_name, "no_model_name_available", "no_revision_available")
        
        if not os.path.exists(model_path):
            missing_evaluations.append(f"{model_name}: Directory not found")
            continue
            
        for task in tasks:
            task_name = task.metadata.name
            task_file = os.path.join(model_path, f"{task_name}.json")
            
            if not os.path.exists(task_file):
                missing_evaluations.append(f"{model_name}: Missing {task_name}")
    
    return missing_evaluations

def generate_results_csv(models, tasks):
    """Generate CSV with all model scores on all tasks."""
    print("📊 Generating comprehensive results CSV...")
    
    base_dir = os.path.join(os.path.dirname(__file__), "../results/evaluation/model_outputs")
    results_data = []
    
    for model_name in models:
        model_dir_name = f"mteb_medical_{model_name.replace('/', '_')}"
        model_path = os.path.join(base_dir, model_dir_name, "no_model_name_available", "no_revision_available")
        
        row = {"Model": model_name}
        task_scores = []
        
        for task in tasks:
            task_name = task.metadata.name
            task_file = os.path.join(model_path, f"{task_name}.json")
            
            if os.path.exists(task_file):
                try:
                    with open(task_file, 'r') as f:
                        task_results = json.load(f)
                    
                    # Extract main score - handle different task formats
                    main_score = 'N/A'
                    
                    if 'scores' in task_results:
                        scores = task_results['scores']
                        
                        # Try common keys in order of preference
                        score_keys = ['test', 'dev', 'all']  # Most common evaluation split names
                        
                        for score_key in score_keys:
                            if score_key in scores:
                                target_scores = scores[score_key]
                                if isinstance(target_scores, list) and len(target_scores) > 0:
                                    main_score = target_scores[0].get('main_score', 'N/A')
                                elif isinstance(target_scores, dict):
                                    main_score = target_scores.get('main_score', 'N/A')
                                
                                if main_score != 'N/A':
                                    break
                        
                        # Ultimate fallback: try to find main_score in any available key
                        if main_score == 'N/A':
                            for key, value in scores.items():
                                if isinstance(value, dict) and 'main_score' in value:
                                    main_score = value['main_score']
                                    break
                                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and 'main_score' in value[0]:
                                    main_score = value[0]['main_score']
                                    break
                    
                    row[task_name] = main_score
                    if isinstance(main_score, (int, float)):
                        task_scores.append(main_score)
                        
                except Exception as e:
                    print(f"Warning: Error reading {task_file}: {e}")
                    row[task_name] = 'Error'
            else:
                row[task_name] = 'Missing'
        
        # Calculate average score
        if task_scores:
            row["Average"] = round(sum(task_scores) / len(task_scores), 4)
        else:
            row["Average"] = 'N/A'
            
        results_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Reorder columns to have Model first, then tasks, then Average
    task_columns = [task.metadata.name for task in tasks]
    column_order = ["Model"] + task_columns + ["Average"]
    df = df[column_order]
    
    # Save CSV
    output_dir = os.path.join(os.path.dirname(__file__), "../results")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "medical_mteb_comprehensive_results.csv")
    
    df.to_csv(csv_path, index=False)
    
    # Display results
    print("\n📋 Medical MTEB Results Summary:")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\n✅ Results saved to: {csv_path}")
    
    return df

def main():
    """Main evaluation function with completeness checking."""
    print("=== Medical MTEB Comprehensive Evaluation (Proper MTEB Framework) ===")
    
    # Get medical tasks
    medical_tasks = get_medical_tasks()
    
    print(f"Medical tasks to evaluate:")
    for task in medical_tasks:
        print(f"  - {task.metadata.name}: {task.metadata.category}")
    
    # Define models to evaluate
    models = [
        # General Models
        # 'google-bert/bert-base-uncased',
        # 'sentence-transformers/average_word_embeddings_glove.6B.300d', 
        # 'princeton-nlp/unsup-simcse-bert-base-uncased',
        # 'princeton-nlp/sup-simcse-bert-base-uncased',
        # 'sentence-transformers/all-MiniLM-L12-v2',
        # 'BAAI/bge-base-en-v1.5',
        # 'openai',
        
        # Medical Models
        # pubmed-based
        # 'abhinand/MedEmbed-small-v0.1',
        'abhinand/MedEmbed-base-v0.1',
        # 'abhinand/MedEmbed-large-v0.1', 
        # 'dmis-lab/biobert-v1.1',
        # 'allenai/scibert_scivocab_uncased',
        'ncbi/MedCPT-Query-Encoder',
        # 'bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16',
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
        'abhinand/MedEmbed-large-v0.1',
        
        # clinical
        'medicalai/ClinicalBERT',
        'emilyalsentzer/Bio_ClinicalBERT',
        'UFNLP/gatortron-base',
        
        
        # Interpretable Models
        # 'bag_of_words',
        # 'CQG-MBQA',
        # 'QAEmb-MBQA',
    ]
    
    print(f"\nChecking completeness for {len(models)} models on {len(medical_tasks)} medical tasks...")
    
    # Check if all evaluations are complete
    missing_evaluations = check_evaluation_completeness(models, medical_tasks)
    
    if not missing_evaluations:
        print("✅ All evaluations are complete!")
        print("📊 Generating comprehensive results CSV...")
        
        # Generate CSV results
        results_df = generate_results_csv(models, medical_tasks)
        
        # Save comprehensive results summary JSON
        output_file = os.path.join(os.path.dirname(__file__), "../results/medical_mteb_comprehensive_results.json")
        
        # Create summary
        summary = {
            'timestamp': str(datetime.now()),
            'evaluation_type': 'Medical MTEB Comprehensive (Proper Framework)',
            'total_models': len(models),
            'successful_models': len(models),  # All complete
            'total_tasks': len(medical_tasks),
            'tasks': [{'name': task.metadata.name, 'category': task.metadata.category} for task in medical_tasks],
            'models_evaluated': models,
            'results_directories': [f"../results/evaluation/model_outputs/mteb_medical_{model.replace('/', '_')}" for model in models],
            'csv_results_file': "../results/medical_mteb_comprehensive_results.csv"
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ All evaluations complete!")
        print(f"📁 CSV results: /home/zhenghong/CQG-MBQA/results/medical_mteb_comprehensive_results.csv")
        print(f"📁 JSON summary: {output_file}")
        
        return results_df
        
    else:
        print(f"❌ Found {len(missing_evaluations)} missing evaluations:")
        for missing in missing_evaluations:
            print(f"  - {missing}")
        
        print(f"\n🔄 Starting evaluation for missing models/tasks...")
        
        # Store all results
        all_results = {}
        
        # Evaluate each model
        for model_name in models:
            model_results = evaluate_single_model(model_name, medical_tasks)
            if model_results is not None:
                all_results[model_name] = model_results
        
        # Save comprehensive results summary
        output_file = os.path.join(os.path.dirname(__file__), "../results/medical_mteb_comprehensive_results.json")
        
        # Create summary
        summary = {
            'timestamp': str(datetime.now()),
            'evaluation_type': 'Medical MTEB Comprehensive (Proper Framework)',
            'total_models': len(models),
            'successful_models': len(all_results),
            'total_tasks': len(medical_tasks),
            'tasks': [{'name': task.metadata.name, 'category': task.metadata.category} for task in medical_tasks],
            'models_evaluated': list(all_results.keys()),
            'results_directories': [f"../results/evaluation/model_outputs/mteb_medical_{model.replace('/', '_')}" for model in all_results.keys()]
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Evaluation complete!")
        print(f"📁 Summary saved to: {output_file}")
        print(f"📁 Individual results in: ../results/evaluation/model_outputs/mteb_medical_* directories")
        print(f"✅ Successfully evaluated: {len(all_results)}/{len(models)} models")
        
        # Try to generate CSV with available results
        if len(all_results) > 0:
            print("\n📊 Generating partial results CSV...")
            try:
                results_df = generate_results_csv(models, medical_tasks)
                return results_df
            except Exception as e:
                print(f"⚠️  Could not generate CSV: {e}")
        
        return all_results

if __name__ == "__main__":
    results = main()