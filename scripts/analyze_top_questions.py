#!/usr/bin/env python3
"""
Analyze CQG-MBQA model results on health/medical documents from MEDI2 dataset.
Finds top-scoring questions for each document and saves results to JSON.
"""

import json
import os
import torch
import numpy as np
from typing import List, Dict, Tuple
import random
from framework.mbqa_model import MultiTaskClassifier

def load_model_and_questions():
    """Load the CQG-MBQA model and questions."""
    print("Loading model and questions...")
    
    # Load questions
    with open("checkpoints/CQG-MBQA/questions.json", "r") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")
    
    # Load model
    model = MultiTaskClassifier(num_labels=len(questions), backbone="WhereIsAI/UAE-Large-V1")
    model.load_state_dict(torch.load("checkpoints/CQG-MBQA/multi_task_classifier_uae_3000000.pt", map_location="cuda:0"))
    model.to("cuda")
    model.eval()
    
    return model, questions

def load_medi2_documents(sample_size: int = 1000) -> List[str]:
    """Load a sample of documents from MEDI2 dataset."""
    print(f"Loading {sample_size} documents from MEDI2 dataset...")
    
    documents = []
    try:
        with open("data/medi2_documents.json", "r") as f:
            # Read the file in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1MB chunks
            content = ""
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                content += chunk
                # Try to parse when we have complete JSON objects
                if content.count('"') > sample_size * 2:  # Rough estimate
                    break
            
            # Try to extract some documents from the partial content
            # Look for complete document strings
            lines = content.split('\n')
            for line in lines[:sample_size]:
                if line.strip() and not line.strip().startswith('[') and not line.strip().endswith(']'):
                    # Clean up the line to extract document text
                    cleaned = line.strip().rstrip(',').strip('"')
                    if len(cleaned) > 50:  # Only take substantial documents
                        documents.append(cleaned)
                    if len(documents) >= sample_size:
                        break
    except Exception as e:
        print(f"Error loading MEDI2 documents: {e}")
        # Fallback: create some sample medical documents for demonstration
        documents = create_sample_medical_documents()
    
    print(f"Loaded {len(documents)} documents")
    return documents

def create_sample_medical_documents() -> List[str]:
    """Create sample medical documents for demonstration."""
    return [
        "Diabetes mellitus is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period. Symptoms often include frequent urination, increased thirst, and increased hunger. If left untreated, diabetes can cause many complications, including cardiovascular disease, stroke, and kidney disease.",
        
        "Hypertension, also known as high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated. High blood pressure typically does not cause symptoms, but long-term high blood pressure is a major risk factor for coronary artery disease and heart failure.",
        
        "Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body. Not all tumors are cancerous; benign tumors do not spread to other parts of the body. Treatment options may include surgery, chemotherapy, radiation therapy, and immunotherapy.",
        
        "Alzheimer's disease is a progressive disorder that causes brain cells to waste away and die. It is the most common cause of dementia, which is characterized by a continuous decline in thinking, behavioral and social skills that disrupts daily functioning. Early symptoms include forgetting recent events or conversations.",
        
        "COVID-19 is a contagious disease caused by severe acute respiratory syndrome coronavirus 2. Most people infected develop mild to moderate illness and recover without special treatment. However, some people develop serious illness requiring medical attention, particularly older adults and those with underlying medical conditions.",
        
        "Depression is a mood disorder that causes persistent feelings of sadness and loss of interest. It affects how you feel, think and behave and can lead to a variety of emotional and physical problems. Treatment typically involves medications, psychotherapy or both.",
        
        "Asthma is a respiratory condition marked by attacks of spasm in the bronchi of the lungs, causing difficulty breathing. It is usually connected to allergic reactions or other forms of hypersensitivity. Common triggers include allergens, exercise, cold air, and stress.",
        
        "Arthritis is inflammation of one or more joints, causing pain and stiffness that can worsen with age. The most common types of arthritis are osteoarthritis and rheumatoid arthritis. Treatment focuses on relieving symptoms and improving quality of life through medication, physical therapy, and lifestyle changes.",
        
        "Heart disease refers to several types of heart conditions, including coronary artery disease, arrhythmias, and congenital heart defects. The most common type in the United States is coronary artery disease, which can lead to heart attack. Risk factors include high blood pressure, high cholesterol, and smoking.",
        
        "Stroke occurs when blood flow to an area of brain is cut off. When this happens, brain cells are deprived of oxygen and begin to die. There are two main types of stroke: ischemic stroke and hemorrhagic stroke. Early treatment is crucial for better outcomes and reduced disability."
    ]

def filter_health_medical_documents(documents: List[str], max_docs: int = 10) -> List[str]:
    """Filter documents to find health/medical related content."""
    print(f"Filtering for health/medical documents...")
    
    medical_keywords = [
        'health', 'medical', 'disease', 'patient', 'treatment', 'therapy', 'clinical',
        'hospital', 'doctor', 'medicine', 'drug', 'diagnosis', 'symptom', 'syndrome',
        'cancer', 'diabetes', 'hypertension', 'infection', 'surgery', 'pharmaceutical',
        'healthcare', 'therapeutic', 'pathology', 'anatomy', 'physiology', 'immunology',
        'cardiology', 'neurology', 'psychiatry', 'radiology', 'biochemistry', 'genetics'
    ]
    
    medical_docs = []
    for doc in documents:
        doc_lower = doc.lower()
        if any(keyword in doc_lower for keyword in medical_keywords):
            medical_docs.append(doc)
            if len(medical_docs) >= max_docs:
                break
    
    print(f"Found {len(medical_docs)} health/medical documents")
    return medical_docs

def get_question_segments_for_document(model, questions: List[str], document: str) -> Dict:
    """Get top 20, middle 20, and bottom 20 scoring questions for a document."""
    with torch.no_grad():
        # Get embeddings for the document
        logits = model([document]).cpu().numpy().flatten()
        
        # Apply sigmoid to get probabilities/scores
        scores = 1 / (1 + np.exp(-logits))
        
        # Get sorted indices
        sorted_indices = np.argsort(scores)
        
        # Get top 20, middle 20, bottom 20
        total_questions = len(questions)
        
        # Top 20 (highest scores)
        top_20_indices = sorted_indices[-20:][::-1]
        top_20 = [{"question": questions[idx], "score": float(scores[idx])} for idx in top_20_indices]
        
        # Middle 20 (around median)
        middle_start = (total_questions - 20) // 2
        middle_20_indices = sorted_indices[middle_start:middle_start + 20]
        middle_20 = [{"question": questions[idx], "score": float(scores[idx])} for idx in middle_20_indices]
        
        # Bottom 20 (lowest scores)
        bottom_20_indices = sorted_indices[:20]
        bottom_20 = [{"question": questions[idx], "score": float(scores[idx])} for idx in bottom_20_indices]
        
        return {
            "top_20": top_20,
            "middle_20": middle_20,
            "bottom_20": bottom_20
        }

def main():
    """Main analysis function."""
    print("Starting CQG-MBQA analysis on health/medical documents...")
    
    # Load model and questions
    model, questions = load_model_and_questions()
    
    # Load and filter documents
    all_documents = load_medi2_documents(sample_size=2000)
    medical_documents = filter_health_medical_documents(all_documents, max_docs=10)
    
    if not medical_documents:
        print("No medical documents found, using sample documents")
        medical_documents = create_sample_medical_documents()
    
    # Analyze each document
    results = {}
    print("\nAnalyzing documents...")
    
    for i, doc in enumerate(medical_documents):
        print(f"Analyzing document {i+1}/{len(medical_documents)}")
        
        # Truncate document for display
        doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
        
        # Get question segments
        question_segments = get_question_segments_for_document(model, questions, doc)
        
        results[f"document_{i+1}"] = {
            "document_preview": doc_preview,
            "full_document": doc,
            "question_analysis": question_segments
        }
        
        # Print results for this document
        print(f"  Document: {doc_preview}")
        print("  Top 5 questions:")
        for j, item in enumerate(question_segments["top_20"][:5], 1):
            print(f"    {j}. {item['question'][:60]}... ({item['score']:.6f})")
        print()
    
    # Save results to JSON with clean formatting
    output_file = "medical_document_analysis_results.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, separators=(',', ': '))
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total documents analyzed: {len(medical_documents)}")
    print(f"Questions per document: Top 20 + Middle 20 + Bottom 20 = 60")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()