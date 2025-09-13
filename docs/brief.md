# CQG-MBQA Framework Optimization - Project Brief

## Overview
Optimizing the CQG-MBQA (Contrastive Question Generation - Multi-task Binary Question Answering) framework to improve embedding quality and interpretability, with future application to medical domain specialization.

## Core Functionality

### Research Objectives
1. **Medical Baseline Evaluation**: Assess current general CQG-MBQA performance on medical texts (MIMIC) using MedSTS
2. **Domain Transfer Analysis**: Understand how well general questions perform on medical content
3. **Embedding Quality**: Improve discrimination by increasing logit separation (high vs low scores)
4. **Interpretability Enhancement**: Reduce cognitive burden by minimizing high-confidence questions per document
5. **Framework Optimization**: Data-driven medical domain adaptation based on evaluation results

### Key Components
- **Contrastive Question Generation (CQG)**: Generate highly discriminative yes/no questions with better selectivity
- **Multi-task Binary QA (MBQA)**: Train classifier to produce more polarized confidence scores (high vs low)
- **Interpretable Embeddings**: Each dimension represents confidence in answering a specific question, optimized for human interpretation

### Current Challenge Analysis
- **Cognitive Overload**: Documents currently trigger 200+ high-confidence questions (e.g., AI Healthcare doc: 265 questions >0.95)
- **Discrimination Quality**: Need better separation between high/low confidence scores for clearer interpretation
- **Question Efficiency**: Too many relevant questions reduce human analytical focus

## Technical Requirements
- **Framework**: Python 3.9+ with PyTorch
- **Models**: Sentence transformers, GPT-4o for question generation  
- **Data**: General domain corpora (MEDI2), with future medical specialization
- **Evaluation**: MTEB benchmark for embedding quality assessment

## Success Metrics
- **Medical Baseline**: Establish CQG-MBQA performance on MIMIC dataset using MedSTS evaluation
- **Domain Transfer**: Analyze which general domain questions are effective for medical texts
- **Quality**: Increase logit separation (more questions <0.1, fewer questions >0.9)
- **Interpretability**: Reduce high-confidence questions per document (target: <50 questions >0.9)
- **Medical Relevance**: Identify gaps requiring medical domain-specific adaptations

## Target Applications
- **Current**: General domain text analysis with improved interpretability
- **Future**: Medical domain specialization (clinical texts, patient records, healthcare research)
- **Research**: Framework optimization techniques applicable to domain-specific adaptations