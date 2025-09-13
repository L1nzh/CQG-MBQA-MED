# CQG-MBQA Framework Optimization Status

## Current State: MEDICAL EVALUATION PHASE 🔬

### Completed Components
- [√] General domain CQG-MBQA framework analysis
- [√] Pretrained model checkpoint validation (9,614 questions)
- [√] Embedding analysis tool (`check_embeddings.py`) with interpretability metrics
- [√] Cognitive burden analysis: 200+ high-confidence questions per document identified
- [√] OpenAI API integration and testing
- [√] GPU management protocol established
- [√] Development environment setup (conda: cqgmbqa)
- [√] Framework structure understanding

### System Architecture
```
CQG-MBQA Framework
├── Pretrained Models
│   ├── CQG-MBQA (general domain)
│   └── QAEmb-MBQA (comparison baseline)
├── Core Framework
│   ├── cqg.py (Contrastive Question Generation model)
│   ├── mbqa.py (Multi-task Binary QA training)
│   ├── mbqa_model.py (model architecture definitions)
│   ├── qaemb.py (QAEmb model implementation)
│   └── utils.py (utility functions)
├── Data Processing
│   └── preprocess-medi2.py (MEDI2 dataset preprocessing)
├── Evaluation Framework
│   ├── run-mteb-*.py (MTEB evaluation for 3 interpretable models)
│   │   ├── Bag of Words
│   │   ├── QAEmb-MBQA  
│   │   └── CQG-MBQA
│   ├── msmarco.py (10 model evaluation: 7 black-box + 3 interpretable)
│   ├── newspectrum.py (10 model evaluation: 7 black-box + 3 interpretable)
│   └── eval_biosses.py (BIOSSES biomedical similarity: 9 models comprehensive)
└── Documentation
    ├── CQG.md, MBQA.md (component docs)
    └── Research workflow docs
```

### Framework Components Detail
**Core Models:**
- **cqg.py**: Contrastive Question Generation - generates discriminative yes/no questions
- **mbqa.py**: Multi-task Binary QA - trains classifier on generated questions  
- **mbqa_model.py**: Model architecture definitions and wrapper classes
- **qaemb.py**: QAEmb baseline model implementation

**Data Processing:**
- **preprocess-medi2.py**: Preprocesses MEDI2 dataset for framework training

**Evaluation Suite:**
- **run-mteb-*.py**: MTEB benchmark evaluation for 3 interpretable embedding models
- **msmarco.py & newspectrum.py**: Comprehensive evaluation of 10 models total:
  - 7 black-box models (Sentence-BERT, OpenAI, GloVe, BERT, SimCSE, etc.)
  - 3 interpretable models (Bag of Words, QAEmb-MBQA, CQG-MBQA)

### Research Environment Status
- **Hardware**: NVIDIA H100 GPUs available (8 total)
- **Software**: Python 3.9, PyTorch, sentence-transformers ready
- **Data**: MEDI2 dataset available (~70GB)
- **Compute**: GPU management protocol tested and working

### Current Research Phase: Medical Baseline Evaluation
- [√] **BIOSSES Comprehensive Evaluation Completed** (9 models: 6 black-box + 3 interpretable)
  
  **🏆 FINAL COMPLETE RANKINGS:**
  1. **OpenAI text-embedding-3-large**: r = 0.8787 (Very strong) - *Best overall* 🥇
  2. **all-MiniLM-L12-v2**: r = 0.8657 (Very strong) 
  3. **unsup-simcse-bert**: r = 0.6850 (Strong)
  4. **sup-simcse-bert**: r = 0.6707 (Strong)
  5. **bag_of_words**: r = 0.6661 (Strong) - *Best interpretable* 🏅
  6. **CQG-MBQA**: r = 0.6377 (Strong) - *Our model* 🎯
  7. **bert-base-uncased**: r = 0.5326 (Moderate)
  8. **QAEmb-MBQA**: r = 0.4890 (Moderate)
  9. **GloVe embeddings**: r = 0.3005 (Weak)
  
  **Key Insights:**
  - **OpenAI dominates** with highest correlation (0.8787)
  - **CQG-MBQA solid 6th place** - competitive performance among all models
  - **Strong interpretability**: CQG-MBQA only 0.03 behind best interpretable model
  - **Clear winner over QAEmb**: 30% better correlation validates our approach
  - **Evaluation script** now integrated into framework (`framework/eval_biosses.py`)

- [ ] MIMIC dataset integration and preprocessing
- [ ] MedSTS evaluation framework implementation  
- [ ] Medical domain embedding quality analysis on larger datasets

### Next Research Phases (Post-Evaluation)
- [ ] Question quality analysis and filtering strategies
- [ ] CQG hyperparameter optimization for better discrimination
- [ ] Medical domain-specific adaptations based on baseline results
- [ ] Interpretability evaluation with reduced cognitive burden

### Current Optimization Targets
- **Quality**: Improve logit separation (current: AI Healthcare 265 questions >0.95, target: <50)
- **Discrimination**: Increase gap between high/low confidence scores
- **Efficiency**: Reduce cognitive load while maintaining semantic quality
- **Selectivity**: Generate fewer but more meaningful discriminative questions

### Research Focus Areas
- Threshold optimization strategies (current thresholds: 0.95/0.9 for high, 0.05/0.1 for low)
- Question filtering and ranking mechanisms
- Training objective modifications for better polarization
- Interpretability metrics and human evaluation protocols

### Experimental Parameters (Established)
```python
# Hyperparameters from paper
k = 5000              # Clusters for question generation
theta = 0.8           # Similarity threshold
t = 4                 # Top questions per cluster
learning_rate = 1e-4  # MBQA training rate
num_steps = 3000000   # Training iterations
seed = 42             # Reproducibility
```