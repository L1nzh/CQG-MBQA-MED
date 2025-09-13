# CQG-MBQA Research Workflow

## Research Process

### 1. Framework Analysis
- Examine existing CQG-MBQA codebase in `framework/` directory
- Understand current architecture and data flow
- Review pretrained models in `checkpoints/`
- Analyze question generation patterns in existing `questions.json`

### 2. Medical Domain Planning
- Reference `CLAUDE.md` for framework adaptation constraints
- Update `status.md` with current research state
- Plan incremental modifications for medical domain
- Identify medical-specific question categories

### 3. Implementation Phases
- **Phase 1**: Medical corpus preparation and preprocessing
- **Phase 2**: Medical question generation using adapted CQG
- **Phase 3**: MBQA training on medical question-answer pairs
- **Phase 4**: Evaluation and benchmarking

### 4. Experimental Validation
- Test on medical text samples
- Compare with general domain performance
- Validate question quality and relevance
- Benchmark against medical NLP baselines

### 5. Documentation and Results
- Log all experimental parameters
- Document question generation patterns
- Record performance metrics
- Update research documentation files

## GPU Management Workflow
```bash
# Step 1: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Step 2: Check available GPUs
nvidia-smi

# Step 3: Switch to free GPU and run
CUDA_VISIBLE_DEVICES=N python script.py
```

## Environment Setup
```bash
# Activate research environment
conda activate cqgmbqa

# Run CQG pipeline
python framework/cqg.py

# Run MBQA training
python framework/mbqa.py

# Evaluate embeddings
python check_embeddings.py
```

## Required Hyperparameters
```python
# CQG Parameters (from paper)
k = 5000          # Number of clusters
theta = 0.8       # Similarity threshold
t = 4             # Questions per cluster
seed = 42         # Reproducibility

# MBQA Parameters
learning_rate = 1e-4
num_steps = 3000000
backbone = "WhereIsAI/UAE-Large-V1"
```

## Data Processing Pipeline

### Medical Text Preparation
1. **Corpus Collection**: Gather medical texts (literature, clinical notes, patient records)
2. **Preprocessing**: Clean and format medical documents
3. **Quality Filtering**: Remove low-quality or irrelevant content
4. **Format Conversion**: Convert to CQG-MBQA compatible JSON format

### Question Generation Standards
1. **Medical Relevance**: Questions must be specific to medical domain
2. **Binary Format**: All questions answerable with yes/no
3. **Discriminative Power**: Questions should distinguish between different medical concepts
4. **Interpretability**: Questions should be clinically meaningful

### Evaluation Protocol
- Use MTEB benchmark adapted for medical tasks
- Compare against general domain CQG-MBQA
- Measure interpretability through question analysis
- Validate clinical relevance with domain experts

## Framework Modification Rules
1. **Preserve Core Architecture**: Maintain CQG-MBQA pipeline structure
2. **Incremental Changes**: Test each modification independently
3. **Backup Originals**: Keep copies of original framework files
4. **Validation Required**: Test on known examples before full deployment
5. **Documentation**: Record all changes and their rationales