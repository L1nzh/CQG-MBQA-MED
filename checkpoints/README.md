# Model Checkpoints

This directory contains trained model weights for the CQG-MBQA framework.

## Available Models

### CQG-MBQA
- **File**: `CQG-MBQA/multi_task_classifier_uae_3000000.pt` (~1.6GB)
- **Questions**: `CQG-MBQA/questions.json`
- **Description**: Contrastive Question Generation + Multi-Binary QA model trained on PubMedQA dataset

### QAEmb-MBQA  
- **File**: `QAEmb-MBQA/multi_task_classifier_uae_3000000.pt` (~1.6GB)
- **Questions**: `QAEmb-MBQA/questions.json`
- **Description**: Question-Answer Embedding + Multi-Binary QA model

## Download Instructions

Due to file size limitations, model weights are not included in this repository. 

To use the framework:

1. **Contact the authors** for access to the trained model weights
2. **Place model files** in their respective directories:
   ```
   checkpoints/
   ├── CQG-MBQA/
   │   ├── multi_task_classifier_uae_3000000.pt
   │   └── questions.json
   └── QAEmb-MBQA/
       ├── multi_task_classifier_uae_3000000.pt
       └── questions.json
   ```

3. **Alternative**: Train your own models using the provided scripts:
   ```bash
   # For PubMedQA dataset
   OPENAI_API_KEY="your-key" CUDA_VISIBLE_DEVICES=0 conda run -n cqgmbqa python framework/run-cqg-mbqa.py
   
   # For MIMIC-IV dataset  
   OPENAI_API_KEY="your-key" CUDA_VISIBLE_DEVICES=0 conda run -n cqgmbqa python framework/run-cqg-mbqa-mimic.py
   ```

## Model Architecture

Both models use:
- **Backbone**: `WhereIsAI/UAE-Large-V1` (or `abhinand/MedEmbed-large-v0.1`)
- **Architecture**: Multi-task classifier with binary QA heads
- **Training**: 3M steps with contrastive learning

## Usage

Models are automatically loaded by the evaluation scripts:
```python
from framework.mbqa_model import MultiTaskClassifier, MBQAMTEBModelWrapper

# CQG-MBQA model
model = get_model("CQG-MBQA")

# QAEmb-MBQA model  
model = get_model("QAEmb-MBQA")
```