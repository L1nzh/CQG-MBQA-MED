from cqg import ContrastiveQuestionGeneration
from mbqa import MBQA
import json
import logging
import os
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("Starting MIMIC-IV CQG-MBQA Training")

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    
    # Load MIMIC preprocessed documents
    with open(os.path.join(dirname, "../data/mimic_documents.json"), "r") as f:
        doc_texts = json.load(f)
    
    logger.info(f"Loaded {len(doc_texts)} MIMIC-IV document chunks")
    
    # Initialize CQG with MedEmbed encoder
    cqg = ContrastiveQuestionGeneration(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="mimic",
        encoder="abhinand/MedEmbed-large-v0.1",
        k=1500  # Use 1500 clusters for MIMIC dataset
    )
    cqg.generate_questions()
    
    # Initialize MBQA with MedEmbed backbone
    mbqa = MBQA(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="mimic",
        backbone="abhinand/MedEmbed-large-v0.1"
    )
    mbqa.collect_training_data_with_cqg()
    mbqa.train_model()