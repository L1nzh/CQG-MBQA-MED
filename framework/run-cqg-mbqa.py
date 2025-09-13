from cqg import ContrastiveQuestionGeneration
from mbqa import MBQA
import json
import logging
import os
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("Starting pubmed")

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, "../data/pubmedqa_documents_5M.json"), "r") as f:
        doc_texts = json.load(f)
    
    logger.info(f"Loaded {len(doc_texts)} documents")
        
    cqg = ContrastiveQuestionGeneration(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="pubmedqa",
        encoder="abhinand/MedEmbed-large-v0.1",
        k=2500  # Use 2500 clusters for PubMedQA dataset
    )
    cqg.generate_questions()
    
    mbqa = MBQA(
        corpus=doc_texts,
        temp_folder="./temp",
        output_folder="./output",
        name="pubmedqa",
        backbone="abhinand/MedEmbed-large-v0.1"
    )
    mbqa.collect_training_data_with_cqg()
    mbqa.train_model()