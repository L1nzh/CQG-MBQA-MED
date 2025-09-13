#!/usr/bin/env python3
"""
Preprocess PubMedQA dataset for CQG-MBQA framework
Read from existing JSON files in /data/pubmed/chunk/
Extract 'contents' by combining title and content, truncate at 5000 characters
"""

import json
import os
import glob
import logging
import random
from typing import List, Dict, Any
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tokenizer (global for multiprocessing)
tokenizer = AutoTokenizer.from_pretrained("abhinand/MedEmbed-large-v0.1")

def extract_contents_field(sample: Dict[str, Any]) -> str:
    """
    Extract the 'contents' field from PubMed sample.
    
    Args:
        sample: PubMed sample containing 'contents' field
    
    Returns:
        Contents text or empty string if not found
    """
    # The data already has a 'contents' field that combines title and content
    if 'contents' in sample and sample['contents']:
        return sample['contents'].strip()
    
    # Fallback: if 'contents' field is missing, try to reconstruct it
    # This matches the example structure: title + content = contents
    if 'title' in sample and 'content' in sample:
        title = sample['title'].strip() if sample['title'] else ""
        content = sample['content'].strip() if sample['content'] else ""
        if title and content:
            return f"{title} {content}"
        elif title:
            return title
        elif content:
            return content
    
    # If individual fields exist but no contents
    for field in ['title', 'content', 'abstract', 'text']:
        if field in sample and sample[field]:
            return sample[field].strip()
    
    return ""

def process_sample(sample: Dict[str, Any], min_tokens: int = 128, max_tokens: int = 512) -> str:
    """
    Process a single sample: extract content and filter by token count.
    
    Args:
        sample: PubMed sample
        min_tokens: Minimum token count
        max_tokens: Maximum token count
    
    Returns:
        Processed text or empty string if filtered out
    """
    try:
        # Extract content
        contents = extract_contents_field(sample)
        if not contents.strip():
            return ""
        
        # Fast token count using tokenizer
        tokens = tokenizer.encode(contents, add_special_tokens=False, truncation=False)
        token_count = len(tokens)
        
        if min_tokens <= token_count <= max_tokens:
            return contents
        else:
            return ""
    except Exception:
        return ""

def filter_by_token_count(text: str, min_tokens: int = 128, max_tokens: int = 512) -> str:
    """Legacy function for compatibility"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_count = len(tokens)
    
    if token_count < min_tokens or token_count > max_tokens:
        return ""
    
    return text

def read_jsonl_files(data_dir: str = "/home/zhenghong/CQG-MBQA/data/pubmed/chunk") -> List[Dict[str, Any]]:
    """
    Read all JSONL files from the specified directory.
    JSONL format: one JSON object per line
    
    Args:
        data_dir: Directory containing JSONL files
    
    Returns:
        List of all samples from JSONL files
    """
    logger.info(f"Reading JSONL files from: {data_dir}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Look for both .jsonl and .json files
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    all_files = jsonl_files + json_files
    if not all_files:
        raise FileNotFoundError(f"No JSONL/JSON files found in: {data_dir}")
    
    logger.info(f"Found {len(jsonl_files)} JSONL files and {len(json_files)} JSON files")
    
    all_samples = []
    for data_file in all_files:
        try:
            logger.info(f"Reading file: {os.path.basename(data_file)}")
            
            if data_file.endswith('.jsonl'):
                # JSONL format: one JSON object per line
                with open(data_file, 'r', encoding='utf-8') as f:
                    file_samples = []
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                sample = json.loads(line)
                                file_samples.append(sample)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Error parsing line {line_num} in {data_file}: {e}")
                                continue
                    all_samples.extend(file_samples)
                    logger.info(f"Loaded {len(file_samples)} samples from {os.path.basename(data_file)}")
                    
            else:
                # Regular JSON format
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    # List of samples
                    all_samples.extend(data)
                    logger.info(f"Loaded {len(data)} samples from {os.path.basename(data_file)}")
                elif isinstance(data, dict):
                    # Single sample or nested structure
                    if 'data' in data:
                        # Nested structure with 'data' key
                        samples = data['data']
                        if isinstance(samples, list):
                            all_samples.extend(samples)
                            logger.info(f"Loaded {len(samples)} samples from {os.path.basename(data_file)}")
                        else:
                            all_samples.append(samples)
                            logger.info(f"Loaded 1 sample from {os.path.basename(data_file)}")
                    else:
                        # Single sample
                        all_samples.append(data)
                        logger.info(f"Loaded 1 sample from {os.path.basename(data_file)}")
            
        except Exception as e:
            logger.warning(f"Error reading {data_file}: {e}")
            continue
    
    logger.info(f"Total samples loaded: {len(all_samples)}")
    return all_samples

def preprocess_pubmedqa(input_dir: str = "/home/zhenghong/CQG-MBQA/data/pubmed/chunk", 
                       output_dir: str = "/home/zhenghong/CQG-MBQA/data") -> List[str]:
    """
    Preprocess PubMedQA dataset from JSONL files for CQG-MBQA framework.
    
    Args:
        input_dir: Directory containing JSONL files
        output_dir: Directory to save processed data
    
    Returns:
        List of processed document contents
    """
    logger.info(f"Preprocessing PubMedQA data from {input_dir}")
    
    try:
        # Read all JSONL files
        all_samples = read_jsonl_files(input_dir)
        
        if not all_samples:
            raise ValueError("No samples found in JSON files")
        
        
        # Show sample structure for debugging
        logger.info(f"Sample keys: {list(all_samples[0].keys())}")
        logger.info(f"First sample preview: {str(all_samples[0])[:500]}...")
        
        # Process all samples using multiprocessing
        logger.info("Processing samples with multiprocessing...")
        
        # Use multiprocessing for faster processing
        num_workers = min(cpu_count(), 8)  # Limit workers to avoid memory issues
        batch_size = 1000
        
        logger.info(f"Using {num_workers} workers, batch size {batch_size}")
        
        all_contents = []
        process_func = partial(process_sample, min_tokens=128, max_tokens=512)
        
        for i in range(0, len(all_samples), batch_size):
            batch = all_samples[i:i + batch_size]
            
            with Pool(num_workers) as pool:
                batch_results = pool.map(process_func, batch)
            
            # Filter out empty results
            valid_results = [content for content in batch_results if content.strip()]
            all_contents.extend(valid_results)
            
            logger.info(f"Processed {i + len(batch)} samples, found {len(valid_results)} valid documents")
        
        logger.info(f"Total processed documents: {len(all_contents)}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed data
        output_file = os.path.join(output_dir, "pubmedqa_documents.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_contents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed data to: {output_file}")
        
        # Show statistics
        if all_contents:
            token_counts = [len(tokenizer.encode(content, add_special_tokens=False)) for content in all_contents]
            lengths = [len(content) for content in all_contents]
            logger.info(f"Token count statistics:")
            logger.info(f"  Mean tokens: {sum(token_counts) / len(token_counts):.1f}")
            logger.info(f"  Min tokens: {min(token_counts)}")
            logger.info(f"  Max tokens: {max(token_counts)}")
            logger.info(f"Character length statistics:")
            logger.info(f"  Mean length: {sum(lengths) / len(lengths):.1f}")
            logger.info(f"  Min length: {min(lengths)}")
            logger.info(f"  Max length: {max(lengths)}")
        
        # Show first few examples
        logger.info("First 3 processed examples:")
        for i, content in enumerate(all_contents[:3]):
            logger.info(f"Example {i+1}: {content[:200]}...")
        
        return all_contents
        
    except Exception as e:
        logger.error(f"Error processing PubMedQA dataset: {e}")
        raise e

def sample_documents(input_file: str, output_dir: str, target_count: int = 5000000, seed: int = 42) -> List[str]:
    """
    Sample a fixed number of documents from existing processed data.
    
    Args:
        input_file: Path to the existing pubmedqa_documents.json file
        output_dir: Directory to save sampled data
        target_count: Target number of documents to sample (default 5M)
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled document contents
    """
    logger.info(f"Sampling {target_count:,} documents from {input_file}")
    
    try:
        # Load existing processed documents
        logger.info("Loading existing processed documents...")
        with open(input_file, 'r', encoding='utf-8') as f:
            all_documents = json.load(f)
        
        total_docs = len(all_documents)
        logger.info(f"Found {total_docs:,} total documents")
        
        if total_docs <= target_count:
            logger.warning(f"Total documents ({total_docs:,}) <= target ({target_count:,}), using all documents")
            sampled_documents = all_documents
        else:
            # Set random seed for reproducibility
            random.seed(seed)
            
            # Randomly sample documents
            logger.info(f"Randomly sampling {target_count:,} documents...")
            sampled_documents = random.sample(all_documents, target_count)
            
            logger.info(f"Sampling completed: {len(sampled_documents):,} documents selected")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sampled data
        sampled_output_file = os.path.join(output_dir, "pubmedqa_documents_5M.json")
        logger.info(f"Saving sampled documents to: {sampled_output_file}")
        
        with open(sampled_output_file, "w", encoding="utf-8") as f:
            json.dump(sampled_documents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Sampled data saved to: {sampled_output_file}")
        
        # Show statistics for sampled data
        if sampled_documents:
            token_counts = [len(tokenizer.encode(content, add_special_tokens=False)) 
                          for content in sampled_documents[:1000]]  # Sample 1000 for stats to avoid memory issues
            lengths = [len(content) for content in sampled_documents[:1000]]
            
            logger.info(f"Sample statistics (from first 1000 documents):")
            logger.info(f"  Mean tokens: {sum(token_counts) / len(token_counts):.1f}")
            logger.info(f"  Min tokens: {min(token_counts)}")
            logger.info(f"  Max tokens: {max(token_counts)}")
            logger.info(f"Character length statistics:")
            logger.info(f"  Mean length: {sum(lengths) / len(lengths):.1f}")
            logger.info(f"  Min length: {min(lengths)}")
            logger.info(f"  Max length: {max(lengths)}")
        
        # Show first few examples
        logger.info("First 3 sampled examples:")
        for i, content in enumerate(sampled_documents[:3]):
            logger.info(f"Example {i+1}: {content[:200]}...")
        
        return sampled_documents
        
    except Exception as e:
        logger.error(f"Error sampling documents: {e}")
        raise e

def main():
    """Main preprocessing function with sampling support."""
    logger.info("=== PubMedQA Preprocessing for CQG-MBQA ===")
    
    try:
        # Define output directory and file paths
        output_dir = "/home/zhenghong/CQG-MBQA/data"
        processed_file = os.path.join(output_dir, "pubmedqa_documents.json")
        
        # Check if processed file already exists
        if os.path.exists(processed_file):
            logger.info(f"✅ Found existing processed file: {processed_file}")
            logger.info("🎯 Executing sampling to 5 million documents...")
            
            # Sample existing processed data to 5M documents
            sampled_documents = sample_documents(
                input_file=processed_file,
                output_dir=output_dir,
                target_count=5000000,
                seed=42
            )
            
            logger.info(f"✅ Sampling completed successfully!")
            logger.info(f"📊 Sampled {len(sampled_documents):,} documents")
            logger.info(f"📁 Sampled data saved to: {output_dir}/pubmedqa_documents_5M.json")
            
        else:
            logger.info("❌ Processed file not found, starting from scratch...")
            logger.info("🔄 Step 1: Processing raw data with token filtering [128, 512]...")
            
            # Find input directory with raw data
            input_dir = os.environ.get("PUBMED_DATA_DIR", "/home/zhenghong/CQG-MBQA/data/pubmed/chunk")
            
            # Try multiple possible paths if default doesn't work
            possible_paths = [
                input_dir,
                "/home/zhenghong/CQG-MBQA/data/pubmed/chunk",
                "/home/zhenghong/CQG-MBQA/data/pubmed",
                "./data/pubmed/chunk",
                "./data/pubmed",
                "../data/pubmed/chunk",
                "../data/pubmed"
            ]
            
            found_path = None
            for path in possible_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    # Look for both JSONL and JSON files
                    jsonl_files = glob.glob(os.path.join(expanded_path, "*.jsonl"))
                    json_files = glob.glob(os.path.join(expanded_path, "*.json"))
                    all_files = jsonl_files + json_files
                    
                    if all_files:
                        found_path = expanded_path
                        logger.info(f"✅ Found raw PubMedQA data at: {found_path}")
                        logger.info(f"📁 Contains {len(jsonl_files)} JSONL files and {len(json_files)} JSON files")
                        break
            
            if not found_path:
                logger.error("❌ Could not find PubMedQA raw data directory")
                logger.info("💡 Please ensure raw data is available at:")
                logger.info("   /home/zhenghong/CQG-MBQA/data/pubmed/chunk")
                logger.info("Or set custom path: export PUBMED_DATA_DIR='/your/path'")
                return
            
            # Step 1: Process and filter raw data
            processed_contents = preprocess_pubmedqa(input_dir=found_path, output_dir=output_dir)
            logger.info(f"✅ Step 1 completed: Processed {len(processed_contents):,} documents")
            logger.info(f"📁 Full processed data saved to: {processed_file}")
            
            # Step 2: Sample processed data to 5M documents
            logger.info("🎯 Step 2: Sampling processed data to 5 million documents...")
            
            sampled_documents = sample_documents(
                input_file=processed_file,
                output_dir=output_dir,
                target_count=5000000,
                seed=42
            )
            
            logger.info(f"✅ Step 2 completed: Sampled {len(sampled_documents):,} documents")
            logger.info(f"📁 Sampled data saved to: {output_dir}/pubmedqa_documents_5M.json")
            
            # Summary
            logger.info("🎉 Complete workflow finished successfully!")
            logger.info(f"📊 Generated 2 files:")
            logger.info(f"   - Full dataset: {processed_file} ({len(processed_contents):,} documents)")
            logger.info(f"   - Sampled dataset: {output_dir}/pubmedqa_documents_5M.json ({len(sampled_documents):,} documents)")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise e

if __name__ == "__main__":
    main()