#!/usr/bin/env python3
"""
MIMIC-IV Discharge Summary Preprocessing Script

Extracts text fields from MIMIC-IV discharge.csv.gz file and converts
to the same JSON format as pubmedqa_documents_5M.json for compatibility
with CQG-MBQA training pipeline.

Output: data/mimic.json - JSON array of discharge summary text strings (raw)
Output: data/mimic_documents.json - JSON array of processed and chunked texts (processed)
"""

import gzip
import csv
import json
import sys
import re
import os
from typing import List
from tqdm import tqdm

def extract_discharge_texts(input_file: str, output_file: str) -> None:
    """Extract text fields from MIMIC-IV discharge CSV and save as JSON array."""
    
    print(f"📋 Processing MIMIC-IV discharge data...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("-" * 80)
    
    discharge_texts = []
    processed_count = 0
    
    try:
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            
            print("Processing discharge summaries...")
            
            for row in csv_reader:
                # Extract the text field
                text_content = row.get('text', '').strip()
                
                if text_content:  # Only include non-empty texts
                    discharge_texts.append(text_content)
                    processed_count += 1
                    
                    # Progress indicator
                    if processed_count % 1000 == 0:
                        print(f"  Processed {processed_count:,} records...")
        
        print(f"\n✅ Successfully extracted {len(discharge_texts):,} discharge summaries")
        
        # Statistics
        if discharge_texts:
            text_lengths = [len(text) for text in discharge_texts]
            avg_length = sum(text_lengths) / len(text_lengths)
            min_length = min(text_lengths)
            max_length = max(text_lengths)
            
            print(f"📊 Text Statistics:")
            print(f"   Total documents: {len(discharge_texts):,}")
            print(f"   Average length: {avg_length:.0f} characters")
            print(f"   Min length: {min_length:,} characters")
            print(f"   Max length: {max_length:,} characters")
        
        # Save to JSON file (same format as pubmedqa_documents_5M.json)
        print(f"\n💾 Saving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(discharge_texts, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully saved {len(discharge_texts):,} documents to {output_file}")
        
        # Show sample
        if discharge_texts:
            print(f"\n📖 Sample discharge summary (first 500 chars):")
            print("-" * 80)
            sample_text = discharge_texts[0][:500]
            print(f"{sample_text}...")
            print("-" * 80)
    
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        sys.exit(1)

def preprocess_text(text: str) -> str:
    """Apply text normalization and de-identification."""
    
    # Step 1: Basic text normalization
    # Remove excessive newlines, tabs, carriage returns
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
    text = re.sub(r'\r+', ' ', text)  # Replace carriage returns with space
    text = re.sub(r'\t+', ' ', text)  # Replace tabs with space
    
    # Remove or replace special characters
    text = re.sub(r'={2,}', '', text)  # Remove multiple equals signs
    text = re.sub(r'-{2,}', '', text)  # Remove multiple dashes
    text = re.sub(r'_{2,}', '', text)  # Remove multiple underscores
    
    # Remove numbered lists that interfere with sentence segmentation
    text = re.sub(r'\b\d+\.', '', text)  # Remove "1." "2." etc.
    
    # Step 2: De-identification
    # Remove basic header information (common patterns)
    text = re.sub(r'Name:\s*___.*?Unit No:\s*___', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'Admission Date:\s*___.*?Discharge Date:\s*___', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'Date of Birth:\s*___.*?Sex:\s*[MF]', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'Service:\s*\w+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Attending:\s*___', '', text, flags=re.IGNORECASE)
    
    # Remove bracketed de-identified information
    text = re.sub(r'\[.*?\]', '', text)
    
    # Replace privacy placeholders with standard token
    text = re.sub(r'___+', '<PHI>', text)
    
    # Additional medical text cleaning
    text = re.sub(r'\bdr\.', 'doctor', text, flags=re.IGNORECASE)
    text = re.sub(r'\bm\.d\.', 'md', text, flags=re.IGNORECASE)
    text = re.sub(r'admission date:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'discharge date:', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    return text

def chunk_text(text: str, chunk_size: int = 318, min_chunk_size: int = 10) -> List[str]:
    """Split text into chunks of approximately chunk_size words."""
    
    words = text.split()
    chunks = []
    
    # Split into regular chunks
    num_chunks = len(words) // chunk_size
    for i in range(num_chunks):
        chunk = ' '.join(words[i * chunk_size:(i + 1) * chunk_size])
        chunks.append(chunk)
    
    # Handle remaining words
    remaining_words = len(words) % chunk_size
    if remaining_words > min_chunk_size:
        remaining_chunk = ' '.join(words[-remaining_words:])
        chunks.append(remaining_chunk)
    
    return chunks

def filter_short_chunks(chunks: List[str], min_tokens: int = 128) -> List[str]:
    """Filter out chunks that are too short to be meaningful."""
    
    filtered_chunks = []
    for chunk in chunks:
        # Rough token count estimation (words * 1.3 for subwords)
        estimated_tokens = len(chunk.split()) * 1.3
        if estimated_tokens >= min_tokens:
            filtered_chunks.append(chunk)
    
    return filtered_chunks

def is_informative_text(text: str) -> bool:
    """Filter out template-only or non-informative texts."""
    
    # Check if text is too short (main filter)
    if len(text.split()) < 50:
        return False
    
    # No requirement on medical keywords — removed the >=3 filter
    
    return True

def advanced_preprocessing(input_file: str, output_file: str, batch_size: int = 5000) -> None:
    """Apply advanced preprocessing to raw MIMIC texts with batch processing."""
    
    print(f"🔧 Advanced MIMIC-IV Text Processing")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Batch size: {batch_size:,}")
    print("-" * 80)
    
    # Load raw texts
    print("📖 Loading raw texts...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_texts = json.load(f)
    
    print(f"Loaded {len(raw_texts):,} raw documents")
    
    processed_chunks = []
    total_docs = len(raw_texts)
    processed_docs = 0
    filtered_docs = 0
    
    print("🔄 Processing texts in batches...")
    
    # Process in batches to avoid memory issues and provide better progress tracking
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        batch_texts = raw_texts[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_docs-1)//batch_size + 1} ({batch_start:,}-{batch_end:,})")
        
        batch_chunks = []
        for text in batch_texts:
            # Step 1-2: Normalize and de-identify
            cleaned_text = preprocess_text(text)
            
            # Step 4: Filter non-informative texts
            if not is_informative_text(cleaned_text):
                filtered_docs += 1
                continue
            
            # Step 3: Chunk the text
            chunks = chunk_text(cleaned_text, chunk_size=318, min_chunk_size=10)
            
            # Filter short chunks
            filtered_chunks = filter_short_chunks(chunks, min_tokens=128)
            
            batch_chunks.extend(filtered_chunks)
            processed_docs += 1
        
        processed_chunks.extend(batch_chunks)
        
        # Progress update
        progress = (batch_end / total_docs) * 100
        print(f"  Batch complete: {len(batch_chunks):,} chunks, {progress:.1f}% total progress")
        print(f"  Running totals: {processed_docs:,} docs processed, {filtered_docs:,} filtered out, {len(processed_chunks):,} total chunks")
        
        # Free memory
        del batch_chunks
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Results:")
    print(f"   Original documents: {len(raw_texts):,}")
    print(f"   Processed documents: {processed_docs:,}")
    print(f"   Filtered out: {filtered_docs:,}")
    print(f"   Final chunks: {len(processed_chunks):,}")
    
    if processed_chunks:
        # Calculate statistics on a sample to avoid memory issues
        sample_size = min(10000, len(processed_chunks))
        sample_chunks = processed_chunks[:sample_size]
        chunk_lengths = [len(chunk.split()) for chunk in sample_chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)
        
        print(f"   Average chunk length: {avg_length:.1f} words (based on {sample_size:,} sample)")
        print(f"   Min chunk length: {min_length} words")
        print(f"   Max chunk length: {max_length} words")
    
    # Save processed chunks
    print(f"\n💾 Saving processed chunks to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_chunks, f, indent=None, ensure_ascii=False, separators=(',', ':'))
    
    print(f"✅ Successfully saved {len(processed_chunks):,} processed chunks")
    
    # Show sample
    if processed_chunks:
        print(f"\n📖 Sample processed chunk:")
        print("-" * 80)
        sample_chunk = processed_chunks[0][:500]
        print(f"{sample_chunk}...")
        print("-" * 80)

def main():
    """Main preprocessing function."""
    
    # File paths
    input_file = "data/MIMICIV/discharge.csv.gz"
    raw_output_file = "data/mimic.json"
    processed_output_file = "data/mimic_documents.json"
    
    print("🏥 MIMIC-IV Discharge Summary Preprocessing")
    print("=" * 80)
    
    # Check if raw extraction already exists
    if os.path.exists(raw_output_file):
        print(f"📁 Found existing {raw_output_file}")
        print("⏭️  Skipping raw extraction, proceeding to advanced processing...")
        print()
        
        # Apply advanced preprocessing
        advanced_preprocessing(raw_output_file, processed_output_file)
    else:
        print("🔍 Raw extraction file not found, starting from beginning...")
        print()
        
        # Step 1: Extract raw texts
        extract_discharge_texts(input_file, raw_output_file)
        print()
        
        # Step 2: Apply advanced preprocessing
        advanced_preprocessing(raw_output_file, processed_output_file)
    
    print("\n" + "=" * 80)
    print("🎉 PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"✅ Raw file: {raw_output_file}")
    print(f"✅ Processed file: {processed_output_file}")
    print(f"📋 Format: JSON array of processed text chunks (compatible with CQG-MBQA)")
    print(f"🔧 Ready for CQG-MBQA training pipeline!")

if __name__ == "__main__":
    main()