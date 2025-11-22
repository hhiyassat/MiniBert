#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process Wikipedia Arabic corpus from multiple folders
Extracts text from JSON format and applies cleaning
Outputs with and without diacritics
"""

import os
import json
from pathlib import Path
from arabic_utils import process_text

def process_wikipedia_corpus():
    """
    Process all Wikipedia files from AA-BS folders
    Each folder contains 100 wiki files (wiki_00 to wiki_99)
    """
    
    # Base directory containing AA, AB, AC... folders
    base_dir = Path("/home/ubuntu/PycharmProjects/Qwen_aric/bert-cv/raw_data/not-dictric")
    
    # Output paths
    output_base = Path("/home/ubuntu/PycharmProjects/Qwen_aric/bert-cv")
    output_with_diacritics = output_base / "wikipedia_with_diacritics.txt"
    output_without_diacritics = output_base / "wikipedia_without_diacritics.txt"
    
    # Also save to full_corpus directory
    corpus_dir = Path("/home/ubuntu/PycharmProjects/Qwen_aric/bert-cv/raw_data/not-dictric/full_corpus")
    corpus_dir.mkdir(exist_ok=True)
    corpus_with_diacritics = corpus_dir / "wikipedia_with_diacritics.txt"
    corpus_without_diacritics = corpus_dir / "wikipedia_without_diacritics.txt"
    
    print("Starting Wikipedia corpus processing...")
    print(f"Base directory: {base_dir}")
    
    # Get all folders (AA, AB, AC, ..., BS)
    folders = sorted([f for f in base_dir.iterdir() if f.is_dir() and len(f.name) == 2])
    print(f"Found {len(folders)} folders to process")
    
    total_articles = 0
    total_lines = 0
    
    # Open all output files
    with open(output_with_diacritics, 'w', encoding='utf-8') as f_with, \
         open(output_without_diacritics, 'w', encoding='utf-8') as f_without, \
         open(corpus_with_diacritics, 'w', encoding='utf-8') as f_corpus_with, \
         open(corpus_without_diacritics, 'w', encoding='utf-8') as f_corpus_without:
        
        # Process each folder
        for folder_idx, folder in enumerate(folders, 1):
            print(f"\nProcessing folder {folder_idx}/{len(folders)}: {folder.name}")
            
            # Get all wiki files in this folder (wiki_00 to wiki_99)
            wiki_files = sorted([f for f in folder.iterdir() if f.name.startswith('wiki_')])
            
            folder_articles = 0
            
            for file_idx, wiki_file in enumerate(wiki_files, 1):
                if file_idx % 20 == 0:
                    print(f"  Processing file {file_idx}/{len(wiki_files)} in {folder.name}...")
                
                try:
                    with open(wiki_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                # Parse JSON
                                article = json.loads(line)
                                text = article.get('text', '')
                                
                                if not text:
                                    continue
                                
                                # Process with diacritics (remove non-Arabic only)
                                text_with_diacritics = process_text(text, remove_diacritics=False)
                                
                                # Process without diacritics
                                text_without_diacritics = process_text(text, remove_diacritics=True)
                                
                                # Write to all output files
                                if text_with_diacritics.strip():
                                    f_with.write(text_with_diacritics + '\n')
                                    f_corpus_with.write(text_with_diacritics + '\n')
                                    total_lines += 1
                                
                                if text_without_diacritics.strip():
                                    f_without.write(text_without_diacritics + '\n')
                                    f_corpus_without.write(text_without_diacritics + '\n')
                                
                                folder_articles += 1
                                
                            except json.JSONDecodeError:
                                continue
                                
                except Exception as e:
                    print(f"  Error processing {wiki_file.name}: {e}")
                    continue
            
            total_articles += folder_articles
            print(f"  Processed {folder_articles} articles from {folder.name}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total folders processed: {len(folders)}")
    print(f"Total articles extracted: {total_articles}")
    print(f"Total lines written: {total_lines}")
    print(f"\nOutput files:")
    print(f"  1. {output_with_diacritics}")
    print(f"  2. {output_without_diacritics}")
    print(f"  3. {corpus_with_diacritics}")
    print(f"  4. {corpus_without_diacritics}")
    print(f"{'='*60}")

if __name__ == "__main__":
    process_wikipedia_corpus()
