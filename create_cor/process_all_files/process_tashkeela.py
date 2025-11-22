import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arabic_utils import (
    process_text,
    read_file,
    write_file,
    find_all_txt_files,
    remove_empty_lines,
    get_file_size_mb
)


def process_files(txt_files, remove_diacritics):
    """Process all txt files and return combined text"""
    processed_texts = []
    
    for idx, file_path in enumerate(txt_files, 1):
        print(f"Processing [{idx}/{len(txt_files)}]: {file_path.name}")
        
        content = read_file(file_path)
        if content:
            processed_content = process_text(content, remove_diacritics)
            processed_texts.append(processed_content)
    
    return processed_texts


def main():
    # Source directory with Tashkeela texts
    source_dir = "/home/ubuntu/PycharmProjects/Qwen_aric/bert-cv/archive/Tashkeela-arabic-diacritized-text-utf8-0.3/texts.txt"
    
    # Output directory - same as Wikipedia (in process_all_files folder)
    output_dir = Path(__file__).parent  # This will be create_cor/process_all_files/
    
    # Also save to full_corpus directory
    corpus_dir = "/home/ubuntu/PycharmProjects/Qwen_aric/bert-cv/raw_data/not-dictric/full_corpus"
    
    print("="*70)
    print("PROCESSING TASHKEELA ARABIC DATASET")
    print("="*70)
    
    if not os.path.exists(source_dir):
        print(f"✗ Error: Source directory not found: {source_dir}")
        return
    
    print(f"\nSource: {source_dir}")
    print("\nFinding all text files...")
    txt_files = find_all_txt_files(source_dir)
    
    if not txt_files:
        print(f"No .txt files found in {source_dir}")
        return
    
    print(f"✓ Found {len(txt_files)} text files")
    
    # Process with diacritics
    print("\n" + "="*70)
    print("PROCESSING: WITH DIACRITICS")
    print("="*70)
    processed_with_diacritics = process_files(txt_files, remove_diacritics=False)
    combined_with_diacritics = "\n".join(processed_with_diacritics)
    combined_with_diacritics = remove_empty_lines(combined_with_diacritics)
    
    # Save to process_all_files directory
    output_with_diacritics_1 = output_dir / "tashkeela_with_diacritics.txt"
    write_file(str(output_with_diacritics_1), combined_with_diacritics)
    print(f"\n✓ Saved to: {output_with_diacritics_1}")
    print(f"✓ Size: {get_file_size_mb(str(output_with_diacritics_1)):.2f} MB")
    
    # Save to corpus directory
    os.makedirs(corpus_dir, exist_ok=True)
    output_with_diacritics_2 = os.path.join(corpus_dir, "tashkeela_with_diacritics.txt")
    write_file(output_with_diacritics_2, combined_with_diacritics)
    print(f"✓ Saved to: {output_with_diacritics_2}")
    
    # Process without diacritics
    print("\n" + "="*70)
    print("PROCESSING: WITHOUT DIACRITICS")
    print("="*70)
    processed_without_diacritics = process_files(txt_files, remove_diacritics=True)
    combined_without_diacritics = "\n".join(processed_without_diacritics)
    combined_without_diacritics = remove_empty_lines(combined_without_diacritics)
    
    # Save to process_all_files directory
    output_without_diacritics_1 = output_dir / "tashkeela_without_diacritics.txt"
    write_file(str(output_without_diacritics_1), combined_without_diacritics)
    print(f"\n✓ Saved to: {output_without_diacritics_1}")
    print(f"✓ Size: {get_file_size_mb(str(output_without_diacritics_1)):.2f} MB")
    
    # Save to corpus directory
    output_without_diacritics_2 = os.path.join(corpus_dir, "tashkeela_without_diacritics.txt")
    write_file(output_without_diacritics_2, combined_without_diacritics)
    print(f"✓ Saved to: {output_without_diacritics_2}")
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETED!")
    print("="*70)
    print(f"Total files processed: {len(txt_files)}")
    print(f"\nOutput files created:")
    print(f"1. {output_with_diacritics_1}")
    print(f"2. {output_with_diacritics_2}")
    print(f"3. {output_without_diacritics_1}")
    print(f"4. {output_without_diacritics_2}")


if __name__ == "__main__":
    main()
