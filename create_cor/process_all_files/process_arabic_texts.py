import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SOURCE_DIR, OUTPUT_DIR, COMBINED_OUTPUT_FILE
from arabic_utils import (
    process_text,
    read_file,
    write_file,
    find_all_txt_files,
    remove_empty_lines,
    get_file_size_mb
)


def get_user_choice():
    while True:
        print("\n" + "="*60)
        print("Arabic Text Processing - Diacritics Options")
        print("="*60)
        print("Do you want to remove Arabic diacritics?")
        print("1. Yes - Remove all diacritics")
        print("2. No - Keep diacritics (only remove sukun from lam)")
        print("="*60)
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == '1':
            print("\n→ Selected: Remove all diacritics")
            return True
        elif choice == '2':
            print("\n→ Selected: Keep diacritics (remove sukun from lam only)")
            return False
        else:
            print("✗ Invalid choice. Please enter 1 or 2.")


def process_files(txt_files, remove_diacritics):
    processed_texts = []
    
    for idx, file_path in enumerate(txt_files, 1):
        print(f"Processing [{idx}/{len(txt_files)}]: {file_path.name}")
        
        content = read_file(file_path)
        if content:
            processed_content = process_text(content, remove_diacritics)
            processed_texts.append(processed_content)
    
    return processed_texts


def main():
    output_file = str(Path(OUTPUT_DIR) / COMBINED_OUTPUT_FILE)
    
    if not os.path.exists(SOURCE_DIR):
        print(f"✗ Error: Source directory not found: {SOURCE_DIR}")
        return
    
    remove_diacritics = get_user_choice()
    
    print("\nInitializing processor...")
    txt_files = find_all_txt_files(SOURCE_DIR)
    
    if not txt_files:
        print(f"No .txt files found in {SOURCE_DIR}")
        return
    
    print(f"Found {len(txt_files)} text files")
    print(f"Processing mode: {'Remove all diacritics' if remove_diacritics else 'Keep diacritics (except sukun on lam)'}")
    
    processed_texts = process_files(txt_files, remove_diacritics)
    combined_text = "\n".join(processed_texts)
    combined_text = remove_empty_lines(combined_text)
    
    write_file(output_file, combined_text)
    
    print(f"\n✓ Processing complete!")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ Total files processed: {len(txt_files)}")
    
    if os.path.exists(output_file):
        file_size = get_file_size_mb(output_file)
        print(f"✓ Output file size: {file_size:.2f} MB")


if __name__ == "__main__":
    main()
