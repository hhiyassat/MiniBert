import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TEST_1_TEXT_FILE_DIR, TEST_1_TEXT_FILE_NAME, OUTPUT_DIR, TEST_1_OUTPUT_FILE
from arabic_utils import (
    process_text,
    read_file,
    write_file,
    find_specific_file,
    remove_empty_lines,
    get_file_size_kb
)


def get_user_choice():
    while True:
        print("\n" + "="*60)
        print("Arabic Text Processing - Al Jazeera Test")
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


def show_comparison(original, processed):
    print(f"  Original (first 200 chars): {original[:200]}")
    print(f"  Processed (first 200 chars): {processed[:200]}")


def process_single_file(file_path, remove_diacritics):
    print(f"\nProcessing: {file_path.name}")
    
    original_content = read_file(file_path)
    if not original_content:
        return ""
    
    processed_content = process_text(original_content, remove_diacritics)
    show_comparison(original_content, processed_content)
    
    return processed_content


def main():
    output_file = str(Path(OUTPUT_DIR) / TEST_1_OUTPUT_FILE)
    
    if not os.path.exists(TEST_1_TEXT_FILE_DIR):
        print(f"✗ Error: Source directory not found: {TEST_1_TEXT_FILE_DIR}")
        return
    
    remove_diacritics = get_user_choice()
    
    print("\nInitializing processor...")
    txt_files = find_specific_file(TEST_1_TEXT_FILE_DIR, TEST_1_TEXT_FILE_NAME)
    
    if not txt_files:
        print(f"File not found: {TEST_1_TEXT_FILE_NAME}")
        return
    
    print(f"Found 1 text file: {TEST_1_TEXT_FILE_NAME}")
    print(f"Processing mode: {'Remove all diacritics' if remove_diacritics else 'Keep diacritics (except sukun on lam)'}")
    
    processed_content = process_single_file(txt_files[0], remove_diacritics)
    
    processed_content = remove_empty_lines(processed_content)
    
    write_file(output_file, processed_content)
    
    print(f"\n✓ Processing complete!")
    print(f"✓ Output saved to: {output_file}")
    
    if os.path.exists(output_file):
        file_size = get_file_size_kb(output_file)
        print(f"✓ Output file size: {file_size:.2f} KB")


if __name__ == "__main__":
    main()
