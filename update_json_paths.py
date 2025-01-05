# 此腳本用於更新 AutoEncoderData 目錄下所有 JSON 文件中的文件路徑，
# 將舊的目錄結構 (NormalSubject/PatientSubject) 更新為新的結構 (OriginalData)

import json
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'path_update_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class JsonPathUpdater:
    def __init__(self, base_dir="WavData/AutoEncoderData"):
        self.base_dir = Path(base_dir)
        self.backup_dir = self.base_dir.parent / f"backup_json_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.processed_files = 0
        self.updated_files = 0
        self.error_files = 0

    def create_backup(self, file_path: Path) -> bool:
        try:
            backup_path = self.backup_dir / file_path.relative_to(self.base_dir)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_path)
            return True
        except Exception as e:
            logging.error(f"Failed to create backup for {file_path}: {str(e)}")
            return False

    def update_path(self, original_path: str) -> str:
        updated_path = re.sub(r'/(?:NormalSubject|PatientSubject)/', '/OriginalData/', original_path)
        return updated_path

    def process_json_file(self, file_path: Path) -> bool:
        try:
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if the file needs updating
            if 'original_file' not in data:
                logging.info(f"Skipping {file_path}: 'original_file' not found")
                return False

            original_path = data['original_file']
            if 'NormalSubject' not in original_path and 'PatientSubject' not in original_path:
                logging.info(f"Skipping {file_path}: no path update needed")
                return False

            # Create backup before modification
            if not self.create_backup(file_path):
                return False

            # Update the path
            data['original_file'] = self.update_path(original_path)

            # Write back the updated JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logging.info(f"Updated {file_path}")
            logging.info(f"Old path: {original_path}")
            logging.info(f"New path: {data['original_file']}")
            return True

        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error in {file_path}: {str(e)}")
            self.error_files += 1
            return False
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            self.error_files += 1
            return False

    def process_directory(self):
        try:
            # Create backup directory
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created backup directory: {self.backup_dir}")

            # Process all JSON files
            for file_path in self.base_dir.rglob("*_tokens_info.json"):
                self.processed_files += 1
                if self.process_json_file(file_path):
                    self.updated_files += 1

            # Log summary
            logging.info(f"\nProcessing Summary:")
            logging.info(f"Total files processed: {self.processed_files}")
            logging.info(f"Files updated: {self.updated_files}")
            logging.info(f"Files with errors: {self.error_files}")

        except Exception as e:
            logging.error(f"Error during directory processing: {str(e)}")

def main():
    updater = JsonPathUpdater()
    updater.process_directory()

if __name__ == "__main__":
    main() 