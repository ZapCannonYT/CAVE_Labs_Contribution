import os
import zipfile
from pathlib import Path

# Config
PROJECT_DIR = Path(__file__).resolve().parent
ZIP_FILE_NAME = PROJECT_DIR / "healthbot_v3.2.zip"

# Exclude directories (exact name match)
EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "env",
    ".git",
    ".cache",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "logs",
}

# Exclude files (extensions or specific names)
EXCLUDE_EXTS = {
    ".gguf",
    ".incomplete",
    ".lock",
    ".zip",  # Don't zip the zip file itself!
}

EXCLUDE_FILES = {
    ".env",
}

def should_exclude(path: Path) -> bool:
    # Check directory exclusion
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    
    # Check file name and extension exclusion
    if path.is_file():
        if path.name in EXCLUDE_FILES or path.suffix.lower() in EXCLUDE_EXTS:
            return True
            
    return False

def zip_project():
    print(f"Creating zip file: {ZIP_FILE_NAME.name}...")
    count = 0
    with zipfile.ZipFile(ZIP_FILE_NAME, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(PROJECT_DIR):
            # Modify dirs in-place to prevent os.walk from traversing excluded dirs
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                file_path = Path(root) / file
                if should_exclude(file_path):
                    continue
                
                # Calculate relative path for the zip archive
                rel_path = file_path.relative_to(PROJECT_DIR)
                zip_file.write(file_path, rel_path)
                count += 1
                
    size_mb = ZIP_FILE_NAME.stat().st_size / (1024 * 1024)
    print(f"Success! Zipped {count} files.")
    print(f"Shared ZIP size: {size_mb:.2f} MB")
    print(f"Saved to: {ZIP_FILE_NAME}")

if __name__ == "__main__":
    zip_project()
