import os
import json
from pathlib import Path

# --- CONFIGURATION ---
OUTPUT_FILE = "project_context_report.txt"

# Files to READ full content (Code, Configs, Documentation)
TEXT_EXTENSIONS = {'.py', '.ipynb', '.md', '.txt', '.json', '.yaml', '.yml', '.sh'}

# Folders to IGNORE completely (Noise)
IGNORE_DIRS = {'.git', '__pycache__', '.ipynb_checkpoints', 'venv', 'env', 'node_modules', '.idea', '.vscode'}

# Folders to SUMMARIZE (List files but DO NOT read content)
# Add your heavy data folders here to prevent the text file from becoming 500MB
DATA_DIRS = {'data', 'raw', 'processed', 'artifacts', 'PhaseA_artifacts', 'PhaseB_artifacts', 'PhaseC_artifacts', 'figures', 'models'}

# Max lines to read per file to prevent overflow
MAX_LINES = 2000 

def get_file_size(path):
    try:
        return f"{path.stat().st_size / 1024:.1f} KB"
    except Exception:
        return "Unknown"

def read_file_content(path):
    """Reads file content. Handles .ipynb specifically to make it readable."""
    try:
        # Special handling for Notebooks: Extract code cells only to save space
        if path.suffix == '.ipynb':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content = []
                for cell in data.get('cells', []):
                    if cell.get('cell_type') == 'code':
                        source = ''.join(cell.get('source', []))
                        content.append(f"# [CELL] \n{source}\n")
                return "\n".join(content)
        
        # Standard text files
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if len(lines) > MAX_LINES:
                return "".join(lines[:MAX_LINES]) + f"\n... [TRUNCATED AFTER {MAX_LINES} LINES] ..."
            return "".join(lines)
            
    except Exception as e:
        return f"[ERROR READING FILE: {e}]"

def main():
    root_dir = Path.cwd()
    report = []
    
    report.append(f"=== PROJECT CONTEXT REPORT ===")
    report.append(f"Root: {root_dir}")
    report.append("="*40 + "\n")

    # 1. Walk through directories
    for path in sorted(root_dir.rglob('*')):
        # Skip directories themselves
        if path.is_dir():
            continue
            
        # Check against IGNORE lists
        parts = path.parts
        if any(ignored in parts for ignored in IGNORE_DIRS):
            continue

        rel_path = path.relative_to(root_dir)
        is_data_dir = any(data_dir in parts for data_dir in DATA_DIRS)
        
        # Header for the file
        report.append(f"FILE: {rel_path}")
        report.append(f"SIZE: {get_file_size(path)}")
        
        # DECISION: Read content or just list it?
        if is_data_dir:
            report.append("[CONTENT SKIPPED: DATA/ARTIFACT DIRECTORY]")
        elif path.suffix in TEXT_EXTENSIONS:
            report.append("-" * 20)
            report.append(read_file_content(path))
            report.append("-" * 20)
        else:
            report.append("[CONTENT SKIPPED: BINARY OR NON-CODE FILE]")
            
        report.append("\n" + "="*40 + "\n")

    # Write Report
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    print(f"✅ Context generated: {OUTPUT_FILE}")
    print(f"   Please upload this file to the chat.")

if __name__ == "__main__":
    main()