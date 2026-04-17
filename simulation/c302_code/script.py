import os

def list_project_files(startpath):
    # Extensions we care about for the paper
    extensions = ['.ipynb', '.csv', '.json', '.txt', '.py']
    
    for root, dirs, files in os.walk(startpath):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                print(f'{subindent}{f}')

# Run this in the directory you want to map
list_project_files('.')
