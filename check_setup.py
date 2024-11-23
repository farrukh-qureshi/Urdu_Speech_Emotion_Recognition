import os

def check_directory_structure():
    print("Checking project setup...")
    
    # Check main directory
    current_dir = os.getcwd()
    print(f"\nCurrent working directory: {current_dir}")
    
    # List all files in current directory
    print("\nFiles in current directory:")
    for item in os.listdir(current_dir):
        print(f"- {item}")
    
    # Check raw_data directory
    data_path = 'raw_data'
    if os.path.exists(data_path):
        print(f"\nContents of {data_path}:")
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                print(f"- {item}/")
                # List few files in emotion directory
                files = os.listdir(item_path)[:3]
                for file in files:
                    print(f"  - {file}")
    else:
        print(f"\nError: {data_path} directory not found!")
    
    # Check required files
    required_files = [
        'main.py',
        'stats_analyzer.py',
        'audio_preprocessing.py',
        'models.py',
        'train.py',
        'dataset.py'
    ]
    
    print("\nChecking required files:")
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} missing")

if __name__ == "__main__":
    check_directory_structure() 