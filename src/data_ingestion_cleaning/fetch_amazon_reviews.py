import os
from pathlib import Path

def main():
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Amazon Reviews dataset from Kaggle...")
    
    cmd = (
        "kaggle datasets download -d bittlingmayer/amazonreviews "
        "-p data/raw --unzip"
    )
    os.system(cmd)

    print("\nDownload complete!")
    print("Files saved to: data/raw")

if __name__ == "__main__":
    main()
