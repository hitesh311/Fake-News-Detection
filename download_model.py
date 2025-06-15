import os
import requests
import zipfile
import sys

def download_file(url, filename):
    """Download a file from a URL with a progress bar."""
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s] %s" % ('=' * done, ' ' * (50-done), f"Downloading {filename}..."))
                sys.stdout.flush()
    print("\nDownload complete!")

def main():
    # Create model_save directory if it doesn't exist
    os.makedirs("model_save", exist_ok=True)
    
    print("Downloading model files...")
    
    # URL to the model files (you'll need to upload these to a file hosting service)
    # For now, this is a placeholder URL - you'll need to replace it with your actual download link
    model_url = "YOUR_MODEL_DOWNLOAD_LINK"  # Replace with your actual download link
    
    if model_url == "YOUR_MODEL_DOWNLOAD_LINK":
        print("\nERROR: Please update the download_model.py script with your model download link.")
        print("1. Upload the model_save directory to a file hosting service (e.g., Google Drive, Dropbox, etc.)")
        print("2. Update the 'model_url' variable in this script with the download link")
        print("3. Ensure the link is a direct download link")
        return
    
    # Download the model files
    zip_path = "model_save/model_files.zip"
    download_file(model_url, zip_path)
    
    # Extract the files
    print("Extracting model files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("model_save")
    
    # Clean up the zip file
    os.remove(zip_path)
    
    print("\nModel files downloaded and extracted successfully!")
    print("You can now run the application with: python app.py")

if __name__ == "__main__":
    main()
