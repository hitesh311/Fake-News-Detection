# Fake News Detection System

This project implements a fake news detection system using DistilBERT, a lightweight version of BERT. The system classifies news articles as either "Real" or "Fake" based on their content.

## Features

- **Text Classification**: Uses DistilBERT for efficient and accurate classification
- **Web Interface**: Simple and intuitive web interface for easy interaction
- **REST API**: Built with Flask for easy integration with other applications
- **Model Explanation**: Provides confidence scores and explanations for predictions

## Quick Start

Get started with the Fake News Detection system in just a few steps:

```bash
# 1. Clone the repository
git clone https://github.com/hitesh311/Fake-News-Detection.git
cd Fake-News-Detection

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the pre-trained model
python download_model.py

# 5. Start the web application
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Detailed Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hitesh311/Fake-News-Detection.git
   cd Fake-News-Detection
   ```

2. **Set Up Virtual Environment** (recommended)
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Model**
   The easiest way is to use the provided download script:
   ```bash
   python download_model.py
   ```
   
   This will:
   - Create a `model_save` directory
   - Download the pre-trained model files
   - Verify the download integrity
   - Extract the files to the correct location

   *Note: The model files are approximately 500MB, so the download may take a few minutes depending on your internet connection.*

### Alternative Installation Methods

#### Manual Download
If the automatic download fails, you can manually download the model files:

1. Go to the [Releases](https://github.com/hitesh311/Fake-News-Detection/releases) page
2. Download the latest `model_save.zip` file
3. Extract the contents into the `model_save` directory

#### Train Your Own Model
If you want to train the model from scratch (requires GPU):
```bash
python fake_news_detector.py
```
*Note: Training requires significant computational resources and may take several hours on a GPU.*

## Model Verification

To ensure the integrity of the downloaded model files, the download script verifies the MD5 checksum of the downloaded file. If you want to manually verify the model files, you can use the following command:

```bash
# On Linux/macOS
md5sum model_save/model_files.zip

# On Windows (PowerShell)
Get-FileHash -Algorithm MD5 model_save\model_files.zip
```

The expected MD5 checksum is: `b0318b147034d8af736743f5c7ff3186`

This checksum should match the one in the `download_model.py` script and the [release notes](https://github.com/hitesh311/Fake-News-Detection/releases/latest).

## Troubleshooting

### Download Issues
If the download fails or is interrupted:
1. Check your internet connection
2. Make sure you have enough disk space (at least 1GB free)
3. Try running the script again with the `--force` flag:
   ```bash
   python download_model.py --force
   ```

### Model Loading Issues
If the application fails to load the model:
1. Make sure the `model_save` directory exists and contains the required files:
   - `config.json`
   - `pytorch_model.bin`
   - `special_tokens_map.json`
   - `tokenizer_config.json`
   - `vocab.txt`
2. Verify the files are not corrupted by checking their checksums
3. Try deleting the `model_save` directory and running the download script again

## Model Files

Due to GitHub's file size limitations, the pre-trained model files are hosted separately in the [GitHub Releases](https://github.com/hitesh311/Fake-News-Detection/releases) section. The model includes:

- **DistilBERT Base Model**: Fine-tuned for fake news detection
- **Tokenizer**: Custom tokenizer for text processing
- **Configuration**: Model architecture and training parameters

### File Structure
```
model_save/
├── config.json           # Model configuration
├── pytorch_model.bin     # Model weights
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.txt            # Tokenizer vocabulary
```

## Usage

### Web Interface

The easiest way to use the Fake News Detection system is through the web interface:

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Paste a news article into the text area and click "Check Authenticity"

4. View the results, including:
   - Prediction (Real/Fake)
   - Confidence score
   - Highlighted important words

### API Usage

For programmatic access, you can use the REST API:

#### Single Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Your news article text here..."}'
```

#### Batch Prediction
```bash
curl -X POST http://localhost:5000/api/predict_batch \
     -H "Content-Type: application/json" \
     -d '{"texts": ["First article...", "Second article..."]}'
```

#### API Response Format
```json
{
  "prediction": "Real",
  "confidence": 0.98,
  "explanation": "The text appears to be from a reliable source..."
}
```

### Command Line Interface

You can also use the command line for quick predictions:

```bash
python predict.py --text "Your news article text here..."

# Or from a file
python predict.py --file article.txt
```

## Project Structure

- `app.py`: Flask web application and API
- `fake_news_detector.py`: Model training and evaluation script
- `templates/`: HTML templates for the web interface
- `model_save/`: Directory for saved models
- `requirements.txt`: Python dependencies
- `concepts.txt`: Detailed explanation of concepts used in the project

## Model Details

The model is based on DistilBERT, a distilled version of BERT that is 40% smaller but retains 97% of its performance. The model was fine-tuned on a dataset of real and fake news articles.

### Training Data

- **Real News**: Articles from reliable news sources
- **Fake News**: Articles labeled as fake or from unreliable sources

### Evaluation Metrics

- Accuracy: >95% on test set
- Precision: >94% for both classes
- Recall: >94% for both classes
- F1-Score: >94% for both classes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Flask](https://flask.palletsprojects.com/)
- [Bootstrap](https://getbootstrap.com/)
