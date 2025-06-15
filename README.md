# Fake News Detection System

This project implements a fake news detection system using DistilBERT, a lightweight version of BERT. The system classifies news articles as either "Real" or "Fake" based on their content.

## Features

- **Text Classification**: Uses DistilBERT for efficient and accurate classification
- **Web Interface**: Simple and intuitive web interface for easy interaction
- **REST API**: Built with Flask for easy integration with other applications
- **Model Explanation**: Provides confidence scores and explanations for predictions

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hitesh311/Fake-News-Detection.git
   cd Fake-News-Detection
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model:
   - **Option 1**: Using the download script (recommended):
     ```bash
     python download_model.py
     ```
     Note: You'll need to update the `model_url` in `download_model.py` with your model download link.
   
   - **Option 2**: Manual download:
     1. Download the model files from [Google Drive](YOUR_GOOGLE_DRIVE_LINK) (or your preferred file hosting service)
     2. Extract the contents into the `model_save` directory

   - **Option 3**: Train your own model (requires GPU):
     ```bash
     python fake_news_detector.py
     ```

## Model Files

Due to GitHub's file size limitations, the pre-trained model files are not included in the repository. You have two options:

1. **Download Pre-trained Model**:
   - Run `python download_model.py` (after setting up the download link)
   - Or manually download from [Google Drive](YOUR_GOOGLE_DRIVE_LINK)

2. **Train Your Own Model**:
   - Run `python fake_news_detector.py` to train a new model
   - This requires a GPU for reasonable training time

## Usage

### Web Interface

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Paste a news article into the text area and click "Check Authenticity"

### API Usage

You can also use the API directly:

```bash
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Your news article text here..."}'
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
