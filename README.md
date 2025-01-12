# Burmese Hate Speech Classification

## Objective

The **Hate Speech Classification** project is a **Streamlit-based web application** that uses deep learning to classify Burmese text as either "hate speech" or "non-hate speech." The goal of this project is to build a machine learning model that can classify text data into categories such as "hate speech" and "non-hate speech." This model aims to help in identifying harmful or offensive content in Burmese text and can be used to filter or monitor online discussions, reviews, and social media posts.

## Features

- **Hate Speech Classification**
   - Classifies Burmese text into "hate speech" or "non-hate speech."
   - Provides an easy-to-use interface for entering text manually.

## Requirements

- Python 3.8+
- simmpst
- Streamlit
- TensorFlow
- Transformers
- NumPy

Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MinThihaTun3012/hate_speech_classification.git
   cd hate_speech_classification
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:

   ```bash
   streamlit run app.py
   ```

## File Structure

```plaintext
hate_speech_classification/
├── __pycache__/           # Cached Python files
├── app.py                 # Main application entry point
├── fix_error/             # Folder for any error fixes (e.g., tokenizer file name error)
├── LICENSE                # License information
├── model_best_weights.keras # Trained model weights
├── requirements.txt       # Required Python packages
├── tokenizer.pkl          # Tokenizer file used for text preprocessing
├── utils.py               # Utility functions (text preprocessing, predictions, random text)
└── README.md              # Project documentation
```

## Usage

1. Type Burmese text in the text box (or) use the **Get Test Data** button to get sample data.
2. Click the **Make Prediction** button to get the prediction.

## Model

The model used for classifying hate speech is a deep learning-based text classifier built using TensorFlow (Keras). The model is trained on `simbolo-ai/burmese-hate-speech-small` from Hugging Face and `train_small.txt` from **Simbolo Tokenizer: multilingual-partial-syllable-tokenizer**.

- [Kaggle Notebook](https://www.kaggle.com/code/minthihatun/burmese-hate-speech-detection) to see the data preprocessing and modeling process.
- [simbolo-ai/burmese-hate-speech-small](https://huggingface.co/datasets/simbolo-ai/burmese-hate-speech-small)
- [train_small.txt](https://github.com/simbolo-ai/multilingual-partial-syllable-tokenizer/blob/main/train_small.txt)

### Preprocessing Steps

- **Tokenization**: Text input is tokenized into subword units using [Simbolo Tokenizer: multilingual-partial-syllable-tokenizer](https://github.com/simbolo-ai/multilingual-partial-syllable-tokenizer).
- **Encoding** : Encode Tokens to Numerical values

### Model Results

The model was trained on a dataset of 9,114 sentences, consisting of 4,557 hate speech and 4,557 non-hate speech sentences. The data was split as follows:

- 70% for training
- 15% for validation
- 15% for testing

During training, the model achieved the following results:

- Accuracy: 0.9961
- Loss: 0.0135
- Validation Accuracy: 0.9737
- Validation Loss: 0.1674

On the testing data, the model achieved the following performance metrics:

- Accuracy: 0.94
- Recall: 0.975
- Precision: 0.973

**Note**: Due to re-running the Kaggle notebook, there may be slight variations in the metrics.

For further details on the training and evaluation process, you can check the [Kaggle Notebook](https://www.kaggle.com/code/minthihatun/burmese-hate-speech-detection).


## Limitations

Due to time constraints, limited data availability, and occasional power shortages, several challenges were encountered during the development of this project. As a result, the model performs well at distinguishing between very formal text and hate speech, but it may not be as robust in classifying other types of text. The model's performance can vary depending on the text's style, context, and length, leading to less reliable results in cases outside of these specific scenarios.

With access to more diverse and larger datasets, as well as further model refinement, the accuracy and reliability of the classification system will improve in the future.

## Acknowledgments

- **TensorFlow**: For building and training the deep learning model.
- **Streamlit**: For creating the interactive web interface.
- **Scikit-learn**: For additional machine learning utilities.
- **Regex**: For text preprocessing to handle Burmese input validation.
- **Simbolo Tokenizer: multilingual-partial-syllable-tokenizer**: For Tokenization and Encoding.

## Other

Feel free to adjust any sections or add more details if needed. Let me know if you need further modifications!

