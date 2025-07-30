# Sentiment Analysis with IMDB Movie Reviews

A machine learning project that performs sentiment analysis on movie reviews using Natural Language Processing (NLP) techniques and Logistic Regression.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a sentiment analysis system that can classify movie reviews as either positive or negative. The model is trained on the IMDB movie reviews dataset containing 50,000 reviews and achieves approximately 79% accuracy on the test set.

## ✨ Features

- **Text Preprocessing**: Comprehensive text cleaning including HTML tag removal, tokenization, stop word removal, and lemmatization
- **TF-IDF Vectorization**: Converts text data into numerical features for machine learning
- **Logistic Regression Model**: Binary classification for sentiment prediction
- **Performance Evaluation**: Detailed metrics including accuracy, precision, recall, and F1-score
- **Interactive Prediction**: Function to predict sentiment of new text inputs

## 📊 Dataset

The project uses the IMDB Movie Reviews dataset:
- **Size**: 50,000 movie reviews
- **Classes**: Positive (25,000) and Negative (25,000) reviews
- **Format**: CSV file with 'review' and 'sentiment' columns
- **Balance**: Perfectly balanced dataset

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd sentiment-analysis-project
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install required packages**:
```bash
pip install pandas scikit-learn nltk spacy matplotlib seaborn
```

4. **Download NLTK data**:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```

5. **Download spaCy model**:
```bash
python -m spacy download en_core_web_sm
```

## 💻 Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**:
```bash
jupyter notebook main.ipynb
```

2. **Execute cells sequentially** to:
   - Load and explore the dataset
   - Preprocess the text data
   - Train the model
   - Evaluate performance
   - Make predictions on new text

### Using the Prediction Function

```python
# Example usage
review_1 = "This movie was absolutely fantastic! The acting was superb and the plot was gripping."
review_2 = "I was so bored throughout the entire film. It was a complete waste of time."

print(f"Review 1 Sentiment: {predict_sentiment(review_1)}")
print(f"Review 2 Sentiment: {predict_sentiment(review_2)}")
```

## 📈 Model Performance

### Current Results
- **Accuracy**: 79.00%
- **Precision**: 
  - Negative: 0.77
  - Positive: 0.82
- **Recall**:
  - Negative: 0.84
  - Positive: 0.73
- **F1-Score**:
  - Negative: 0.80
  - Positive: 0.77

### Confusion Matrix
```
                Predicted
              Neg    Pos
Actual  Neg   43     8
        Pos   13    36
```

## 📁 Project Structure

```
sentiment-analysis-project/
│
├── main.ipynb              # Main Jupyter notebook
├── imdb_DataSet.csv       # IMDB dataset (not included - download separately)
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## 🛠 Technologies Used

- **Python 3.x**: Programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning library
- **NLTK**: Natural language processing
- **spaCy**: Advanced NLP processing
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## 🔄 Text Preprocessing Pipeline

1. **HTML Tag Removal**: Clean HTML tags from reviews
2. **Lowercasing**: Convert all text to lowercase
3. **Punctuation Removal**: Remove punctuation and numbers
4. **Tokenization**: Split text into individual words
5. **Stop Word Removal**: Remove common English stop words
6. **Lemmatization**: Reduce words to their base form

## 🎯 Future Improvements

- [ ] Experiment with other algorithms (Random Forest, SVM, Neural Networks)
- [ ] Implement cross-validation for better model evaluation
- [ ] Add support for multi-class sentiment analysis
- [ ] Create a web interface for real-time predictions
- [ ] Optimize hyperparameters using GridSearch
- [ ] Add word cloud visualizations
- [ ] Implement deep learning models (LSTM, BERT)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📝 Notes

- The model is trained on a subset of 500 reviews for demonstration purposes
- For full dataset training, remove the sampling line: `df = df.sample(500).copy()`
- Ensure the IMDB dataset file is in the same directory as the notebook
- The preprocessing may take several minutes for the full dataset

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Happy Analyzing! 🎬📊**