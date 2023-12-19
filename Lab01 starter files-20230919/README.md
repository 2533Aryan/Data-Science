# Fake News Detection

This project implements a fake news detection system using machine learning techniques. The model, based on TF-IDF vectorization and a PassiveAggressiveClassifier, achieves precision, recall, and F1-score values above 90% for both 'FAKE' and 'REAL' classes.

## Usage

### (1) Install Dependencies:

pip install numpy pandas scikit-learn nltk seaborn matplotlib

- numpy
- pandas
- scikit-learn
- nltk
- seaborn
- matplotlib

### (2) Download NLTK Resources:

import nltk
nltk.download('punkt')

### (3) Run the Code:

Execute the fake_news_detection.py script to train the model and evaluate its performance.


## Dataset

The project uses the 'news.csv' dataset (it should be in the project folder) with 7796 rows and 4 columns: identification, title, text, and labels denoting 'REAL' or 'FAKE' news.