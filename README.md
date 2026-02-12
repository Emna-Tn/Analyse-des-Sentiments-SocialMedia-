# Sentiment Analysis on Social Media Data

## Project Overview
This project aims to analyze public sentiment from social media text data using
Natural Language Processing (NLP) techniques and machine learning models.

We explore, clean, and preprocess textual data, then apply sentiment analysis
using both lexicon-based and machine learning approaches.

##  Dataset
The dataset file, named sentimentsdataset.csv, encapsulates diverse social media insights. It comprises user-generated content, sentiment labels, timestamps, platform details, trending hashtags, user engagement metrics, and geographical origins. With additional columns for extracted date and time components, this dataset is a valuable resource for sentiment analysis, trend identification, and temporal analysis on social media platforms.


##  Technologies Used
- Python
- Pandas, NumPy
- NLTK (VADER)
- Scikit-learn
- Matplotlib, Seaborn, Plotly


##  Methodology

### 1. Data Cleaning & Preprocessing
- Text normalization (lowercase, punctuation removal)
- Tokenization
- Stopwords removal
- lemmatization

### 2. Sentiment Analysis (Lexicon-based)
- VADER SentimentIntensityAnalyzer
- Compound score
- Sentiment classification: Positive / Neutral / Negative

### 3. Feature Extraction
- TF-IDF Vectorization (max_features = 5000)

### 4. Machine Learning Models
- Passive Aggressive Classifier
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Multinomial Naive Bayes

### 5. Hyperparameter Optimization
- RandomizedSearchCV
- Evaluation using accuracy, F1-score, confusion matrix

---

##  Results
- Best model: Passive Aggressive Classifier
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Visualization: Sentiment distribution, confusion matrix

---

##  Visualizations
- Sentiment distribution (pie chart)
- Sentiment vs Time / Platform / Country
- Word frequency analysis
- Confusion matrix



