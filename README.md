# Book Genre Classification Using NLP

This project is part of a Master's thesis titled **"Klasifikacija žanrova knjiga primjenom tehnika obrade prirodnog jezika"** (Classification of Book Genres Using Natural Language Processing Techniques). The goal is to classify book genres based on their descriptions using various NLP techniques and machine learning algorithms.

## Author: Ivana Herceg  

**Keywords**: Natural Language Processing (NLP), TF-IDF, Word2Vec, Doc2Vec, multi-label classification, logistic regression, Random Forest, Voting Classifier


## Overview
This project implements multi-label classification of book genres based on textual descriptions using Natural Language Processing (NLP) techniques. The dataset is preprocessed, and different vectorization techniques are applied, such as TF-IDF, CountVectorizer, Word2Vec, and Doc2Vec. The model performance is evaluated using various classifiers and ensemble methods.

## Tools and Libraries
- **Pandas**: Data manipulation
- **Scikit-learn**: For machine learning and model evaluation
- **Gensim**: For Word2Vec and Doc2Vec models
- **Matplotlib**: For data visualization
- **Iterstrat**: For multi-label stratified cross-validation

## Dataset
The dataset used is a CSV file (`books.csv`) containing book descriptions and genres.

## Steps
1. **Preprocessing**: Text data is cleaned, including removing numbers and punctuation.
2. **Vectorization**: TF-IDF, CountVectorizer, Word2Vec, and Doc2Vec are used to convert text to numerical representations.
3. **Modeling**: Various models such as Logistic Regression, Random Forest, and ensemble classifiers are applied.
4. **Evaluation**: The models are evaluated using accuracy, Hamming loss, and classification reports.

## Key Visualizations
- **Genre Distribution**: Bar plot of actual genre counts.
- **Model Performance**: Classification reports for each model.

## Results
The models are evaluated with different vectorization techniques:
- **TF-IDF**: Achieves good accuracy with efficient vectorization.
- **Word2Vec & Doc2Vec**: Provide improved performance over TF-IDF.
- **Ensemble Model**: Combines multiple classifiers for enhanced results.

## Usage
The project provides functions to:
- Train models with different vectorization methods.
- Predict book genres based on descriptions.
- Evaluate model performance with various metrics.

## Conclusion
This project demonstrates the application of multiple NLP techniques to classify book genres and provides a comprehensive evaluation of different models.
