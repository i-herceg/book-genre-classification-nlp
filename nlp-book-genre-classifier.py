import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib.pyplot as plt


# Ucitavanje podataka
file_path = 'books.csv'
books_df = pd.read_csv(file_path)

# Uklanjanje redova sa NaN vrijednostima u stupci 'description'
books_df = books_df.dropna(subset=['description'])

# Funkcija za pretprocesiranje teksta
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Uklanjanje brojeva
    text = re.sub(r'[^\w\s]', '', text)  # Uklanjanje interpunkcije
    text = text.lower()  # Transformacija u mala slova
    words = text.split()
    filtered_words = [word for word in words if word not in sklearn_stop_words]
    processed_text = ' '.join(filtered_words)
    return processed_text

# Funkcija za ciscenje i normalizaciju zanrova
def clean_genres(genres):
    cleaned_genres = []
    for genre in genres:
        genre = genre.strip()  # Uklanjanje vodecih i pratecih razmaka
        genre = genre.replace("'", "").replace("[", "").replace("]", "")  # Uklanjanje nepotrebnih znakova
        cleaned_genres.append(genre)
    return cleaned_genres

# Primjena pretprocesiranja u stupcu koji sadrži opise knjiga
books_df['processed_description'] = books_df['description'].apply(preprocess_text)

# Pretvaranje zanrova u listu i čišćenje
books_df['genres'] = books_df['genres'].apply(lambda x: clean_genres(x.split(',')))

# Vektorizacija tekstualnih opisa koristeći TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(books_df['processed_description'])

# Pretvaranje žanrova u binarni format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(books_df['genres'])

# Definiranje minimalnog broja uzoraka po žanru
min_samples = 3000

# Brojanje pojavljivanja svakog žanra
genre_counts = np.sum(y, axis=0)

# Zadrzavanje samo žanrova koji imaju dovoljan broj uzoraka
genres_to_keep = np.where(genre_counts >= min_samples)[0]

# Filtriranje podataka
y_filtered = y[:, genres_to_keep]
X_filtered = X_tfidf

# Stvaranje novog MultiLabelBinarizer za filtrirane podatke
mlb_filtered = MultiLabelBinarizer(classes=mlb.classes_[genres_to_keep])
y_filtered = mlb_filtered.fit_transform(books_df['genres'])

# Balansirana podjela podataka na trening i test set
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_indices, test_indices = next(mskf.split(X_filtered, y_filtered))

X_train = X_filtered[train_indices]
X_test = X_filtered[test_indices]
y_train = y_filtered[train_indices]
y_test = y_filtered[test_indices]

# Treniranje modela Logistic Regression
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Predikcija na test setu
y_pred = model.predict(X_test)

# Evaluacija modela
filtered_genres = mlb_filtered.classes_
print("Accuracy with TF-IDF:", accuracy_score(y_test, y_pred))
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Classification Report with TF-IDF:\n", classification_report(y_test, y_pred, target_names=filtered_genres, zero_division=0))

# Brojanje stvarnih i predvidenih žanrova u test setu
actual_genre_counts = np.sum(y_test, axis=0)
predicted_genre_counts = np.sum(y_pred, axis=0)

# Vizualizacija distribucije stvarnih žanrova
plt.figure(figsize=(15, 7))
plt.bar(filtered_genres, actual_genre_counts, alpha=0.7, label='Actual')
#plt.bar(filtered_genres, predicted_genre_counts, alpha=0.5, label='Predicted')
plt.xlabel('Genres')
plt.ylabel('Number of Samples')
plt.title('Actual vs Predicted Genre Distribution')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Funkcija za predikciju žanrova za određenu knjigu
def predict_genres(book_title):
    book = books_df[books_df['title'] == book_title]
    if book.empty:
        print(f"Knjiga sa naslovom '{book_title}' nije pronađena.")
        return
    
    actual_genres = book['genres'].values[0]
    print(f"Stvarni žanrovi za '{book_title}': {', '.join(actual_genres)}")
    
    book_description = book['processed_description'].values[0]
    book_vectorized = tfidf_vectorizer.transform([book_description])
    predicted_genres = model.predict(book_vectorized)
    predicted_genres_list = mlb_filtered.inverse_transform(predicted_genres)
    
    print(f"Predviđeni žanrovi za '{book_title}': {', '.join(predicted_genres_list[0])}")

# Primjer korištenja funkcije za predikciju žanrova
#predict_genres('Harry Potter and the Sorcerer\'s Stone')

# Vektorizacija prebrojavanjem (CountVectorizer)
count_vectorizer = CountVectorizer(max_features=5000)
X_count = count_vectorizer.fit_transform(books_df['processed_description'])

# Balansirana podjela podataka na trening i test set za CountVectorizer
train_indices, test_indices = next(mskf.split(X_count, y_filtered))

X_train_count = X_count[train_indices]
X_test_count = X_count[test_indices]
y_train_count = y_filtered[train_indices]
y_test_count = y_filtered[test_indices]

# Treniranje modela
model_count = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_count.fit(X_train_count, y_train_count)

# Predikcija na test setu
y_pred_count = model_count.predict(X_test_count)

# Evaluacija modela
print("Accuracy with CountVectorizer:", accuracy_score(y_test_count, y_pred_count))
print("Hamming Loss with CountVectorizer:", hamming_loss(y_test_count, y_pred_count))
print("Classification Report with CountVectorizer:\n", classification_report(y_test_count, y_pred_count, target_names=filtered_genres, zero_division=0))

# Korištenje Word2Vec
sentences = [desc.split() for desc in books_df['processed_description']]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.train(sentences, total_examples=word2vec_model.corpus_count, epochs=10)

def get_avg_word2vec(text, model):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    avg_vector = np.mean(word_vectors, axis=0)
    return avg_vector

X_word2vec = np.array([get_avg_word2vec(desc, word2vec_model) for desc in books_df['processed_description']])

scaler = StandardScaler()
X_word2vec_scaled = scaler.fit_transform(X_word2vec)

# Balansirana podjela podataka na trening i test set za Word2Vec
train_indices, test_indices = next(mskf.split(X_word2vec_scaled, y_filtered))

X_train_w2v = X_word2vec_scaled[train_indices]
X_test_w2v = X_word2vec_scaled[test_indices]
y_train_w2v = y_filtered[train_indices]
y_test_w2v = y_filtered[test_indices]

# Treniranje modela
model_w2v = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_w2v.fit(X_train_w2v, y_train_w2v)

# Predikcija na test setu
y_pred_w2v = model_w2v.predict(X_test_w2v)

# Evaluacija modela
print("Accuracy with Word2Vec:", accuracy_score(y_test_w2v, y_pred_w2v))
print("Hamming Loss with Word2Vec:", hamming_loss(y_test_w2v, y_pred_w2v))
print("Classification Report with Word2Vec:\n", classification_report(y_test_w2v, y_pred_w2v, target_names=filtered_genres, zero_division=0))

# Korištenje Doc2Vec za vektorizaciju
tagged_data = [TaggedDocument(words=desc.split(), tags=[str(i)]) for i, desc in enumerate(books_df['processed_description'])]
doc2vec_model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=1, workers=4, epochs=10)

X_doc2vec = np.array([doc2vec_model.infer_vector(desc.split()) for desc in books_df['processed_description']])

# Balansirana podjela podataka na trening i test set za Doc2Vec
train_indices, test_indices = next(mskf.split(X_doc2vec, y_filtered))

X_train_d2v = X_doc2vec[train_indices]
X_test_d2v = X_doc2vec[test_indices]
y_train_d2v = y_filtered[train_indices]
y_test_d2v = y_filtered[test_indices]

# Treniranje modela
model_d2v = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_d2v.fit(X_train_d2v, y_train_d2v)

# Predikcija na test setu
y_pred_d2v = model_d2v.predict(X_test_d2v)

# Evaluacija modela
print("Accuracy with Doc2Vec:", accuracy_score(y_test_d2v, y_pred_d2v))
print("Hamming Loss with Doc2Vec:", hamming_loss(y_test_d2v, y_pred_d2v))
print("Classification Report with Doc2Vec:\n", classification_report(y_test_d2v, y_pred_d2v, target_names=filtered_genres, zero_division=0))

# Grid Search za tuning hiperparametara Logistic Regression modela
parameters = {
    'estimator__C': [0.1, 1, 10, 100],
    'estimator__penalty': ['l2'],
    'estimator__solver': ['lbfgs']
}
grid_search = GridSearchCV(OneVsRestClassifier(LogisticRegression(max_iter=1000)), param_grid=parameters, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predikcija na test setu s najboljim modelom
y_pred_best = best_model.predict(X_test)
print("Accuracy with Best Model from Grid Search:", accuracy_score(y_test, y_pred_best))
print("Hamming Loss Best Model from Grid Search:", hamming_loss(y_test, y_pred_best))
print("Classification Report with Best Model from Grid Search:\n", classification_report(y_test, y_pred_best, target_names=filtered_genres, zero_division=0))

# Ensemble metoda sa OneVsRestClassifier
ensemble_model = OneVsRestClassifier(VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
], voting='soft'))

# Treniranje ensemble modela
ensemble_model.fit(X_train, y_train)

# Predikcija na test setu
y_pred_ensemble = ensemble_model.predict(X_test)
print("Accuracy with Ensemble Model:", accuracy_score(y_test, y_pred_ensemble))
print("Hamming Loss with Ensemble Model:", hamming_loss(y_test, y_pred_ensemble))
print("Classification Report with Ensemble Model:\n", classification_report(y_test, y_pred_ensemble, target_names=filtered_genres, zero_division=0))
