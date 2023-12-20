import os
import re
import random
import string
import numpy as np
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(rf"[{string.punctuation}0-9]", " ", text)
    tokens = text.split()
    return " ".join(tokens)

# 1. Data Extraction and Sampling
zip_file_path = '/Users/loxlikooy/Downloads/books.zip'
extract_folder_path = '/Users/loxlikooy/Downloads/books/books'  # Adjusted path
num_samples_per_book = 30
sample_length = 300
samples = []
labels = []

# Extracting the zip file
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder_path)

# Checking the extracted content and reading files
book_files = os.listdir(extract_folder_path)

for file_name in book_files:
    lang_code = re.match(r'([a-z]+)-', file_name)
    if not lang_code:
        continue
    lang_code = lang_code.group(1)

    with open(os.path.join(extract_folder_path, file_name), 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    content = content[200:-200]

    for _ in range(num_samples_per_book):
        start_idx = random.randint(0, len(content) - sample_length - 1)
        samples.append(content[start_idx:start_idx + sample_length])
        labels.append(lang_code)

# 2. Data Preprocessing
cleaned_samples = [preprocess_text(sample) for sample in samples]

# 3. Feature Extraction
vectorizer_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), max_features=100)
X_tfidf = vectorizer_tfidf.fit_transform(cleaned_samples).toarray()

# Dimensionality Reduction
svd = TruncatedSVD(n_components=30)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X_lsa = lsa.fit_transform(X_tfidf)

# Label Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# 4. Model Training and Evaluation
# 4. Model Training and Evaluation



# Model Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_lsa, y, test_size=0.2, random_state=42)
knn_optimized = KNeighborsClassifier(n_neighbors=11)
knn_optimized.fit(X_train, y_train)

# Evaluation
y_pred = knn_optimized.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Displaying evaluation metrics
print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Testing Phrases
phrases = [
    ("Hola, ¿cómo estás hoy? Espero que estés disfrutando aprender sobre el aprendizaje automático!", "esp"),
    ("Bonjour, comment allez-vous aujourd'hui? J'espère que vous appréciez d'apprendre sur l'apprentissage automatique!", "fra"),
    ("Hallo, wie geht es Ihnen heute? Ich hoffe, Sie haben Spaß am Lernen über maschinelles Lernen!", "deu"),
    ("Ciao, come stai oggi? Spero che ti stia divertendo ad imparare sull'apprendimento automatico!", "ita"),
    ("Hei, hvordan har du det i dag? Jeg håper du liker å lære om maskinlæring!", "nor"),
    ("Oi, como você está hoje? Espero que você esteja gostando de aprender sobre aprendizado de máquina!", "prt"),
    ("Hej, hur mår du idag? Jag hoppas att du tycker om att lära dig om maskininlärning!", "swe"),
    ("Hallo, hoe gaat het met je vandaag? Ik hoop dat je het leuk vindt om over machine learning te leren!", "dut"),
    ("Hei, miten voit tänään? Toivottavasti nautit koneoppimisen opiskelusta!", "fin"),
    ("Cześć, jak się masz dzisiaj? Mam nadzieję, że lubisz uczyć się o uczeniu maszynowym!", "pol"),
    ("Sziasztok, hogy vagytok ma? Remélem, élvezitek a gépi tanulás tanulását!", "hun"),
    ("Ahoj, jak se máš dnes? Doufám, že tě baví učit se o strojovém učení!", "czh")
]

results_optimized = []

for phrase, true_lang in phrases:
    preprocessed_phrase = preprocess_text(phrase)
    phrase_features_tfidf = vectorizer_tfidf.transform([preprocessed_phrase]).toarray()
    phrase_features_lsa = lsa.transform(phrase_features_tfidf)
    predicted_label = knn_optimized.predict(phrase_features_lsa)[0]
    predicted_language = label_encoder.inverse_transform([predicted_label])[0]
    is_correct = predicted_language == true_lang
    results_optimized.append((phrase, true_lang, predicted_language, is_correct))

# Displaying header for phrase tests
print("\nTesting Phrases:\n")
print(f"{'Phrase':<100}{'True Language':<15}{'Predicted Language':<20}{'Is Correct':<10}\n")
for phrase, true_lang, predicted_language, is_correct in results_optimized:
    # Displaying results
    print(f"{phrase[:97]:<100}{true_lang:<15}{predicted_language:<20}{str(is_correct):<10}\n")