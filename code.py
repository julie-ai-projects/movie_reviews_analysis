# 🎬 Анализ отзывов о фильмах
# Автор: Заводскова Юлия
# Проект для портфолио Data Science

# 1. Импорт библиотек 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import re

# 2. Пример данных 
data = {
    'review': [
        "Amazing movie! I loved the story and characters.",
        "Terrible plot, I wasted my time.",
        "It was okay, not great but not bad.",
        "Absolutely fantastic performance by the lead actor!",
        "I didn't like it, the script was weak."
    ],
    'sentiment': [1, 0, 1, 1, 0]  # 1 — положительный, 0 — отрицательный
}

df = pd.read_csv('data/data_sample.csv')
df.head()

# 3. Очистка текста
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

df['clean_review'] = df['review'].apply(clean_text)

# 4. Разделение данных 
X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Векторизация текста 
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Обучение модели 
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 7. Предсказание и оценка
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# 8. Простая визуализация
labels = ['Negative', 'Positive']
counts = df['sentiment'].value_counts()

plt.bar(labels, counts)
plt.title('Distribution of Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
