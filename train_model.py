from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import os


df = pd.read_csv("dataset/transactions.csv")


category_mapping = {
    "Restaurants": "Food",
    "Fast Food": "Food",
    "Coffee Shops": "Food",
    "Food & Dining": "Food",

    "Movies & DVDs": "Entertainment",
    "Music": "Entertainment",
    "Television": "Entertainment",

    "Utilities": "Bills",
    "Internet": "Bills",
    "Mobile Phone": "Bills",

    "Gas & Fuel": "Travel",

    "Paycheck": "Income",

    "Credit Card Payment": "Banking"
}


df["Category"] = df["Category"].replace(category_mapping)

print(df["Category"].value_counts())


X = df["Description"]
y = df["Category"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True
)

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vector, y_train)


y_pred = model.predict(X_test_vector)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, "vectorizer.pkl")
)