
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Doc du lieu
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Kiem tra cac cot cua du lieu
print("Cac cot:", train_df.columns.tolist())
print(train_df.head())

# Du lieu can co hai cot: message va label
X_train = train_df["message"].fillna("").astype(str)
y_train = train_df["label"]

X_test = test_df["message"].fillna("").astype(str)
y_test = test_df["label"]

# Neu nhan la chuoi thi chuan hoa ve 0 va 1
def normalize_label(series):
    mapping = {
        "0": 0,
        "1": 1,
        "ham": 0,
        "legitimate": 0,
        "not spam": 0,
        "spam": 1,
        "scam": 1
    }

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    result = series.astype(str).str.strip().str.lower().map(mapping)

    if result.isna().any():
        raise ValueError("Nhan khong hop le. Hay kiem tra cot label.")

    return result.astype(int)


y_train = normalize_label(y_train)
y_test = normalize_label(y_test)

# 2. Tao dac trung Bag of Words
# Unigram: mot tu; bigram: cum hai tu
vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    lowercase=True
)

# Chi hoc tu vung tu tap train
X_train_vec = vectorizer.fit_transform(X_train)

# Tap test chi duoc bien doi, khong hoc lai tu vung
X_test_vec = vectorizer.transform(X_test)

print("So mau train:", X_train_vec.shape[0])
print("So dac trung:", X_train_vec.shape[1])

# 3. Huan luyen Naive Bayes
model = MultinomialNB(alpha=1.0)
model.fit(X_train_vec, y_train)

# 4. Danh gia tren tap test doc lap
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# 5. Thu du doan van ban moi
def predict_spam(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    probabilities = model.predict_proba(message_vec)[0]

    print("\nVan ban:", message)

    if prediction == 1:
        print("Ket qua: Spam")
    else:
        print("Ket qua: Not Spam (Ham)")

    print("P(Ham | message):", round(probabilities[0], 4))
    print("P(Spam | message):", round(probabilities[1], 4))


predict_spam(
    "Chuc mung ban trung thuong, "
    "hay bam vao lien ket de nhan qua"
)

predict_spam(
    "Toi se den thu vien luc 8 gio sang mai"
)
