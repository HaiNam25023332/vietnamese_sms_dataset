import hashlib
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
FEATURES = 8192
BATCH_SIZE = 32
EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


def normalize_labels(series):
    mapping = {
        "0": 0, "1": 1, "ham": 0, "legitimate": 0,
        "not spam": 0, "spam": 1, "scam": 1,
    }
    if pd.api.types.is_numeric_dtype(series):
        labels = series.astype(int)
    else:
        labels = series.astype(str).str.strip().str.lower().map(mapping)
        if labels.isna().any():
            raise ValueError("Cot label phai gom 0/1 hoac ham/spam.")
        labels = labels.astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Nhan phai chi gom 0 (Ham) va 1 (Spam).")
    return labels.to_numpy(dtype=np.float32)


def text_to_features(text):
    """Turn a message into a normalized character n-gram hash vector."""
    text = str(text).lower()
    counts = Counter()
    for n in range(2, 6):
        for start in range(max(0, len(text) - n + 1)):
            ngram = text[start:start + n].encode("utf-8")
            # Stable hash (unlike Python's randomized built-in hash).
            index = int.from_bytes(
                hashlib.blake2b(ngram, digest_size=8).digest(), "little"
            ) % FEATURES
            counts[index] += 1

    vector = np.zeros(FEATURES, dtype=np.float32)
    for index, count in counts.items():
        vector[index] = count
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def make_matrix(messages):
    return np.stack([text_to_features(message) for message in messages])


# Split train data into train/validation without using the independent test set.
y_all = normalize_labels(train_df["label"])
indices = np.arange(len(train_df))
rng = np.random.default_rng(SEED)
fit_indices, val_indices = [], []
for label in (0, 1):
    group = indices[y_all == label]
    rng.shuffle(group)
    val_count = max(1, round(len(group) * 0.2))
    val_indices.extend(group[:val_count])
    fit_indices.extend(group[val_count:])

messages = train_df["message"].fillna("").astype(str).tolist()
test_messages = test_df["message"].fillna("").astype(str).tolist()
X_all = make_matrix(messages)
X_test = make_matrix(test_messages)

X_fit = torch.tensor(X_all[fit_indices], dtype=torch.float32)
y_fit = torch.tensor(y_all[fit_indices], dtype=torch.float32)
X_val = torch.tensor(X_all[val_indices], dtype=torch.float32, device=DEVICE)
y_val = torch.tensor(y_all[val_indices], dtype=torch.float32, device=DEVICE)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
y_test = normalize_labels(test_df["label"])

fit_loader = DataLoader(
    TensorDataset(X_fit, y_fit), batch_size=BATCH_SIZE, shuffle=True
)

class SMSDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURES, 256),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


model = SMSDNN().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = float("inf")
best_state = None
patience = 5
stale_epochs = 0

print(f"Device: {DEVICE}")
print(f"So mau train: {len(fit_indices)}")
print(f"So mau validation: {len(val_indices)}")
print(f"So mau test: {len(test_df)}")
print(f"So dac trung hash: {FEATURES}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in fit_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_y)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val).item()
        val_accuracy = ((torch.sigmoid(val_logits) >= 0.5) == y_val.bool()).float().mean().item()

    print(
        f"Epoch {epoch:02d}/{EPOCHS} - "
        f"train_loss={total_loss / len(fit_indices):.4f} "
        f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        stale_epochs = 0
    else:
        stale_epochs += 1
        if stale_epochs >= patience:
            print("Dung som: validation loss khong con cai thien.")
            break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    test_probabilities = torch.sigmoid(model(X_test_tensor)).cpu().numpy()
y_pred = (test_probabilities >= 0.5).astype(int)

accuracy = float(np.mean(y_pred == y_test))
print(f"\nTest accuracy: {accuracy:.4f}")
for label, name in ((0, "Ham"), (1, "Spam")):
    true_positive = int(np.sum((y_test == label) & (y_pred == label)))
    false_positive = int(np.sum((y_test != label) & (y_pred == label)))
    false_negative = int(np.sum((y_test == label) & (y_pred != label)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"{name}: precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}")

print("Confusion matrix (rows=true [Ham, Spam], columns=predicted [Ham, Spam]):")
print(np.array([
    [np.sum((y_test == 0) & (y_pred == 0)), np.sum((y_test == 0) & (y_pred == 1))],
    [np.sum((y_test == 1) & (y_pred == 0)), np.sum((y_test == 1) & (y_pred == 1))],
], dtype=int))

torch.save({"state_dict": model.state_dict(), "features": FEATURES}, "sms_dnn_pytorch.pt")
print("Da luu model: sms_dnn_pytorch.pt")


def predict_spam(message):
    features = torch.tensor(text_to_features(message), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        spam_probability = torch.sigmoid(model(features)).item()
    prediction = int(spam_probability >= 0.5)
    print("\nTin nhan:", message)
    print("Ket qua:", "Spam" if prediction else "Ham (tin nhan hop le)")
    print(f"P(Ham): {1 - spam_probability:.4f}")
    print(f"P(Spam): {spam_probability:.4f}")
    return prediction, spam_probability


# Thu bo dau # de thu du doan tin nhan moi:
predict_spam("Chuc mung ban trung thuong, bam vao lien ket de nhan qua")
