import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, GRU,
    GlobalAveragePooling1D, Dropout, Bidirectional
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------
# 1. Synthetic Context-Aware Text Data Generation
# -----------------------------
def generate_text_data(n_samples=2000):
    np.random.seed(42)

    positive = ["I love this product", "This is amazing", "Great experience", "Absolutely wonderful", "Highly recommended"]
    negative = ["I hate this", "Very bad experience", "Terrible product", "Not good at all", "Worst ever"]
    neutral = ["It is okay", "Not sure about this", "Average", "Nothing special", "It works fine"]

    texts = []
    labels = []

    for _ in range(n_samples):
        choice = np.random.choice(["pos", "neg", "neu"])
        if choice == "pos":
            texts.append(np.random.choice(positive))
            labels.append("positive")
        elif choice == "neg":
            texts.append(np.random.choice(negative))
            labels.append("negative")
        else:
            texts.append(np.random.choice(neutral))
            labels.append("neutral")

    return pd.DataFrame({"text": texts, "label": labels})


# -----------------------------
# 2. Data Loading & Saving
# -----------------------------
def create_and_save_dataset():
    df = generate_text_data()
    df.to_csv("nlp_text_classification_data.csv", index=False)
    print("Dataset saved as nlp_text_classification_data.csv")


# -----------------------------
# 3. Data Loading & Inspection
# -----------------------------
def load_and_inspect_data():
    df = pd.read_csv("IMDB Dataset.csv")

    df = df.rename(columns={
        "review": "text",
        "sentiment": "label"
    })

    df = df[["text", "label"]].dropna()

    print("\nDataset Head:\n", df.head())
    print("\nDataset Shape:", df.shape)
    print("\nLabel Counts:\n", df["label"].value_counts())

    return df


# -----------------------------
# 4. Preprocessing (Tokenization + Padding)
# -----------------------------
MAX_VOCAB = 5000
MAX_LEN = 20

def preprocess_data(df):
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

    le = LabelEncoder()
    labels = le.fit_transform(df["label"])

    joblib.dump(tokenizer, "nlp_tokenizer.pkl")
    joblib.dump(le, "nlp_label_encoder.pkl")

    print("Tokenizer & Label Encoder saved.")
    return padded, labels


# -----------------------------
# 5. Deep Learning NLP Model (Hybrid BiLSTM + Attention-like Dense)
# -----------------------------
def build_nlp_model():
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(MAX_VOCAB, 64, input_length=MAX_LEN)(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(3, activation='softmax')(x)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------
# 6. Training
# -----------------------------
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_nlp_model()
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("Training NLP Model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        callbacks=[early]
    )

    model.save("nlp_context_classifier.h5")
    print("Model saved as nlp_context_classifier.h5")
    return model, X_test, y_test


# -----------------------------
# 7. Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)

    print("\nAccuracy:", accuracy_score(y_test, y_pred_class))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_class))

    cm = confusion_matrix(y_test, y_pred_class)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Context-Aware NLP Confusion Matrix")
    plt.tight_layout()
    plt.savefig("nlp_confusion_matrix.png")
    plt.show()


# -----------------------------
# 8. Inference Function
# -----------------------------
def nlp_predict(text):
    tokenizer = joblib.load("nlp_tokenizer.pkl")
    le = joblib.load("nlp_label_encoder.pkl")
    model = load_model("nlp_context_classifier.h5")

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    pred = model.predict(padded)
    label = np.argmax(pred)

    return le.inverse_transform([label])[0]


# -----------------------------
# 9. Main Execution
# -----------------------------
def main():
    if not os.path.exists("nlp_text_classification_data.csv"):
        create_and_save_dataset()

    df = load_and_inspect_data()
    X, y = preprocess_data(df)
    model, X_test, y_test = train_models(X, y)
    evaluate_model(model, X_test, y_test)

    sample = "This is absolutely wonderful"
    print("\nSample Prediction:", nlp_predict(sample))


if __name__ == "__main__":
    main()
