import json
import numpy as np
import tensorflow as tf
from keras.utils import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load trained model
model = load_model("intent_model.h5")

# Load tokenizer
with open("tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(json.load(f))

# Load label encoder
with open("label_encoder.json", "r") as f:
    label_classes = json.load(f)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(label_classes)

# Load test data (same format as training data)
with open("train_data.json", "r") as f:
    data = json.load(f)

texts = data["texts"]
true_labels = label_encoder.transform(data["labels"])

# Preprocess test data
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=5, padding="post")

# Make predictions
predictions = model.predict(padded_sequences)
predicted_labels = np.argmax(predictions, axis=1)

# Evaluate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))