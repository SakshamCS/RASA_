import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load dataset
with open("train_data.json", "r") as f:
    data = json.load(f)

texts = data["texts"]
labels = data["labels"]

# Encode labels as numbers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Tokenize text data
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=5, padding="post")

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=5),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(padded_sequences, np.array(labels_encoded), epochs=100, verbose=1)

# Save model and necessary files
model.save("intent_model.h5")
with open("tokenizer.json", "w") as f:
    json.dump(tokenizer.to_json(), f)
with open("label_encoder.json", "w") as f:
    json.dump(label_encoder.classes_.tolist(), f)

print("Model trained and saved successfully!")