from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import json
from keras.utils import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

@DefaultV1Recipe.register("IntentClassifier", is_trainable=True)
class IntentClassifier(GraphComponent):
    def __init__(self, component_config=None):
        super().__init__(component_config)

        # Load model, tokenizer, and label encoder
        self.model = tf.keras.models.load_model("intent_model.h5")
        with open("tokenizer.json", "r") as f:
            self.tokenizer = tokenizer_from_json(json.load(f))
        with open("label_encoder.json", "r") as f:
            label_classes = json.load(f)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(label_classes)

    def process(self, message, **kwargs):
        sequence = self.tokenizer.texts_to_sequences([message.text])
        padded_sequence = pad_sequences(sequence, maxlen=5, padding="post")
        prediction = self.model.predict(padded_sequence)

        intent = self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)

        message.set("intent", {"name": intent, "confidence": confidence}, add_to_output=True)

    def train(self, training_data, **kwargs) -> dict:
        # Here you might implement any training logic if necessary
        # Returning a fingerprintable output
        return {
            "model": self.model,  # Include model or other relevant info
            "trained": True,
            "component_name": "IntentClassifier"
        }