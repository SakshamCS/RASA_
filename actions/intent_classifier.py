from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import json
import pickle
from keras.utils import pad_sequences # Updated import

@DefaultV1Recipe.register("IntentClassifier", is_trainable=False)
class IntentClassifier(GraphComponent):
    def __init__(self, component_config=None):
        super().__init__(component_config)
        
        # Load model
        self.model = tf.keras.models.load_model("intent_classifier.h5")
        
        # Load tokenizer
        with open("tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        # Load intent mapping
        with open("intent_to_index.pkl", "rb") as f:
            self.intent_to_index = pickle.load(f)

        # Reverse mapping from index to intent name
        self.index_to_intent = {v: k for k, v in self.intent_to_index.items()}

    def process(self, messages, **kwargs):
        # Convert input text to model format
        sequence = self.tokenizer.texts_to_sequences([messages.text])
        sequence = pad_sequences(sequence, padding="post", maxlen=10)

        # Predict intent
        predictions = self.model.predict(sequence)[0]
        intent_index = np.argmax(predictions)
        confidence = predictions[intent_index]

        # Set intent in Rasa message
        messages.set("intent", {"name": self.index_to_intent[intent_index], "confidence": float(confidence)}, add_to_output=True)

    # def train(self, training_data, **kwargs) -> dict:
    #     # Return a fingerprintable output
    #     return {
    #         "trained": True,                  # Indicate that the component has been trained
    #         "component_name": "IntentClassifier"  # Provide the name of the component
    #     }

    @classmethod
    def load(cls, meta, model_dir: str, model_metadata, cached_component, **kwargs):
        # Initialize the component using the metadata and model directory
        component = cls(meta)
        
        # Load the model, tokenizer, and intent mappings as needed
        component.model = tf.keras.models.load_model(f"{model_dir}/intent_classifier.h5")
        
        with open(f"{model_dir}/tokenizer.pkl", "rb") as f:
            component.tokenizer = pickle.load(f)

        with open(f"{model_dir}/intent_to_index.pkl", "rb") as f:
            component.intent_to_index = pickle.load(f)

        # Reverse mapping from index to intent name
        component.index_to_intent = {v: k for k, v in component.intent_to_index.items()}
        
        return component
