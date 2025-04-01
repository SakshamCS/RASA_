# Rasa Setup & Chatbot Development - TASK 1

## Overview
This project sets up a Rasa-based chatbot that handles three intents (greeting, FAQ, and small talk) and includes a custom action to fetch live weather data from an external API.

##Result Video

https://github.com/user-attachments/assets/7f1eb21e-5456-4fbd-bc69-e6e12db02a2e

## Prerequisites
- Python 3.8–3.10
- pip
- Virtual environment (recommended)

## Installation & Setup

1. **Install Dependencies**
   ```bash
   python -m ensurepip --default-pip
   python -m pip install --upgrade pip
   python -m pip install virtualenv
   ```
2. **Create & Activate Virtual Environment**
   ```bash
   python -m venv rasa_env
   # On Windows
   rasa_env\Scripts\activate
   # On macOS/Linux
   source rasa_env/bin/activate
   ```
3. **Install Rasa**
   ```bash
   pip install rasa
   ```
4. **Verify Installation**
   ```bash
   rasa --version
   ```

## Creating the Chatbot

1. **Initialize Rasa Project**
   ```bash
   rasa init --no-prompt
   ```
2. **Define Intents** (Edit `data/nlu.yml`):
   ```yaml
   nlu:
   - intent: greet
     examples: |
       - Hi
       - Hello
   - intent: faq
     examples: |
       - What is Rasa?
   - intent: small_talk
     examples: |
       - How are you?
   ```
3. **Define Responses** (Edit `domain.yml`):
   ```yaml
   responses:
     utter_greet:
       - text: "Hello! How can I help you today?"
     utter_faq:
       - text: "Rasa is an open-source framework for AI chatbots."
     utter_small_talk:
       - text: "I'm just a chatbot, but I'm doing great!"
   ```
4. **Define Story Flow** (Edit `data/stories.yml`):
   ```yaml
   stories:
   - story: greeting_story
     steps:
     - intent: greet
     - action: utter_greet
   ```
5. **Train the Model**
   ```bash
   rasa train
   ```

## Testing the Chatbot

1. **Run Rasa Shell**
   ```bash
   rasa shell
   ```
2. **Test NLU Component**
   ```bash
   rasa shell nlu
   ```

## Implementing a Custom Action

1. **Install Requests Library**
   ```bash
   pip install requests
   ```
2. **Define Custom Action** (Create `actions/actions.py`):
   ```python
   import requests
   from rasa_sdk import Action, Tracker
   from rasa_sdk.executor import CollectingDispatcher

   class ActionGetWeather(Action):
       def name(self):
           return "action_get_weather"

       def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
           city = "Sydney"
           api_key = "YOUR_API_KEY"
           url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
           response = requests.get(url).json()
           message = f"The weather in {city} is {response['main']['temp']}°C with {response['weather'][0]['description']}."
           dispatcher.utter_message(text=message)
           return []
   ```
3. **Register Action in `domain.yml`**:
   ```yaml
   actions:
     - action_get_weather
   ```
4. **Enable Action Server** (Edit `endpoints.yml`):
   ```yaml
   action_endpoint:
     url: "http://localhost:5055/webhook"
   ```
5. **Run Action Server**
   ```bash
   rasa run actions
   ```
6. **Test Weather Query**
   ```bash
   rasa shell
   ```
   Try: *"What's the weather like?"*

## Final Testing
1. **Start Chatbot & Action Server**
   ```bash
   rasa run actions & rasa shell
   ```
2. **Interactive Learning (Optional)**
   ```bash
   rasa interactive
   ```
3. **UI Testing with Rasa X (Optional)**
   ```bash
   pip install rasa-x --extra-index-url https://pypi.rasa.com/simple
   rasa x
   ```

## Conclusion
Your Rasa chatbot is now set up with basic intents and a weather-fetching custom action. You can expand it by adding more intents, responses, and integrations!

# Deep Learning Component for Intent Classification - TASK 2

## Overview
This task involves creating a custom deep learning model for intent classification instead of using Rasa’s default pipeline. We will train a TensorFlow-based model on a small dataset, evaluate its accuracy, and integrate it into the Rasa pipeline.

## Result Video

https://github.com/user-attachments/assets/7f978ca6-5232-4b13-9211-b469788c18b2

---

## Installation

### Step 1: Install Dependencies
Before starting, install the necessary dependencies:
```bash
pip install tensorflow numpy sklearn
```
Verify the TensorFlow installation:
```python
import tensorflow as tf
print(tf.__version__)
```
If TensorFlow prints a version number, it is installed correctly.

---

## Dataset Preparation

### Step 2: Create a Training Dataset
Create a `train_data.json` file with user queries and their corresponding intents:
```json
{
  "texts": [
    "hi", "hello", "hey",
    "how are you?", "what's up?", "how's it going?",
    "what's the weather like?", "tell me the weather", "how's the weather today?",
    "can you fetch stock prices?", "give me stock updates", "what's the stock market doing?"
  ],
  "labels": [
    "greet", "greet", "greet",
    "small_talk", "small_talk", "small_talk",
    "ask_weather", "ask_weather", "ask_weather",
    "ask_stock", "ask_stock", "ask_stock"
  ]
}
```
This dataset consists of four intents:
- `greet`: "hi", "hello"
- `small_talk`: "how are you?", "what’s up?"
- `ask_weather`: "what’s the weather like?"
- `ask_stock`: "can you fetch stock prices?"

---

## Training the Model

### Step 3: Implement Deep Learning Model

Create a `train_intent_classifier.py` file with the following code:

#### 3.1 Import Libraries
```python
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
```

#### 3.2 Load and Preprocess Data
```python
# Load dataset
with open("train_data.json", "r") as f:
    data = json.load(f)

texts = data["texts"]
labels = data["labels"]

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Tokenize text
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=5, padding="post")
```

#### 3.3 Build and Train the Model
```python
# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=5),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(padded_sequences, np.array(labels_encoded), epochs=100, verbose=1)
```

#### 3.4 Save Model and Tokenizer
```python
model.save("intent_model.h5")
with open("tokenizer.json", "w") as f:
    json.dump(tokenizer.to_json(), f)
with open("label_encoder.json", "w") as f:
    json.dump(label_encoder.classes_.tolist(), f)
```

Run the script:
```bash
python train_intent_classifier.py
```
---

## Evaluating the Model

### Step 4: Test the Model
Create `test_intent_classifier.py` with the following code:

#### 4.1 Load Model and Tokenizer
```python
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder

# Load model
model = load_model("intent_model.h5")

# Load tokenizer
with open("tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(json.load(f))

# Load label encoder
with open("label_encoder.json", "r") as f:
    label_classes = json.load(f)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(label_classes)
```

#### 4.2 Define Prediction Function
```python
def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=5, padding="post")
    prediction = model.predict(padded_sequence)
    intent = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return intent

# Test model
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    intent = predict_intent(user_input)
    print(f"Predicted Intent: {intent}")
```

Run:
```bash
python test_intent_classifier.py
```

---
## Further Evaluating Model Accuracy

#### 4.3 Load the Trained Model

Create a new Python script (evaluate_model.py) and load the model:
```python
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
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
```
#### 4.3 Run the Evaluation

Run the script:
```bash
python evaluate_model.py
```
This will print the accuracy and a detailed classification report.

## Integration with Rasa

### Step 5: Modify Rasa Configuration

#### 5.1 Update `config.yml`
```yaml
language: en
pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: actions.intent_classifier.IntentClassifier
  - name: EntitySynonymMapper
  - name: ResponseSelector
  - name: DIETClassifier
    epochs: 100
```

#### 5.2 Implement Custom Intent Classifier (`intent_classifier.py`)
```python
from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import json
import pickle
from keras.utils import pad_sequences

@DefaultV1Recipe.register("IntentClassifier", is_trainable=False)
class IntentClassifier(GraphComponent):
    def __init__(self, component_config=None):
        super().__init__(component_config)
        self.model = tf.keras.models.load_model("intent_classifier.h5")
        with open("tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)
        with open("intent_to_index.pkl", "rb") as f:
            self.intent_to_index = pickle.load(f)
        self.index_to_intent = {v: k for k, v in self.intent_to_index.items()}

    def process(self, messages, **kwargs):
        sequence = self.tokenizer.texts_to_sequences([messages.text])
        sequence = pad_sequences(sequence, padding="post", maxlen=10)
        predictions = self.model.predict(sequence)[0]
        intent_index = np.argmax(predictions)
        confidence = predictions[intent_index]
        messages.set("intent", {"name": self.index_to_intent[intent_index], "confidence": float(confidence)}, add_to_output=True)
```

---

## Running Rasa

### Step 6: Train and Test Rasa

Run:
```bash
rasa train
rasa shell
```
Test with:
```bash
You: hello
Bot: Hello! How can I help you?
```

---

## Conclusion

🎯 You’ve implemented a deep learning intent classifier for Rasa.



