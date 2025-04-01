# Rasa Setup & Chatbot Development

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

