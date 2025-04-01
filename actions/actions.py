from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Import custom modules
from actions.weather import get_weather
from actions.stockPrice import get_stock_price

class ActionGetWeather(Action):
    def name(self) -> Text:
        return "action_get_weather"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        city = None

        for entity in tracker.latest_message['entities']:
            if entity['entity'] == 'city':
                city = entity['value']

        if not city:
            dispatcher.utter_message(text="I couldn't identify the city. Please specify a valid city.")
            return []

        # Call the function from weather.py
        weather_info = get_weather(city)

        dispatcher.utter_message(text=weather_info)
        return []

class ActionGetStockPrice(Action):
    def name(self) -> Text:
        return "action_get_stock_price"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        stock_symbol = None

        for entity in tracker.latest_message['entities']:
            if entity['entity'] == 'stock_symbol':
                stock_symbol = entity['value']

        if not stock_symbol:
            dispatcher.utter_message(text="Please provide a stock symbol.")
            return []

        # Call the function from stock.py
        stock_info = get_stock_price(stock_symbol)

        dispatcher.utter_message(text=stock_info)
        return []