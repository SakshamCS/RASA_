import requests

def get_weather(city):
    api_key = "1ccbd77270dc3974a3499e299ab7941a"  # Replace with your actual API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if response.status_code == 200 and "main" in data:
        temperature = data["main"]["temp"]
        weather_description = data["weather"][0]["description"]
        return f"The weather in {city} is {weather_description} with a temperature of {temperature}°C."
    else:
        return f"Sorry, I couldn't fetch the weather for {city}. Please try again."