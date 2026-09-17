import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

api_key = os.getenv("OPEN_ROUTER_API_KEY")
assert api_key, "OPEN_ROUTER_API_KEY not set"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


'''
    Define the tool (function) that we want to call
'''

def get_weather(latitude,longitude):
    """This is publicly available API that returns the weather for a given location"""
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data["current"]


'''
    Call model with get_weather tool defined
'''

tools = [
    {
        "type":"function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for the provided coordinates in celsius",
            "parameters":{
                "type":"object",
                "properties":{
                    "latitude": {"type":"number"},
                    "longitude": {"type":"number"}
                },
                "required":["latitude","longitude"],
                "additionalProperties": False
            },
            "strict":True
        }
    }
]

system_prompt = "You are a helpful weather assistant."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather like in Bengaluru today?"}
]

completion = client.chat.completions.create(
    model='openrouter/free',
    messages=messages,
    tools=tools
)

'''
    Model decides to call function(s)
'''

completion.model_dump()

'''
    Execute get weather function
'''

def call_function(name,args):
    if name == "get_weather":
        return get_weather(**args)

for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name,args)
    messages.append({"role":"tool", "tool_call_id":tool_call.id, "content": json.dumps(result)})

'''
    Supply result and call model again
'''

class WeatherResponse(BaseModel):
    temperature: float = Field(description="The current celsius for the given temperature")
    response: str = Field(description="A natural language response to the user's question")


completion_2 = client.beta.chat.completions.parse(
    model='openrouter/free',
    messages=messages,
    tools=tools,
    response_format=WeatherResponse
)

'''
    Check model response
'''

final_response = completion_2.choices[0].message.parsed
final_response.temperature
final_response.response