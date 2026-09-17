import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

api_key = os.getenv("OPEN_ROUTER_API_KEY")
assert api_key, "OPEN_ROUTER_API_KEY not set"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

'''
    Define the response format in a Pydantic model
'''

class CalenderEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


'''
    Call the model
'''

completion = client.beta.chat.completions.parse(
    messages=[
            {"role": "system", "content":"Extract the event information."},
            {
                "role":"user",
                "content": "Alice and Bob are going to a science fair on Friday"
            }
    ],
    model='nvidia/nemotron-3-super-120b-a12b:free',
    response_format=CalenderEvent,
    
)

'''
    Parse the response
'''

event = completion.choices[0].message.parsed
event.name
event.date
event.participants