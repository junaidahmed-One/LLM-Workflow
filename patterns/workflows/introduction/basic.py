import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

api_key = os.getenv("OPEN_ROUTER_API_KEY")
assert api_key, "OPEN_ROUTER_API_KEY not set"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

completion = client.chat.completions.create(
    messages=[
            {"role": "system", "content":"You're a helpful assistant."},
            {
                "role":"user",
                "content": "Write a limerick about the Python programming language."
            }
    ],
    model='google/gemma-4-31b-it:free',
    
)

response = completion.choices[0].message.content
print(response)