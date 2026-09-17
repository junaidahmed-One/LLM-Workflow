import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

api_key = os.getenv("OPEN_ROUTER_API_KEY")
assert api_key, "OPEN_ROUTER_API_KEY not set"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def search_kb(question:str):
    """
    Loads the whole knowledge base from the JSON file
    """

    with open("kb.json", "r") as f:
        return json.load(f)


tools = [
    {
        "type":"function",
        "function":{
            "name": "search_kb",
            "description": "Get the answer for the user's question from the knowledge base.",
            "parameters":{
                "type": "object",
                "properties": {
                    "question":{"type": "string"}
                }
            },
            "required": ["question"],
            "additionalProperties": False,
            "strict": True
        },
    }
]

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."

messages = [
    {"role":"system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"}
]


completion = client.chat.completions.create(
    model="openrouter/free",
    messages=messages,
    tools=tools
)

completion.model_dump()

def call_function(name,args):
    if name == "search_kb":
        return search_kb(**args)

for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name,args)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })


class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's questions")
    source: int = Field(description="The record id of the answer.")


completion_2 = client.beta.chat.completions.parse(
    model="openrouter/free",
    messages=messages,
    tools=tools,
    response_format=KBResponse
)

final_response = completion_2.choices[0].message.parsed
final_response.answer
final_response.source