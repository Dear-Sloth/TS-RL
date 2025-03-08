
from openai import OpenAI
import aiohttp
import asyncio

def call_llm(system, user):
    client = OpenAI(api_key="sk-350fda12de864af383621d1928c179a4", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
        max_tokens=8000,
        temperature=1.0,
    )
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    return reasoning_content, content

async def async_call_llm(system, user):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": "Bearer sk-350fda12de864af383621d1928c179a4",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "max_tokens": 8000,
        "temperature": 1.0,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                reasoning_content = data["choices"][0]["message"]["reasoning_content"]
                content = data["choices"][0]["message"]["content"]
                return reasoning_content, content
            else:
                return None, None