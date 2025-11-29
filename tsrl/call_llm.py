
from openai import OpenAI
import aiohttp
import asyncio
TIMEOUT = 6000  
MAX_RETRIES = 3 
def call_llm(system, user):
    client = OpenAI(api_key="sk-ad429bbaf73e486ba66b29ebc209ea38", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
        max_tokens=8000,
        temperature=0.0,
    )
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    return reasoning_content, content

async def async_call_llm(system, user):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": "Bearer sk-ad429bbaf73e486ba66b29ebc209ea38",
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
        "temperature": 0.0,
    }
    
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        reasoning_content = data["choices"][0]["message"]["reasoning_content"]
                        content = data["choices"][0]["message"]["content"]
                        return reasoning_content, content
                    else:
                        print(f"⚠️ API 调用失败，状态码: {response.status}，第 {attempt} 次重试\n")
        except asyncio.TimeoutError:
            print(f"⚠️ API 超时，正在进行第 {attempt} 次重试...\n")
        except aiohttp.ClientError as e:
            print(f"⚠️ API 请求失败（{e}），第 {attempt} 次重试...\n")
        
        await asyncio.sleep(2 ** attempt)  

    print("API 调用失败，所有重试均已耗尽")
    return None, None  