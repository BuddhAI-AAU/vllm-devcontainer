from openai import OpenAI

# Point the client to your local vLLM server
client = OpenAI(
    api_key="EMPTY",                     # vLLM ignores the key unless you enable auth
    base_url="http://localhost:8000/v1"  # your local server
)

# Simple prompt loop
while True:
    prompt = input("You: ")
    if not prompt.strip():
        break

    response = client.completions.create(
        model="allenai/OLMo-1B-hf",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )

    print("Model:", response.choices[0].text.strip())
