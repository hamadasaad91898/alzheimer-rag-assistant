import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_CHAT_DEPLOYMENT")


client = OpenAI(
    api_key=api_key,
    base_url=endpoint.rstrip("/") + "/openai/v1/"
)


response = client.responses.create(
    model=deployment,
    input="Say only: GPT-5.6 Sol is working"
)


print("Model:", deployment)
print("Response:")
print(response.output_text)