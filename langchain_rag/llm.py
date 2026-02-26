import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()



client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class GroqLLM:

    def invoke(self, prompt: str) -> str:

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert MySQL assistant. Return ONLY valid SQL."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=200
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            print("LLM Error:", e)
            return ""


llm = GroqLLM()

