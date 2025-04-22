import os
import requests
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Variables necesarias
GROQ_API_KEY  = os.getenv("GROQ_API_KEY_")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama3-8b-8192")

def groq_chat(prompt: str, max_tokens: int = 200) -> str:
    if not GROQ_API_KEY:
        raise ValueError("❌ Missing GROQ_API_KEY")

    # Llamada correcta a la API de Groq usando POST
    response = requests.post(
        f"{GROQ_API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

