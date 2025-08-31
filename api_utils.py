import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)


def generate_related_words(theme: str, n: int = 15):
    """Ask Gemini for related words to a theme."""
    prompt = f"Give me {n} words related to '{theme}'. Return ONLY a JSON array of words."
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()
        return json.loads(text) if text.startswith("[") else []
    except Exception as e:
        print("Error generating words:", e)
        return []


def generate_trivia(word: str):
    """Ask Gemini for a fun trivia fact or sentence about a word."""
    prompt = f"Give me a short trivia fact or fun sentence using the word '{word}' (max 40 words)."
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Error generating trivia:", e)
        return "Trivia not available."


def generate_cluster_summary(words):
    """Summarize what words in a cluster have in common."""
    prompt = f"Summarize in 1-2 sentences what these words have in common: {', '.join(words)}"
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Error generating cluster summary:", e)
        return "No summary available."
