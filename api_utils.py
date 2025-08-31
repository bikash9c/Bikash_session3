import os
import re, json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)


def generate_related_words(theme: str):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Generate about 15 words related to the theme '{theme}' as a JSON list of strings only."
    response = model.generate_content(prompt)
    text = response.text.strip()

    # Extract JSON array even if wrapped in markdown/code fences
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception as e:
            print("JSON parsing error:", e, "Raw:", text)
            return []

    print("No JSON list found, raw response:", text)
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

def get_gemini_embeddings(words: list):
    """Return embeddings for a list of words using Gemini embeddings API"""
    model = "models/embedding-001"
    embeddings = []
    for word in words:
        resp = genai.embed_content(model=model, content=word)
        embeddings.append(resp["embedding"])
    return embeddings
