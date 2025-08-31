# 🌌 Semantic Explorer

A Flask web app that visualizes semantic relationships between words using embeddings, PCA, and Plotly — powered by Google Gemini.

## 🚀 Features
- Enter a **theme** (e.g., "space", "emotions").
- Generate ~15 related words using **Gemini**.
- Embed words with `sentence-transformers`.
- Reduce to 2D using PCA.
- Interactive **Plotly scatter plot**.
- Click a word → see a fun **trivia fact**.
- (Optional) Cluster words with k-means and get **cluster summaries**.

## 📦 Setup
```bash
git clone https://github.com/your-repo/semantic-explorer.git
cd semantic-explorer
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
