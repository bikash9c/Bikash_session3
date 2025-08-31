from flask import Flask, render_template, request, jsonify
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import time

from api_utils import (
    generate_related_words,
    generate_trivia,
    generate_cluster_summary,
    get_gemini_embeddings
)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/explore", methods=["POST"])
def explore():
    start_time = time.time()

    data = request.get_json()
    theme = data.get("theme")

    # 1. Generate related words (Gemini)
    words = generate_related_words(theme)
    if not words:
        return jsonify({"error": "Failed to generate related words."}), 500

    # 2. Embeddings from Gemini + PCA
    embeddings = get_gemini_embeddings(words)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    df = pd.DataFrame({"word": words, "x": coords[:, 0], "y": coords[:, 1]})

    # 3. Clustering
    kmeans = KMeans(n_clusters=min(3, len(words)), random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)

    cluster_summaries = []
    for c in sorted(df["cluster"].unique()):
        cluster_words = df[df["cluster"] == c]["word"].tolist()
        summary = generate_cluster_summary(cluster_words)
        cluster_summaries.append({"id": int(c), "words": cluster_words, "summary": summary})

    elapsed_time = round(time.time() - start_time, 2)

    return jsonify({
        "data": df.to_dict(orient="records"),
        "cluster_summaries": cluster_summaries,
        "theme": theme,
        "elapsed_time": elapsed_time
    })


@app.route("/trivia", methods=["POST"])
def trivia():
    data = request.get_json()
    word = data.get("word")
    if not word:
        return jsonify({"error": "No word provided"}), 400

    trivia_text = generate_trivia(word)
    return jsonify({"trivia": trivia_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

