from flask import Flask, render_template, request, jsonify
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import json

from api_utils import (
    generate_related_words,
    generate_trivia,
    generate_cluster_summary,
    get_gemini_embeddings  # embeddings helper
)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        theme = request.form["theme"]

        # 1. Generate related words using Gemini
        words = generate_related_words(theme)
        if not words:
            return render_template("index.html", error="Failed to generate related words.")

        # 2. Get Gemini embeddings + reduce to 2D with PCA
        embeddings = get_gemini_embeddings(words)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)

        df = pd.DataFrame({"word": words, "x": coords[:, 0], "y": coords[:, 1]})

        # 3. Clustering is mandatory
        kmeans = KMeans(n_clusters=min(3, len(words)), random_state=42)
        df["cluster"] = kmeans.fit_predict(embeddings)

        # Generate summaries for each cluster
        cluster_summaries = []
        for c in sorted(df["cluster"].unique()):
            cluster_words = df[df["cluster"] == c]["word"].tolist()
            summary = generate_cluster_summary(cluster_words)
            cluster_summaries.append({
                "id": int(c),
                "words": cluster_words,
                "summary": summary
            })

        # 4. Prepare JSON for frontend
        data_json = df.to_json(orient="records")

        return render_template(
            "index.html",
            words=words,
            data_json=data_json,
            cluster_summaries=cluster_summaries,
            theme=theme,
            status_msg="⏳ Generating semantic plot using Gemini embeddings + PCA + KMeans clustering... "
                       "💡 You can hover over each point to see its cluster info."
        )

    return render_template("index.html")


@app.route("/trivia", methods=["POST"])
def trivia():
    data = request.get_json()
    word = data.get("word")
    if not word:
        return jsonify({"error": "No word provided"}), 400

    trivia_text = generate_trivia(word)
    return jsonify({"trivia": trivia_text})


if __name__ == "__main__":
    app.run(debug=True)
