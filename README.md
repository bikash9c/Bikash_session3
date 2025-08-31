# 🔎 Semantic Explorer (Flask + Gemini + Plotly)

This project is an **interactive semantic exploration app** built with **Flask**, **Google Gemini API**, **Plotly**, and **scikit-learn**.  

You can enter a **word or short phrase** (a "theme"), and the app will:  
1. Use **Gemini** to generate related words.  
2. Create embeddings with **Gemini**.  
3. Apply **PCA** to reduce dimensions.  
4. Cluster words using **KMeans**.  
5. Use **Gemini** again to summarize each cluster.  
6. Display results on an **interactive Plotly graph** where you can hover/click on words to get trivia.

---

## 🚀 Features
- **Semantic Map:** Words are plotted in 2D using PCA.  
- **Clustering:** Each word is assigned a cluster with color-coding.  
- **Cluster Summaries:** Auto-generated summaries of clusters with Gemini.  
- **Trivia on Click:** Click any word on the plot to get trivia from Gemini.  
- **Progress Feedback:** Users see step-by-step status updates (fake ticks) while the plot is generated.  

---

## 📂 Project Structure
