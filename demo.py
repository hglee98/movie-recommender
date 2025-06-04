import gradio as gr
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024
)

client = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi('1'))
db = client.sample_mflix
collection = db.movies


def recommend_movie(query_text):
    results = collection.aggregate([{
        "$vectorSearch": {
            "queryVector": embedding_model.embed_query(query_text),
            "path": "plot_embedding",
            "numCandidates": 100,
            "limit": 4,
            "index": "PlotSemanticindex",
        }
    }])
    print(results)

    recommendations = ""
    for doc in results:
        title = doc.get('title', 'Unknown Title')
        plot = doc.get('plot', 'No description available.')
        poster = doc.get('poster', '')

        image_html = f"<img src='{poster}' alt='{title} 포스터' width='200'>" if poster else ""

        recommendations += f"""
                <div style='margin-bottom: 20px;'>
                    <h3>{title}</h3>
                    {image_html}
                    <p>{plot}</p>
                </div>
                """
    return recommendations


demo = gr.Interface(
    fn=recommend_movie,
    inputs=gr.Textbox(lines=2, placeholder="Type your movie-related query here..."),
    outputs=gr.HTML(),
    title="Movie Recommendation System",
    description="입력한 질문을 기반으로 MongoDB에서 영화를 추천해줍니다. (벡터 검색 기반)"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
