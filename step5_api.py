import os
import numpy as np
import pandas as pd
import openai
import faiss

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Reuse logic from previous steps
from step2_embeddings import (
    embed_videos,
    get_embedding_openai,
    EMBEDDING_MODEL,
    OPENAI_API_KEY
)

from step3_faiss import (
    build_faiss_index,
    prepare_embeddings_for_faiss
)

from step4_rerank import (
    simple_rerank,
    cosine_similarity
)

app = FastAPI(title="Video Recommendation API", version="0.1.0")


#####################
# Data/Model Classes
#####################
class RecommendationRequest(BaseModel):
    query_text: str
    top_k: int = 5


#####################
# Global Variables
#####################
df_videos = None
faiss_index = None
embedding_dim = None


#####################
# Startup Logic
#####################
@app.on_event("startup")
def load_resources():
    """
    This function is called once when the server starts.
    We'll load our data, build (or load) Faiss index, etc.
    """
    global df_videos, faiss_index, embedding_dim

    print("[Step 5] Loading data and building Faiss index...")

    # 1. Load videos
    videos_path = "mock_data/videos.csv"
    if not os.path.exists(videos_path):
        raise FileNotFoundError(
            f"{videos_path} not found. Make sure you ran Step 1 to generate mock data."
        )
    df_videos_local = pd.read_csv(videos_path)

    # 2. (Optional) Merge popularity info from interactions
    interactions_path = "mock_data/interactions.csv"
    if os.path.exists(interactions_path):
        df_interactions = pd.read_csv(interactions_path)
        video_popularity = df_interactions.groupby("video_id")["watch_count"].sum().reset_index()
        video_popularity.rename(columns={"watch_count": "popularity_score"}, inplace=True)
        df_videos_local = pd.merge(df_videos_local, video_popularity, on="video_id", how="left")
    else:
        df_videos_local["popularity_score"] = 1  # fallback if no interactions file

    # 3. Embed videos (if not done already)
    df_videos_local = embed_videos(df_videos_local, use_openai=True)

    # 4. Build Faiss index
    embedding_matrix = prepare_embeddings_for_faiss(df_videos_local)
    index = build_faiss_index(embedding_matrix)

    df_videos = df_videos_local
    faiss_index = index
    embedding_dim = embedding_matrix.shape[1]

    print("[Step 5] Server startup complete. Index is ready.")


#####################
# Helper Functions
#####################
def retrieve_candidates_with_faiss(query_text: str, faiss_index, df_videos, top_k=5):
    """
    Using Faiss to get top_k candidates quickly.
    """
    openai.api_key = OPENAI_API_KEY
    query_vector = get_embedding_openai(query_text, EMBEDDING_MODEL).astype(np.float32)
    query_vector_2d = query_vector[np.newaxis, :]

    distances, indices = faiss_index.search(query_vector_2d, top_k)

    candidate_rows = []
    for rank_idx, vid_idx in enumerate(indices[0]):
        row_dict = df_videos.iloc[vid_idx].to_dict()
        row_dict["faiss_l2_distance"] = float(distances[0][rank_idx])
        candidate_rows.append(row_dict)

    candidate_df = pd.DataFrame(candidate_rows)
    return candidate_df, query_vector


#####################
# API Endpoints
#####################
@app.post("/recommend")
def recommend_videos(req: RecommendationRequest) -> List[dict]:
    """
    Given a user query + optional top_k,
    retrieve top_k candidates from Faiss and then re-rank them with simple_rerank.
    Return the final results as JSON.
    """
    query_text = req.query_text
    top_k = req.top_k

    # 1. Retrieve from Faiss
    candidate_df, user_query_vector = retrieve_candidates_with_faiss(
        query_text, faiss_index, df_videos, top_k=top_k * 2
    )
    # We retrieve more than final top_k for better re-ranking. E.g., top_k * 2 or top_k * 3.

    # 2. Rename popularity_score -> watch_count to reuse simple_rerank logic
    if "popularity_score" in candidate_df.columns:
        candidate_df = candidate_df.rename(columns={"popularity_score": "watch_count"})

    # 3. Re-rank
    reranked_df = simple_rerank(candidate_df, user_query_vector)

    # 4. Get final top_k
    top_final = reranked_df.head(top_k).copy()

    # 5. Return as JSON
    results = []
    for _, row in top_final.iterrows():
        results.append({
            "video_id": int(row["video_id"]),
            "title": row["title"],
            "category": row.get("category", ""),
            "faiss_l2_distance": row["faiss_l2_distance"],
            "final_score": row["final_score"],
        })

    return results


#####################
# Run the server
#####################
# You typically run with: uvicorn step5_api:app --reload
# e.g. python -m uvicorn step5_api:app --host 0.0.0.0 --port 8000 --reload
