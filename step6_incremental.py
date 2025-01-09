import os
import numpy as np
import pandas as pd
import openai
import faiss

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

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
from step4_rerank import simple_rerank

########################################################
# FASTAPI APP WITH INCREMENTAL UPDATES & LOGGING
########################################################

app = FastAPI(title="Video Recommendation API (Step 6)", version="0.2.0")


##############
# Data Models
##############
class RecommendationRequest(BaseModel):
    query_text: str
    top_k: int = 5

class InteractionLog(BaseModel):
    user_id: int
    video_id: int
    watch_count: int = 1  # or watch_time, like/dislike, etc.

class NewVideo(BaseModel):
    video_id: int
    title: str
    category: str
    # Additional fields as needed


########################
# Global Data & Index
########################
df_videos = None        # Will hold video metadata + embeddings
df_interactions = None  # Will hold user->video interactions
faiss_index = None      # Faiss index
embedding_dim = None

########################
# Startup / Initialization
########################
@app.on_event("startup")
def load_resources():
    """
    On startup, load existing data, build Faiss index.
    We'll keep them in global variables.
    """
    global df_videos, df_interactions, faiss_index, embedding_dim
    print("[Step 6] Loading data on startup...")

    # 1. Load existing videos
    videos_path = "mock_data/videos.csv"
    df_videos_local = pd.read_csv(videos_path)
    
    # 2. Load or create interactions
    interactions_path = "mock_data/interactions.csv"
    if os.path.exists(interactions_path):
        df_inter_local = pd.read_csv(interactions_path)
    else:
        # If not found, create an empty DataFrame
        df_inter_local = pd.DataFrame(columns=["user_id", "video_id", "watch_count"])

    # 3. Compute popularity from interactions
    if not df_inter_local.empty:
        video_popularity = df_inter_local.groupby("video_id")["watch_count"].sum().reset_index()
        video_popularity.rename(columns={"watch_count": "popularity_score"}, inplace=True)
        df_videos_local = pd.merge(df_videos_local, video_popularity, on="video_id", how="left")
    else:
        df_videos_local["popularity_score"] = 1

    # Fill NaNs
    df_videos_local["popularity_score"] = df_videos_local["popularity_score"].fillna(1)

    # 4. Embed videos if needed
    df_videos_local = embed_videos(df_videos_local, use_openai=True)

    # 5. Build Faiss index
    embedding_matrix = prepare_embeddings_for_faiss(df_videos_local)
    index = build_faiss_index(embedding_matrix)

    # 6. Assign to globals
    df_videos = df_videos_local
    df_interactions = df_inter_local
    faiss_index = index
    embedding_dim = embedding_matrix.shape[1]

    print("[Step 6] Initialization complete. Index ready.")


########################
# Helper Functions
########################
def retrieve_candidates_with_faiss(query_text: str, top_k=5):
    global df_videos, faiss_index

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


def refresh_faiss_index():
    """
    Rebuild the Faiss index from df_videos.
    You can do partial updates if you prefer, but here we rebuild for simplicity.
    """
    global df_videos, faiss_index, embedding_dim
    embedding_matrix = prepare_embeddings_for_faiss(df_videos)
    faiss_index = build_faiss_index(embedding_matrix)
    embedding_dim = embedding_matrix.shape[1]


########################
# API Endpoints
########################
@app.post("/recommend")
def recommend_videos(req: RecommendationRequest) -> List[dict]:
    """
    Retrieve and re-rank top_k recommendations.
    """
    top_k = req.top_k

    # First, get top candidates from Faiss
    candidate_df, user_query_vector = retrieve_candidates_with_faiss(req.query_text, top_k=top_k*2)

    # Re-rank using simple_rerank (with popularity)
    if "popularity_score" in candidate_df.columns:
        # rename for simple_rerank
        candidate_df = candidate_df.rename(columns={"popularity_score": "watch_count"})
    reranked_df = simple_rerank(candidate_df, user_query_vector)

    # Take final top_k
    top_final = reranked_df.head(top_k).copy()

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


@app.post("/log_interaction")
def log_interaction(log: InteractionLog) -> dict:
    """
    Log a user->video interaction. 
    We'll increment the watch_count in df_interactions and 
    refresh popularity in df_videos if needed.
    """
    global df_videos, df_interactions

    # 1. Append to df_interactions
    new_row = {
        "user_id": log.user_id,
        "video_id": log.video_id,
        "watch_count": log.watch_count
    }
    df_interactions = pd.concat([df_interactions, pd.DataFrame([new_row])], ignore_index=True)

    # 2. Update popularity in df_videos
    popularity_sum = df_interactions[df_interactions["video_id"] == log.video_id]["watch_count"].sum()
    # Find the row in df_videos
    idx = df_videos.index[df_videos["video_id"] == log.video_id]
    if not idx.empty:
        df_videos.loc[idx, "popularity_score"] = popularity_sum
    else:
        # If the video_id doesn't exist, do nothing or handle error
        pass

    # 3. Optionally, refresh the Faiss index if we want real-time updates
    #    (But for a large system, you might only do it periodically.)
    refresh_faiss_index()

    # 4. Save to disk so we don't lose updates (in real life, you'd likely use a DB)
    if not os.path.exists("mock_data"):
        os.makedirs("mock_data")
    df_interactions.to_csv("mock_data/interactions.csv", index=False)

    return {
        "message": f"Logged interaction for user={log.user_id}, video={log.video_id} (watch_count={log.watch_count}).",
        "current_popularity": int(popularity_sum),
    }


@app.post("/add_video")
def add_video(new_video: NewVideo) -> dict:
    """
    Add a new video to df_videos, embed it, and add it to the Faiss index (or rebuild).
    """
    global df_videos

    # Check if video_id already exists
    if new_video.video_id in df_videos["video_id"].values:
        return {"error": f"Video ID {new_video.video_id} already exists."}

    new_row_dict = {
        "video_id": new_video.video_id,
        "title": new_video.title,
        "category": new_video.category,
        "popularity_score": 1.0  # default
    }
    # Convert to DataFrame and append
    new_df = pd.DataFrame([new_row_dict])
    df_videos = pd.concat([df_videos, new_df], ignore_index=True)

    # Re-embed just the new row (or re-embed everything)
    # For efficiency, embed only the new item:
    from step2_embeddings import get_embedding_openai
    openai.api_key = OPENAI_API_KEY

    text_to_embed = f"{new_video.title} {new_video.category}"
    embedding_vec = get_embedding_openai(text_to_embed, EMBEDDING_MODEL).astype(np.float32)

    df_videos.at[df_videos.index[-1], "embedding"] = embedding_vec

    # Refresh the Faiss index
    refresh_faiss_index()

    # Also persist to disk, so we don't lose it
    if not os.path.exists("mock_data"):
        os.makedirs("mock_data")
    # Overwrite the CSV (for demonstration). A real system might do partial updates or DB.
    df_videos.to_csv("mock_data/videos.csv", index=False)

    return {"message": f"Video {new_video.video_id} added successfully."}
