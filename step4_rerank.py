import os
import numpy as np
import pandas as pd
import openai
import faiss

from step2_embeddings import (
    load_video_data,
    embed_videos,
    get_embedding_openai,
    OPENAI_API_KEY,
    EMBEDDING_MODEL
)
from step3_faiss import (
    build_faiss_index,
    prepare_embeddings_for_faiss
)


######################
# Simple Re-rank Logic
######################
def simple_rerank(candidate_df, user_query_vector):
    """
    Given a DataFrame of candidate items (with columns like embedding, watch_count, etc.),
    apply a simple re-ranking strategy:
      1. Compute embedding similarity
      2. Combine similarity with 'watch_count' or 'popularity' to get a final score

    The final output is the DataFrame sorted by that final score.
    """
    # We'll compute a raw similarity first
    similarity_scores = []
    for idx, row in candidate_df.iterrows():
        item_embedding = row["embedding"]
        sim = cosine_similarity(user_query_vector, item_embedding)
        # Example combination:
        # final_score = alpha * sim + beta * (popularity measure)
        # For now, let's just do something with 'watch_count' if it exists
        watch_count = row.get("watch_count", 1)  # fallback if missing
        final_score = sim + 0.02 * watch_count  # made-up weighting

        similarity_scores.append(final_score)

    candidate_df["final_score"] = similarity_scores
    # sort by descending final_score
    candidate_df = candidate_df.sort_values("final_score", ascending=False)
    return candidate_df


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


##################
# Main Logic
##################
def retrieve_candidates_with_faiss(query_text: str, faiss_index, df_videos, top_k=20):
    """
    Using Faiss to get top_k candidates quickly. We'll do a final re-rank on these.
    """
    openai.api_key = OPENAI_API_KEY
    query_vector = get_embedding_openai(query_text, EMBEDDING_MODEL).astype(np.float32)
    query_vector_2d = query_vector[np.newaxis, :]

    distances, indices = faiss_index.search(query_vector_2d, top_k)

    # Build a DataFrame of the top_k candidates
    candidate_rows = []
    for rank_idx, vid_idx in enumerate(indices[0]):
        row_dict = df_videos.iloc[vid_idx].to_dict()
        row_dict["faiss_l2_distance"] = distances[0][rank_idx]
        candidate_rows.append(row_dict)
    candidate_df = pd.DataFrame(candidate_rows)
    return candidate_df, query_vector


def main():
    # 1. Load video data
    df_videos = load_video_data("mock_data/videos.csv")

    # 2. In a real scenario, we'd also have some usage metrics or "popularity" stats (like watch_count)
    #    For the sake of demonstration, let's join data from interactions.csv to get watch_count.
    interactions_path = "mock_data/interactions.csv"
    df_interactions = pd.read_csv(interactions_path)

    # We might do something like "average watch_count" per video or sum it as a popularity measure:
    video_popularity = df_interactions.groupby("video_id")["watch_count"].sum().reset_index()
    video_popularity.rename(columns={"watch_count": "popularity_score"}, inplace=True)

    # Merge popularity info into df_videos
    df_videos = pd.merge(df_videos, video_popularity, on="video_id", how="left")

    # 3. Generate embeddings (if not already present)
    df_videos = embed_videos(df_videos, use_openai=True)

    # 4. Build Faiss index
    embedding_matrix = prepare_embeddings_for_faiss(df_videos)
    index = build_faiss_index(embedding_matrix)

    # 5. Retrieve candidates
    query_text = "I love tech videos from Category_2"
    candidate_df, user_query_vector = retrieve_candidates_with_faiss(query_text, index, df_videos, top_k=5)

    print("[Step 4] Candidates retrieved from Faiss:")
    print(candidate_df[["video_id", "title", "faiss_l2_distance", "popularity_score"]])

    # 6. Re-rank with additional metadata
    #    For illustration, let's say we also want to incorporate popularity_score
    #    We'll rename popularity_score -> watch_count if we want to keep the same naming logic
    candidate_df.rename(columns={"popularity_score": "watch_count"}, inplace=True)

    reranked_df = simple_rerank(candidate_df, user_query_vector)

    # 7. Show final top results
    top_final = reranked_df.head(3)

    print("\nFinal Top-3 after re-ranking:")
    for idx, row in top_final.iterrows():
        print(f" - Video ID: {row['video_id']}, Title: {row['title']}, Final Score: {row['final_score']:.4f}")


if __name__ == "__main__":
    main()
