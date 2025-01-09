import os
import numpy as np
import pandas as pd

import faiss  # pip install faiss-cpu
import openai

from step2_embeddings import (
    load_video_data,
    cosine_similarity,  # We'll keep this for a quick comparison check if needed
    get_embedding_openai,
    EMBEDDING_MODEL,
    OPENAI_API_KEY
)

# import faiss
# res = faiss.StandardGpuResources()  
# index_flat = faiss.IndexFlatL2(embedding_dim)
# gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)  # 0 is the GPU ID
# gpu_index.add(embedding_matrix)


"""
Step 3: We'll build on the embeddings from Step 2, but instead of storing them in a Pandas DataFrame column,
we'll load or generate them, then add them to a Faiss index for efficient similarity search.
"""


def build_faiss_index(embeddings: np.ndarray):
    """
    Build and return a simple Faiss index.
    embeddings: A 2D float32 numpy array of shape [num_items, embedding_dim].
    """
    # 1. Determine embedding dimensions
    embedding_dim = embeddings.shape[1]

    # 2. Create a Faiss index (exact, L2-based)
    #    'IndexFlatL2' is a straightforward index that does exact search using L2 distance.
    #    For very large scale, one might use an IVF or HNSW-based index for approximate search.
    index = faiss.IndexFlatL2(embedding_dim)

    # 3. Add the embeddings to the index
    index.add(embeddings)
    return index


def prepare_embeddings_for_faiss(df_videos):
    """
    Convert the 'embedding' column from a list of floats (Python object) to a float32 NumPy array.
    """
    # df_videos["embedding"] is a column of NumPy arrays (float64). Faiss needs float32.
    embedding_list = df_videos["embedding"].tolist()  # shape: [num_items, embedding_dim]
    embedding_array = np.array(embedding_list, dtype=np.float32)
    return embedding_array


def retrieve_faiss_top_videos(query_text: str, faiss_index, df_videos, top_k=3):
    """
    Given a user query, embed it using OpenAI, then search in the Faiss index 
    to find the top_k most similar items.
    """
    # 1. Get query embedding
    openai.api_key = OPENAI_API_KEY
    query_embedding = get_embedding_openai(query_text, EMBEDDING_MODEL).astype(np.float32)

    # 2. Faiss expects shape [n, d], so we reshape to (1, embedding_dim)
    query_embedding_2d = query_embedding[np.newaxis, :]

    # 3. Search
    #    Faiss' .search() returns two arrays: distances, indices
    #    distances shape: [n, topK], indices shape: [n, topK]
    #    n = number of queries (here, 1)
    distances, indices = faiss_index.search(query_embedding_2d, top_k)

    # We'll interpret the results
    results = []
    for rank, idx in enumerate(indices[0]):
        video_id = df_videos.iloc[idx]["video_id"]
        title = df_videos.iloc[idx]["title"]
        # distance is L2, so smaller = more similar. We might convert it to similarity if needed.
        l2_distance = distances[0][rank]
        # Quick hack: we can invert L2 distance or just keep it as is. 
        # If we want to see approximate "similarity" we can do e.g., 1/(1 + l2_distance).
        results.append((video_id, title, l2_distance))

    return results


def main():
    # 1. Load data
    df_videos = load_video_data("mock_data/videos.csv")

    # 2. Check if we already have embeddings from Step 2
    #    We can embed again, or we can handle it in code. For demonstration,
    #    let's assume we embed again (repeated from Step 2).
    #    Alternatively, if you have a CSV that already stores embeddings, you could load them.
    from step2_embeddings import embed_videos
    df_videos = embed_videos(df_videos, use_openai=True)

    print("[Step 3] Embeddings generated. Building Faiss index...")

    # 3. Build Faiss index
    embedding_matrix = prepare_embeddings_for_faiss(df_videos)
    index = build_faiss_index(embedding_matrix)

    # 4. Test retrieval with Faiss
    test_query = "I love watching Category_2 videos about technology"
    top_results = retrieve_faiss_top_videos(test_query, index, df_videos, top_k=3)

    print(f"\nUser Query: {test_query}")
    print("Top videos (Faiss index):")
    for (video_id, title, dist) in top_results:
        print(f"  - ID: {video_id}, Title: {title}, L2 Distance: {dist:.4f}")

    # Let's also compare with the naive approach from Step 2 to see if the results align
    from step2_embeddings import retrieve_top_videos
    naive_results = retrieve_top_videos(test_query, df_videos, top_k=3, use_openai=True)

    print("\nFor comparison, naive approach (direct cosine similarity) gave:")
    for (vid_id, title, score) in naive_results:
        print(f"  - ID: {vid_id}, Title: {title}, Cosine Score: {score:.4f}")


if __name__ == "__main__":
    main()
