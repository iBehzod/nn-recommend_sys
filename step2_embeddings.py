import os
import pandas as pd
import numpy as np

# If you want to use OpenAI
import openai

# If you do NOT want to use OpenAI and prefer local embeddings, comment out the above import
# and uncomment the below line
# from sentence_transformers import SentenceTransformer

##########################
# Configurable constants #
##########################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
EMBEDDING_MODEL = "text-embedding-ada-002"  # For OpenAI

# If using Sentence Transformers locally:
# MODEL_NAME = "all-MiniLM-L6-v2"  # an example of a small model
##########################


def get_embedding_openai(text: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Get the embedding vector for a piece of text from OpenAI's API.
    """
    response = openai.Embedding.create(
        input=[text],
        model=model
    )
    # The 'data' field is a list of embeddings for each input
    embedding_vector = response['data'][0]['embedding']
    return np.array(embedding_vector, dtype=float)


def get_embedding_local(text: str, model):
    """
    Get embedding from a local (SentenceTransformer) model.
    """
    embedding_vector = model.encode([text])[0]
    return np.array(embedding_vector, dtype=float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_video_data(csv_path="mock_data/videos.csv"):
    """
    Load the video data from Step 1.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Make sure to run Step 1 first.")
    df_videos = pd.read_csv(csv_path)
    return df_videos


def embed_videos(df_videos, use_openai=True):
    """
    Embed each video's title and category (as a proxy for 'description').
    Store the embedding in a new 'embedding' column.
    """
    embeddings = []
    if use_openai:
        # Set your API key (or comment out if not using)
        openai.api_key = OPENAI_API_KEY

    else:
        # If you're using local embeddings:
        # model = SentenceTransformer(MODEL_NAME)
        pass

    for _, row in df_videos.iterrows():
        title = row["title"]
        category = row["category"]
        text_to_embed = f"{title} {category}"

        if use_openai:
            embedding = get_embedding_openai(text_to_embed)
        else:
            # embedding = get_embedding_local(text_to_embed, model)
            # For now, just raise NotImplemented error if we haven't set up local
            raise NotImplementedError("Local embedding not implemented in this example")

        embeddings.append(embedding)

    df_videos["embedding"] = embeddings
    return df_videos


def retrieve_top_videos(user_query: str, df_videos, top_k=3, use_openai=True):
    """
    Given a user's query (text), retrieve the top_k most similar videos.
    """
    if use_openai:
        openai.api_key = OPENAI_API_KEY
        user_query_vector = get_embedding_openai(user_query)
    else:
        raise NotImplementedError("Local embedding not implemented in this example.")

    results = []
    for _, row in df_videos.iterrows():
        sim_score = cosine_similarity(user_query_vector, row["embedding"])
        results.append((row["video_id"], row["title"], sim_score))

    # Sort by descending similarity
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]


def main():
    # 1. Load video data
    df_videos = load_video_data()

    # 2. Generate embeddings
    df_videos = embed_videos(df_videos, use_openai=True)

    print("[Step 2] Generated embeddings. Sample:")
    print(df_videos.head())

    # 3. Test retrieval
    test_query = "I want to see videos related to Category_2"
    top_videos = retrieve_top_videos(test_query, df_videos, top_k=3, use_openai=True)

    print(f"\nUser Query: {test_query}")
    print("Top videos:")
    for vid_id, title, score in top_videos:
        print(f"  - ID: {vid_id}, Title: {title}, Score: {score:.4f}")


if __name__ == "__main__":
    main()
