import pandas as pd
import os

def generate_mock_data(num_users=5, num_videos=10):
    """Generate a small set of mock data for demonstration."""
    # User data: just an ID and maybe a "preferred category" for now
    users = []
    for user_id in range(1, num_users+1):
        users.append({
            "user_id": user_id,
            "preferred_category": f"Category_{(user_id % 3)+1}"
        })
    df_users = pd.DataFrame(users)

    # Video (item) data: each has ID, title, category
    videos = []
    for vid_id in range(1, num_videos+1):
        videos.append({
            "video_id": vid_id,
            "title": f"Video_{vid_id}",
            "category": f"Category_{(vid_id % 3)+1}"
        })
    df_videos = pd.DataFrame(videos)

    # Interaction data: user -> video watch
    interactions = []
    for user_id in range(1, num_users+1):
        for vid_id in range(1, num_videos+1):
            # For simplicity, assume that every user has watched every video 
            # but we can add a simple random watch count or rating
            watch_count = 0
            if (user_id % 3) == (vid_id % 3):  # Some pattern for "interest"
                watch_count = 3  # They "watched" it more often
            else:
                watch_count = 1  # minimal watch

            interactions.append({
                "user_id": user_id,
                "video_id": vid_id,
                "watch_count": watch_count
            })
    df_interactions = pd.DataFrame(interactions)

    return df_users, df_videos, df_interactions


def main():
    # 1. Generate data
    df_users, df_videos, df_interactions = generate_mock_data(
        num_users=5,  # you can vary these numbers
        num_videos=10
    )

    # 2. Create an output directory if needed
    out_dir = "mock_data"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 3. Save CSV files
    df_users.to_csv(os.path.join(out_dir, "users.csv"), index=False)
    df_videos.to_csv(os.path.join(out_dir, "videos.csv"), index=False)
    df_interactions.to_csv(os.path.join(out_dir, "interactions.csv"), index=False)

    print("\n[Step 1] Mock data created and saved in 'mock_data' folder.")
    print("Users:")
    print(df_users.head())
    print("Videos:")
    print(df_videos.head())
    print("Interactions:")
    print(df_interactions.head())

if __name__ == "__main__":
    main()
