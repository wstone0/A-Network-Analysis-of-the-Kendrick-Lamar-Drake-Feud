import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the dataset
df = pd.read_csv('kendrick_master_dataset.csv')

# 2. Load Model AND force it onto the CPU to bypass the Mac/MPS bug
model = SentenceTransformer(
    'jinaai/jina-embeddings-v2-base-en', 
    trust_remote_code=True, 
    device='cpu' # <--- FIX #1
)

print("Converting lyrics to vectors...")
# 3. Add batch_size=4 to stop the AI from eating 12GB of RAM at once
embeddings = model.encode(
    df['lyrics'].tolist(), 
    show_progress_bar=True, 
    batch_size=4 # <--- FIX #2
)

# 4. Calculate the 109x109 Cosine Similarity Matrix
similarity_matrix = cosine_similarity(embeddings)

# 5. Convert to readable DataFrame
similarity_df = pd.DataFrame(
    similarity_matrix, 
    index=df['track_name'], 
    columns=df['track_name']
)

# Test it out!
target_song = "Cartoon and Cereal"
print(f"\nSongs most similar to '{target_song}':")
top_matches = similarity_df[target_song].sort_values(ascending=False).head(4)
print(top_matches)

# Export the similarity matrix to a CSV file
similarity_df.to_csv("kendrick_similarity_matrix.csv")