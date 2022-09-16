from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'all-MiniLM-L6-v2', 
    cache_folder = "sbert_cache",
    device = "cpu"
)

