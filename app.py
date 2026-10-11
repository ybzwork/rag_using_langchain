from data_loader import load_all_documents
from embeddings import EmbeddingPipeline

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    #print(docs)
    chunks = EmbeddingPipeline().chunk_documents(docs)
    chunkvectors = EmbeddingPipeline().embed_chunks(chunks)
    print(chunkvectors)
