from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


def build_vector_store(chunks):

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    faiss.write_index(
        index,
        "database/faiss.index"
    )

    with open(
        "database/chunks.pkl",
        "wb"
    ) as f:
        pickle.dump(chunks, f)

    print("Index Saved")


def load_store():

    index = faiss.read_index(
        "database/faiss.index"
    )

    with open(
        "database/chunks.pkl",
        "rb"
    ) as f:
        chunks = pickle.load(f)

    return index, chunks