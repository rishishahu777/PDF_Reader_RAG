from sentence_transformers import SentenceTransformer
from vector_store import load_store
import numpy as np

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


def retrieve_context(
    question,
    top_k=3
):

    index, chunks = load_store()

    query_embedding = model.encode(
        [question]
    )

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    context = ""

    for idx in indices[0]:
        context += chunks[idx]
        context += "\n\n"

    return context