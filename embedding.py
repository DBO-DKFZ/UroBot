from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    model = None

    @classmethod
    def initialize_model(cls, model_id="mixedbread-ai/mxbai-embed-large-v1"):
        if cls.model is None:
            cls.model = SentenceTransformer(model_id)

    def __call__(self, input: Documents) -> Embeddings:
        assert self.model is not None, "Model not initialized. Call initialize_model first."
        embeddings = self.model.encode(input).tolist()

        return embeddings
