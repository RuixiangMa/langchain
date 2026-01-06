"""SentenceTransformer embeddings for multilingual semantic similarity."""

from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, ConfigDict


class SentenceTransformerEmbeddings(BaseModel, Embeddings):
    """SentenceTransformer embedding models for multilingual semantic similarity.
    This wrapper provides SentenceTransformer models for production use,
    supporting multiple languages and true semantic similarity.
    """

    model_name: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", description="Model name from sentence-transformers hub.")

    device: Optional[str] = Field(default=None, description="Device to run the model on (cpu, cuda, etc.).")

    normalize_embeddings: bool = Field(default=True, description="Whether to normalize embeddings to unit length.")

    cache_folder: Optional[str] = Field(default=None, description="Path to cache the model.")

    use_auth_token: Optional[str] = Field(default=None, description="HuggingFace authentication token.")

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True
    )

    def __init__(self, **kwargs: Any):
        """Initialize the embeddings."""
        super().__init__(**kwargs)
        self._model = None

    @property
    def model(self) -> Any:
        """Lazy load the SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_folder,
                    use_auth_token=self.use_auth_token,
                )
            except ImportError as e:
                raise ImportError(
                    "Could not import sentence-transformers package. "
                    "Please install it with `pip install sentence-transformers`."
                ) from e

        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]
