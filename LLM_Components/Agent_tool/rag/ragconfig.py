from typing import Dict, Any

class RagConfig:
    EMBEDDING_MODEL_NAME: str = "thenlper/gte-large"
    DATA_DIR: str = r"C:\Users\heman\Desktop\Coding\LlmsComponents\LLM_Components\chat_bot"
    CHUNK_SIZE: int = 512
    K: int = 5  # Number of similar documents to retrieve
    LLM_MODEL: str = "gpt-4o-mini"

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('__') and not callable(value)
        }