from typing import List, Optional, Any, Callable
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from rag_config import DocumentSplitterconfig
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]
class DocumentSplitter:
    def __init__(
        self,
        config: DocumentSplitterconfig
    ):
        self.config = config
       
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            separators=self.config.separators,
            keep_separator=self.config.keep_separator,
            is_separator_regex=self.config.is_separator_regex,
            tokenizer=self.config.tokenizer,
            chunk_size=self.config.chunk_size,
            chunk_overlap=int(self.config.chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            **self.config.additional_args
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        
        Args:
            documents (List[Document]): List of input documents.
        
        Returns:
            List[Document]: List of split documents.
        """
        split_docs = []
        
        for doc in documents:
            splits = self.text_splitter.split_text(doc.page_content)
            for split in splits:
                split_doc = Document(
                    page_content=split,
                    metadata={**doc.metadata, "chunk_size": self.chunk_size}
                )
                split_docs.append(split_doc)
        
        # Remove duplicates
        unique_splits = list({doc.page_content: doc for doc in split_docs}.values())
        
        return unique_splits


