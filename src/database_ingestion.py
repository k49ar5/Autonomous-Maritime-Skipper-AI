from logger_config import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from ollama import embeddings

PDF_FILE = "../SL_WP_IALA-Maritime-Buoyage-System.pdf"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bouye_laws"


try:
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
except Exception as e:
    logger.error(f"Filed to open a file: {e}")
    exit(1)

logger.info(f"File has  {len(docs)} sites.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap= 90,
    length_function= len,
    is_separator_regex= False
)

chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
try:

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True
    )
    logger.info("Qdrant has been created")
except Exception as e:
        logger.error(f"Qdrant hasn't been created: {e}")