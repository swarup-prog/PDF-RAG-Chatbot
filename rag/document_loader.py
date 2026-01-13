from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_pdf(pdf_path: str, html_url: str = None):
    """Loads a PDF and splits it into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    if html_url:
        loader = UnstructuredHTMLLoader(html_url)
        docs = loader.load()
    else:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

    chunks = splitter.split_documents(docs)
    return chunks