# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
import argparse
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build(data_path, recursive):
    loader = DirectoryLoader(data_path,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader,
                             recursive=recursive,
                             show_progress=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Build FAISS Vector Database")
    parser.add_argument('-p', type=str, default=cfg.DATA_PATH,
                        help="Path to the directory containing PDF files. Default is set in config.")
    parser.add_argument('-r', type=bool, default=True,
                        help="Whether to load files recursively. Default is True.")
    
    args = parser.parse_args()
    
    run_db_build(args.p, args.r)
