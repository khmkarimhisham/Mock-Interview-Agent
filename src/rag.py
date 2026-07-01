import os
import glob
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

KNOWLEDGE_BASE_DIR = "knowledge_base"
CHROMA_DB_DIR = ".chroma_db"

def load_documents():
    documents = []
    # Load .txt files
    for txt_file in glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "**/*.txt"), recursive=True):
        try:
            loader = TextLoader(txt_file, encoding='utf-8')
            documents.extend(loader.load())
            print(f"Loaded {txt_file}")
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
            
    # Load .md files
    for md_file in glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "**/*.md"), recursive=True):
        try:
            loader = TextLoader(md_file, encoding='utf-8') # Using TextLoader for simplicity as Unstructured requires more deps
            documents.extend(loader.load())
            print(f"Loaded {md_file}")
        except Exception as e:
            print(f"Error loading {md_file}: {e}")
            
    return documents

def get_retriever():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Check if DB exists
    if os.path.exists(CHROMA_DB_DIR):
        print("Loading existing Chroma vector database...")
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    else:
        print("Creating new Chroma vector database from knowledge_base...")
        documents = load_documents()
        
        if not documents:
            print("Warning: No documents found in knowledge_base/. Creating an empty vector store.")
            # Create a dummy doc to initialize the DB if empty
            from langchain_core.documents import Document
            documents = [Document(page_content="This is a mock interview knowledge base.", metadata={"source": "system"})]
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=CHROMA_DB_DIR
        )
        print(f"Persisted {len(splits)} chunks to database.")
        
    # Return a retriever
    return vectorstore.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    # Test RAG script
    retriever = get_retriever()
    print("Retriever initialized successfully.")
