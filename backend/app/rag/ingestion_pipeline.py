import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="backend/data"):
    """Load all files from the data directory"""
    print(f"Loading documents from {docs_path}...")

    #Check if data directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your text files.")

    #Load all files form the data directory 
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    #Provides a list of langchain docs
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No files have been found in {docs_path}. Please add some documents.")
    
    for i, doc in enumerate(documents[:2]): #Show first 2 docs, create a doc per page (chunking system will break this down more)
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"Content Preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0): #800 characters
    """Split documents into smaller chunks with overlap"""
    print("Splitting docments into chunks...")

    #Most important part of this function
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]): #Show first 5 chunks
            print(f"\nChunk {i+1}:")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content: {chunk.page_content}")
            print("-"*50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks.")

    return chunks


def create_vector_store(chunks, persist_directory="backend/database/chroma_db"): #second is where we want the vector db locally
    """Create and store the embeddings in a ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    #Initialize the embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    #Create ChromaDB vector store
    #This function creates the vector embedding versions and then stores them in vector db
    print("Creating ChromaDB vector store:")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"} #Specifying the algorithm to use
    )

    print(f"Vector store created and saved to {persist_directory}.")

    return vectorstore

    

def main():
    
    #1. Loading source documents
    documents = load_documents(docs_path="backend/data")

    #2. Chunking the documents
    chunks = split_documents(documents)

    #3. Send the chunks to the embedding model and store the embeddings in a vector db
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()
