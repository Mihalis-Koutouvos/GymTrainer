from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

#Loads our environment variables from the .env file
load_dotenv()

persistent_directory = "backend/data/chroma_db"

#Initializing the vector embeddings and the vector store
#Embedding
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

#Vector store
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"} #With RAG, best to use cosine similarity when comparing the query and document embeddings
)

#Cosine similarity assigns a score from 0 to 1 based on the angle between the query and document embeddings. 
#A score of 1 means the vectors are identical in direction (highly similar), while a score of 0 means they are
#orthogonal (not similar at all).
#Cosine similarity formula: (A * B) / (||A|| * ||B||)
#A * B is the dot product of the two vectors A and B
#||A|| and ||B|| are the magnitudes (lengths) of the vectors A and B
#Ex.) Say we got a cosine similarity score of 0.98: This means that the query 
#and the document are very similar in meaning, as their embeddings point in almost the same direction in the vector space.

#Key insight into modern embedding models:
#With modern embedding models, all vectors are normalized to have a magnitude of 1.0.
#This means that ||A|| and ||B|| are both equal to 1
#As a result, we simplify the consine similarity formula to A * B. 

#Begin searching for relevant documents by defining a query and then using the retriever
query = "How should I handle fatigue when lifting?"

#Defining our retriever: This searches for the top three chunks with the highest
#similarity scores compared to the query embedding. k represents the chunks
retriever = db.as_retriever(search_kwargs={"k": 5})

#Calling the retriever with our query to get the relevant documents (top 5 chunks)
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("----Context----")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i}:\n{doc.page_content}\n")


######LLM Portion

#Combine the query and the relevant document contents
#Query is at the end of combined_input, and chunks are used right below Documents
#chr is a new-line character

#This is essentially our system prompt for the LLM, and we are telling it to not use its own material if 
#the provided documents do not contain the answer to the question.
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

#Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

#Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

#Invoke the model with the combined input
result = model.invoke(messages)

#Display the full result and content only
print("\n ---Generated Response---")
print("Content only:")
print(result.content)