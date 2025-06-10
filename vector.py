#Importing all relavant data
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

#Bring in the csv file 
df = pd.read_csv("realistic_restaurant_reviews.csv")

#Define any embedding model from Ollama
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

#Check whether this location already exists
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

#If location doesn't exists we are going to prepare data by adding documents and ids
if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " +row["Review"],
            metadata = {"rating":row["Rating"], "date":row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

#Initializing vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

#If location doesn't exists this is required, because this will atomatically embed documents add it to vector store
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

#Allows to grab the relavant document
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
