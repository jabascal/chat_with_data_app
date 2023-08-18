""" Based on LangChain course "Chat with Your Data "
    
    See:
        https://www.deeplearning.ai/short-courses/
        https://python.langchain.com/

""" 
import os

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain.vectorstores import Chroma, DocArrayInMemorySearch


# Load document with langchain.document_loaders
def read_doc(doc_type, doc_path, mode_print=False):
    if doc_type == "pdf":
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(doc_path)
    elif doc_type == "url":
        from langchain.document_loaders import WebBaseLoader
        url = doc_path
        loader = WebBaseLoader(url)
    elif doc_type == "youtube":
        from langchain.document_loaders.blob_loaders.youtube_audio import \
            YoutubeAudioLoader
        from langchain.document_loaders.generic import GenericLoader
        from langchain.document_loaders.parsers import OpenAIWhisperParser

        save_path = "./downloads/youtube/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Delete all files in the directory to avoid confussion    
        files = os.listdir(save_path)
        for file in files:
            os.remove(file) 
        url = doc_path
        loader = GenericLoader(YoutubeAudioLoader([url], save_path), 
                               OpenAIWhisperParser())
    docs = loader.load()
    if mode_print is True:
        print(f"Loaded {len(docs)} pages/documents")
        print(f"First page: {docs[0].metadata}")
        print(docs[0].page_content[:500])
    return docs

def read_split_doc(doc_type, doc_path, chunk_size, chunk_overlap, mode_print=False): 
    # Read doc using langchain
    docs = read_doc(doc_type, doc_path)
    # -------------------------
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap)

    docs_split = text_splitter.split_documents(docs)
    if mode_print is True:
        print(f"Split into {len(docs_split)} chunks")
        print(f"First chunk: {docs_split[0].metadata}")
        print(docs_split[0].page_content)
    return docs_split

# --------------------------
def load_db_chroma_from_disk(embedding, persist_path):
    # Load db from disk
    db = Chroma(persist_directory=persist_path,
                               embedding_function=embedding)
    print(f"Loaded vector database with {db._collection.count()} documents")
    return db

def create_db_from_documents(docs_split, embedding):
    # Create vector database from document    
    db = DocArrayInMemorySearch.from_documents(
        docs_split, 
        embedding=embedding)
    #print(f"Created vector database with {db._collection.count()} documents")
    #db.persist() # save the vectorstore to disk for future use     
    return db

def load_db(file, chain_type, k, llm_name="gpt-3.5-turbo"):
    # Load documents, split them into chunks, and create a vector database. 
    # Then, return a chatbot chain that uses the vector database as a retriever.
    
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    # define embedding
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=openai.api_key), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,        
    )
    return qa 

# --------------------------
def init_embedding(openai_api_key):
    # Define embedding
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)    
    return embedding

def init_llm(llm_name, temperature, openai_api_key=openai.api_key):
    # Init the LLM
    llm = ChatOpenAI(model_name=llm_name,
                    temperature=temperature,
                    openai_api_key=openai_api_key)
    return llm

def init_qa_chain_from_llm(llm, db):
    # Init memory and chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=db.as_retriever(),
        memory=memory
    )
    return qa_chain, memory

def init_llm_qa_chain(llm_name, temperature, openai_api_key, db):
    # Init LLM and QA chain so it can be modified from the chat interface

    # Init the LLM
    llm = init_llm(llm_name, temperature, openai_api_key)
    # Init memory and chain
    qa_chain, memory = init_qa_chain_from_llm(llm, db)
    return llm, qa_chain, memory

# Call QA chain
def qa_call(input):
    output = qa_chain({"question": input})
    return output
def qa_answer(input):
    # Return the answer from the QA call
    return qa_call(input)['answer']
def qa_input_msg_history(input, history):
    # QA function that inputs the answer and the history
    # History managed internally by ChatInterface
    answer = qa_answer(input)
    return answer

