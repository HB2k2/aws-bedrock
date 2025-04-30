import json
import os
import sys
import boto3
import streamlit as st

# We will be using Titan Embedddings Model to generate Embeddings

from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## Vector Embeddings And Vector Store

from langchain_community.vectorstores import FAISS

## LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients

bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


## Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # in our testing character split works better with this pdf data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Emmbedding and Vector Store

def get_vector_store(docs):
    # Create the vector store using FAISS and the Bedrock embeddings
    vector_store = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store.save_local("faiss_index")

def get_claude_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="",
                client=bedrock,
                model_kwargs={'maxTokens':512})
    
    return llm

def get_llama_llm():
    llm = Bedrock(model_id = "",
                  client =bedrock,
                  model_kwargs={'max_gen_len':512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a
concise answer to the question at the end but use atleast summize with 
250 words with detailed explanation. If you don't know the answer, 
just say that you don't know. Don't try to make up an answer.
<context>
{context}</context>
Question: {question}
Assistant: """


PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"],
)

def get_response_llm(llm,vector_store, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    answer = qa({"query": query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with your PDF using AWS Bedrock")

    user_question = st.text_input("Ask a question about the PDF:")  

    with st.sidebar:
        st.title("Update or Create Vector Store")

        if st.button("Vectors Update"):
            with st.spinner("Updating Vectors..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vectors Updated Successfully!")

    if st.button("Clude Output"):
        with st.spinner("Loading..."):
            vector_store = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm = get_claude_llm()
            answer = get_response_llm(llm, vector_store, user_question)
            st.write(answer)
            st.success("Answer Generated Successfully!")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_llama_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()