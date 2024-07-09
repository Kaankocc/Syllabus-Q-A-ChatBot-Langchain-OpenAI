import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing_extensions import Concatenate
from langchain_community.vectorstores import Chroma

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.llms import OpenAI


load_dotenv()

st.set_page_config(page_title="Syllabus Q&A Bot", page_icon="📚")

st.title("Syllabus Q&A Bot📚")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader for syllabus PDF
uploaded_file = st.file_uploader("Upload your course syllabus (PDF)", type="pdf")

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


def combine_docs(similar_docs):
    # Initialize an empty list to store the content of each document
    document_contents = []

    # Iterate through similar_docs and collect the page_content
    for doc in similar_docs:
        document_contents.append(doc.page_content)

    # Join all the collected content with newline characters
    combined_answer = "\n".join(document_contents)
    return combined_answer



def get_response(query, context, chat_history):
    template = """
    You are a helpful assistant. Answer the following question considering both the history of the conversation and the piece of context. If you don't know the answer, just say that you don't know, don't try to make up an answer.:

    Chat history: {chat_history}
        
    Context: {context}

    User question: {user_question}
    """
        
    prompt = ChatPromptTemplate.from_template(template)
        
    llm = ChatOpenAI()
        
    chain = prompt | llm | StrOutputParser()
        
    return chain.stream({
            "chat_history": chat_history,
            "user_question": query,
            "context": context
     })

if uploaded_file is not None:
    
     # Save the uploaded file temporarily (optional, depending on library requirements)
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize PyPDFLoader with the saved file path
    try:
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        # Initialize text splitter
         # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        # Split the document into chunks
        splits = text_splitter.split_documents(docs)


    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        # Clean up: remove temporary file after processing
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
    
    embedding = OpenAIEmbeddings()
    
    # Create the Vector store
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding
    )
    
    # Hide file uploader after upload
    st.markdown("---")
    
     # Bot introduction
    with st.chat_message("AI"):
        st.write("Hi, I am your helpful assistant and I am here to answer your questions about your course syllabus.")
    

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)
                


    # user input 
    user_query = st.chat_input("Your message")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            similar_docs = vectordb.max_marginal_relevance_search(user_query, k=3, fetch_k=3)
            pretty_print_docs(similar_docs)
            combined_answer = combine_docs(similar_docs)
            ai_response = st.write_stream(get_response(user_query, combined_answer, st.session_state.chat_history))
            
        st.session_state.chat_history.append(AIMessage(ai_response))