import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.document_loaders import WebBaseLoader
import pandas as pd

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_web_text(urls):
    text = ""
    for url in urls:
        loader = WebBaseLoader(url)
        documents = loader.load()
        for doc in documents:
            text += doc.page_content
    return text

def get_csv_text(csv_files):
    text = ""
    for csv in csv_files:
        df = pd.read_csv(csv)
        text += df.to_string(index=False)
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=60,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your data", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your documents :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        
        # Upload PDFs
        pdf_docs = st.file_uploader(
            "Upload your PDFs here", accept_multiple_files=True, type=["pdf"])
        
        # Upload CSVs
        csv_files = st.file_uploader(
            "Upload your CSVs here", accept_multiple_files=True, type=["csv"])

        # Enter URLs
        urls = st.text_area("Enter URLs here (one per line)").splitlines()

        if st.button("Process"):
            with st.spinner("Processing documents"):
                raw_text = ""

                # Process PDFs
                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)
                
                # Process CSVs
                if csv_files:
                    raw_text += get_csv_text(csv_files)
                
                # Process URLs
                if urls:
                    raw_text += get_web_text(urls)
                
                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
