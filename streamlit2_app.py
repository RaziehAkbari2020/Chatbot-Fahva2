import os.path
import pickle
import sqlite3
import datetime

import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS

MAX_FILE_SIZE = 10 * 1024 * 1024  # حجم حداکثری برای فایل PDF
DB_NAME = 'chat_history.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                 (username TEXT, message TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

def save_chat_history(username, chat_history):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    for entry in chat_history:
        c.execute("INSERT INTO chat_history (username, message, timestamp) VALUES (?, ?, ?)",
                  (username, entry, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_chat_history(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT message, timestamp FROM chat_history WHERE username = ?", (username,))
    history = c.fetchall()
    conn.close()
    return history

with st.sidebar:
    st.title('Build your own ChatBot')

def main():
    st.header('Upload your Source')
    pdf = st.file_uploader('Upload your PDF', type=['pdf'])

    # افزودن امکان ورود به نشست
    username = st.text_input('Enter your name (optional):')
    if username:
        st.write(f'Hello, {username}!')
    else:
        username = 'anonymous'

    chat_history = []  # لیست برای نگهداری تاریخچه چت

    if pdf is not None:
        if pdf.size > MAX_FILE_SIZE:
            st.error('File size exceeds the maximum limit of 10 MB.')
            return

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text=text)
        
        storename = pdf.name[:-4]
        if os.path.exists(f'{storename}.pkl'):
            with open(f'{storename}.pkl', "rb") as f:
                Vectorstore = pickle.load(f)
        else:
            embedding = OpenAIEmbeddings(openai_api_key="YOUR_API_KEY")
            Vectorstore = FAISS.from_texts(chunks, embedding=embedding)
            with open(f'{storename}.pkl', "wb") as f:
                pickle.dump(Vectorstore, f)

    query = st.text_input('Your query here')

    if query:
        chat_history.append("پرسش: " + query)
        docs = Vectorstore.similarity_search(query=query, k=3)
        llm = OpenAI(openai_api_key="YOUR_API_KEY")
        chain = load_qa_chain(llm=llm, chain_type='stuff')
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.markdown(response)
        chat_history.append("پاسخ: " + response)

        save_chat_history(username, chat_history)

    # افزودن گزینه برای نمایش تاریخچه چت
    if st.button('Show Chat History'):
        history = get_chat_history(username)
        for message, timestamp in history:
            st.text(f"{timestamp}: {message}")

if __name__ == '__main__':
    init_db()
    main()

