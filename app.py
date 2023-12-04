import pickle
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
load_dotenv()


# Sidebar
with st.sidebar:
    st.title("🤗 💬 LLM chat")

def main():
    st.header("Chat with PDF 💭")

    # Upload a PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # get the Data form the PDF
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        st.write(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)

        # Split the text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = splitter.split_text(text)
        # st.write(chunks)

        # build embeddings space using the chunks.
        # if file exsist 
        filename = uploaded_file.name[:-4]
        if os.path.exists(filename + ".pkl"):
            with open(filename + ".pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write("loaded vector store from file...", VectorStore)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embeddings)
            with open(filename + ".pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write("computed and saved vector store to file...", VectorStore)

        # accept user questions / queries
        query = st.text_input("Ask a question about the PDF")
        if query:
            # # get the vector of the query
            most_similar_text = VectorStore.similarity_search(query, k=3)
            # send texts to llm with query.
            llm = OpenAI(temperature=0 , model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm = llm , chain_type="stuff")
            with get_openai_callback() as  cb:
                response = chain.run(question=query, input_documents=most_similar_text)
                print(cb)
            st.write("ChatPDF: "+response)



if __name__=="__main__":
    main() 