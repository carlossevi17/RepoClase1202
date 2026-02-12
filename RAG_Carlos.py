import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Configuración de la interfaz
st.set_page_config(page_title="SafeBank AI Reader", page_icon="📖")
st.title("📖 Analizador de Manuales SafeBank")

# Barra lateral para configuración
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu Groq API Key:", type="password")
    modelo = st.selectbox("Modelo", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
    archivo = st.file_uploader("Sube el PDF del manual", type="pdf")

# Inicialización del sistema RAG
if archivo and api_key:
    # Guardar el PDF temporalmente para que el Loader pueda leerlo
    with open("temp_manual.pdf", "wb") as f:
        f.write(archivo.getbuffer())

    # Procesamiento del documento
    @st.cache_resource # Esto evita que se procese el PDF cada vez que haces una pregunta
    def procesar_pdf(ruta):
        loader = PyMuPDFLoader(ruta)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    retriever = procesar_pdf("temp_manual.pdf")
    st.success("✅ Manual analizado y listo para preguntas.")

    # Interfaz de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if pregunta := st.chat_input("¿Qué quieres saber del manual?"):
        st.session_state.messages.append({"role": "user", "content": pregunta})
        st.chat_message("user").write(pregunta)

        # Búsqueda y Generación
        contexto_docs = retriever.invoke(pregunta)
        contexto_texto = "\n\n".join([doc.page_content for doc in contexto_docs])
        
        # Prompt directo (más estable que las cadenas pre-hechas)
        template = ChatPromptTemplate.from_messages([
            ("system", "Responde basándote solo en este contexto:\n\n{context}"),
            ("human", "{input}")
        ])
        
        llm = ChatGroq(groq_api_key=api_key, model=modelo, temperature=0.3)
        chain = template | llm | StrOutputParser()
        
        respuesta = chain.invoke({"context": contexto_texto, "input": pregunta})
        
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        st.chat_message("assistant").write(respuesta)
else:
    st.info("Por favor, introduce tu API Key y sube un archivo PDF para comenzar.")
