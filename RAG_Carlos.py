import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

st.set_page_config(page_title="Math RAG Pro", page_icon="📐", layout="wide")
st.title("📐 Solucionador Avanzado de Sistemas Lineales")

with st.sidebar:
    st.header("🔑 Seguridad")
    groq_key = st.text_input("Groq API Key:", type="password")
    hf_token = st.text_input("Hugging Face Token:", type="password")
    st.divider()
    archivo = st.file_uploader("Sube el PDF con ecuaciones", type="pdf")

if archivo and groq_key and hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    
    with open("temp.pdf", "wb") as f:
        f.write(archivo.getbuffer())

    @st.cache_resource
    def procesar_pdf_mejorado(ruta):
        loader = PyMuPDFLoader(ruta)
        docs = loader.load()
        
        # Reducimos el overlap y ajustamos el tamaño para no romper ecuaciones
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        # Aumentamos k a 6 para captar mejor el entorno de la ecuación
        return vectorstore.as_retriever(search_kwargs={"k": 6})

    retriever = procesar_pdf_mejorado("temp.pdf")
    
    if pregunta := st.chat_input("Escribe el sistema o la página donde está..."):
        st.chat_message("user").write(pregunta)
        
        # Recuperación
        docs = retriever.invoke(pregunta)
        contexto_sucio = "\n---\n".join([d.page_content for d in docs])

        # Prompt ultra-específico para "leer" matemáticas mal formateadas
        template = ChatPromptTemplate.from_messages([
            ("system", """Eres un experto en álgebra lineal capaz de leer texto extraído de PDFs que puede estar mal formateado.
            
            TAREA:
            1. Analiza el contexto y busca patrones numéricos que parezcan sistemas de ecuaciones (ej. variables x, y, z o coeficientes alineados).
            2. Reconstruye las ecuaciones correctamente.
            3. Resuelve el sistema paso a paso usando el método de Gauss o sustitución.
            4. Si los datos están incompletos debido al formato del PDF, indica qué parte falta.
            
            CONTEXTO DEL PDF:
            {context}"""),
            ("human", "{input}")
        ])

        llm = ChatGroq(groq_api_key=groq_key, model="llama-3.3-70b-versatile", temperature=0)
        chain = template | llm | StrOutputParser()
        
        with st.chat_message("assistant"):
            response = chain.invoke({"context": contexto_sucio, "input": pregunta})
            st.markdown(response)
            with st.expander("Ver texto extraído (Contexto Real)"):
                st.text(contexto_sucio)
else:
    st.info("Introduce las llaves y el PDF para empezar.")
