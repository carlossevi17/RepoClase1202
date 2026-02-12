import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Configuración de la interfaz con estilo matemático
st.set_page_config(page_title="Math Solver RAG", page_icon="🧮")
st.title("🧮 Solucionador de Sistemas de Ecuaciones")
st.markdown("Sube un PDF con sistemas de ecuaciones y yo los resolveré por ti.")

# Barra lateral para credenciales y configuración
with st.sidebar:
    st.header("🔑 Credenciales")
    groq_key = st.text_input("Groq API Key:", type="password")
    hf_token = st.text_input("Hugging Face Token:", type="password")
    
    st.divider()
    st.header("⚙️ Configuración")
    modelo = st.selectbox("Modelo LLM", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
    archivo = st.file_uploader("Sube el PDF de matemáticas", type="pdf")

# Validación de credenciales antes de procesar
if archivo and groq_key and hf_token:
    # Configurar el token en el entorno
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    with open("temp_math.pdf", "wb") as f:
        f.write(archivo.getbuffer())

    @st.cache_resource
    def procesar_pdf_matematico(ruta, _token):
        loader = PyMuPDFLoader(ruta)
        docs = loader.load()
        # Chunks más pequeños para no perder detalles de las ecuaciones
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        # Inicializar embeddings usando el token proporcionado
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    try:
        retriever = procesar_pdf_matematico("temp_math.pdf", hf_token)
        st.success("✅ PDF matemático procesado. ¡Listo para resolver!")
    except Exception as e:
        st.error(f"Error al procesar: {e}")

    # Historial de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if pregunta := st.chat_input("Ej: Resuelve el sistema de ecuaciones de la página 2"):
        st.session_state.messages.append({"role": "user", "content": pregunta})
        st.chat_message("user").write(pregunta)

        # RAG: Recuperación de contexto
        contexto_docs = retriever.invoke(pregunta)
        contexto_texto = "\n\n".join([doc.page_content for doc in contexto_docs])
        
        # Prompt especializado en Matemáticas
        template = ChatPromptTemplate.from_messages([
            ("system", """Eres un experto en álgebra lineal. Tu tarea es identificar y resolver sistemas de ecuaciones lineales extraídos del contexto proporcionado.
            
            REGLAS:
            1. Muestra el sistema de ecuaciones claramente.
            2. Resuelve paso a paso (puedes usar sustitución, igualación o reducción).
            3. Si el sistema no tiene solución o tiene infinitas, explícalo.
            4. Si no encuentras ecuaciones en el contexto, dilo amablemente.
            
            Contexto:
            {context}"""),
            ("human", "{input}")
        ])
        
        llm = ChatGroq(groq_api_key=groq_key, model=modelo, temperature=0.1)
        chain = template | llm | StrOutputParser()
        
        with st.spinner("Calculando solución..."):
            respuesta = chain.invoke({"context": contexto_texto, "input": pregunta})
        
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        st.chat_message("assistant").write(respuesta)
else:
    st.info("Configura las API Keys y sube el PDF matemático en la barra lateral para empezar.")
