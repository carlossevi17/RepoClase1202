import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SafeBank AI Support", page_icon="🏦", layout="wide")

# Estilo personalizado para un toque original
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatFloatingInputContainer { background-color: #ffffff; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7d32, #1b5e20); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: Configuración y Carga ---
with st.sidebar:
    st.title("🏦 SafeBank AI Config")
    api_key = st.text_input("Introduce tu Groq API Key:", type="password")
    
    st.divider()
    model_id = st.selectbox("Modelo", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
    temp = st.slider("Creatividad (Temperature)", 0.0, 1.0, 0.3)
    
    uploaded_file = st.file_uploader("Sube el manual de SafeBank (PDF)", type="pdf")
    
    if st.button("🚀 Inicializar Conocimiento"):
        if not api_key or not uploaded_file:
            st.error("Por favor, introduce la API Key y sube el PDF.")
        else:
            with st.spinner("Procesando manual..."):
                # Guardar temporalmente
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 1. Cargar y Dividir
                loader = PyMuPDFLoader("temp.pdf")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                
                # 2. Embeddings y Vector Store
                embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
                vectorstore = FAISS.from_documents(chunks, embeddings)
                
                # Guardar en sesión
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                st.session_state.ready = True
                st.success("¡Base de datos lista!")

# --- LÓGICA RAG ---
def get_rag_chain():
    llm = ChatGroq(groq_api_key=api_key, model=model_id, temperature=temp)
    
    system_prompt = (
        "Eres el asistente oficial de SafeBank. Usa el contexto para responder "
        "de forma profesional y amable. Si no sabes la respuesta, sugiere contactar "
        "al soporte humano en support@safebank.com.\n\nContexto: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(st.session_state.retriever, question_answer_chain)

# --- INTERFAZ DE CHAT ---
st.title("💬 SafeBank Support Center")
st.caption("Asistente inteligente entrenado con el manual operativo v2026")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    if "ready" not in st.session_state:
        st.error("Primero inicializa el conocimiento en la barra lateral.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            rag_chain = get_rag_chain()
            response = rag_chain.invoke({"input": prompt})
            full_response = response["answer"]
            st.markdown(full_response)
            
            # Botón expandible para ver las fuentes (toque profesional)
            with st.expander("Ver fuentes consultadas"):
                for doc in response["context"]:
                    st.write(f"- {doc.page_content[:200]}...")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
