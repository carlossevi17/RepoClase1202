import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

## Función de generación dinámica con la API Key del usuario
def buscar_jugador(api_key, pais, equipo, detalles):
    # Inicializamos el modelo dentro de la función usando la key del input
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=api_key
    )

    template = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en historia del baloncesto y scouting de la NBA y ligas internacionales."),
        ("human", "Dime un jugador de baloncesto destacado que sea de {pais} y juegue (o haya jugado) en el equipo {equipo}. {detalles}"),
    ])

    chain = template | llm | StrOutputParser()
    res = chain.invoke({"pais": pais, "equipo": equipo, "detalles": detalles})
    return res

# Configuración de la interfaz
st.set_page_config(page_title="Basket Scout 🏀", page_icon="🏀")
st.title("Buscador de Jugadores 🏀")

# --- SECCIÓN DE API KEY ---
api_key = st.sidebar.text_input("Introduce tu Groq API Key:", type="password")
st.sidebar.caption("Consigue tu llave en [console.groq.com](https://console.groq.com/keys)")

# Inputs de usuario
pais = st.selectbox("Selecciona el País:", 
                   ['España', 'Estados Unidos', 'Argentina', 'Francia', 'Serbia', 'Eslovenia', 'Grecia', 'Canadá', 'Lituania', 'Brasil'])

equipo = st.text_input("Equipo (NBA o Europa):", placeholder="Ej: Los Angeles Lakers, Real Madrid, Golden State Warriors...")

modo = st.radio("Preferencia:", ["Jugador Actual", "Leyenda Histórica", "Cualquiera"])

if st.button("¡Buscar Jugador!"):
    if not api_key:
        st.error("Por favor, introduce tu API Key de Groq en la barra lateral.")
    elif not equipo:
        st.warning("Por favor, escribe el nombre de un equipo.")
    else:
        detalles_extra = f"Prefiero un jugador que sea: {modo}."
        
        try:
            with st.spinner('Consultando la base de datos de la liga...'):
                respuesta = buscar_jugador(api_key, pais, equipo, detalles_extra)
                st.subheader(f"Resultado para {equipo}:")
                st.markdown(respuesta)
        except Exception as e:
            st.error(f"Error de autenticación o de sistema: {e}")
