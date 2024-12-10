import os
import time
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModel
from groq import Groq

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate


#APIs
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#SPECS
PINECONE_CLOUD = os.environ.get('PINECONE_CLOUD')
PINECONE_REGION = os.environ.get('PINECONE_REGION')

# Embeddings
def embedder():
    """
    Esta función carga el modelo de embeddings y lo devuelve.

    Returns:
    - embedding_model: El modelo de embeddings de lenguaje natural
    """

    embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    return embedding_model

# Connections
def pinecone_connection():
    """
    Esta función se conecta a Pinecone y devuelve el objeto de conexión.
    
    Returns:
    - pc: El objeto de conexión a Pinecone
    """

    # Se abre la conexión con Pinecone
    print("Connecting to Pinecone...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except:
        print("Error connecting to Pinecone")
        return None    
    return pc

def groq_connection():
    """
    Esta función se conecta a Groq y devuelve el objeto de conexión.
    
    Returns:
    - groq_client: El objeto de conexión a Groq
    """

    print("Connecting to Groq...")
    try:
        groq_client = Groq(
            # GROQ_API_KEY is the default and can be omitted if in ENV variables
            api_key=GROQ_API_KEY,
        )
    except:
        print("Error connecting to Groq")
        return None
    return groq_client


# Pinecone
def pinecone_search(user_query, embedding_model, pinecone_index, pinecone_connection):
    """
    Esta función busca en el índice de Pinecone el vector más cercano a la consulta del usuario.
    
    Args:
    - user_query: La consulta del usuario
    - embedding_model: El modelo de embeddings de lenguaje natural
    - pinecone_index: El nombre del índice de Pinecone
    - pinecone_connection: El objeto de conexión a Pinecone
    
    Returns:
    - result: El resultado de la búsqueda en Pinecone
    """

    # Obtener el embedding de la consulta
    user_query_embedding = embedding_model.encode([user_query]).tolist() # Necesario convertirlo a una lista

    # Se conecta al índice
    pinecone_index = pinecone_connection.Index(pinecone_index)
    time.sleep(1) # Esperar un segundo para que se cargue el índice

    # Buscar el vector más cercano usando Pinecone
    result = pinecone_index.query(
        namespace="default",
        vector=user_query_embedding, 
        top_k=1,
        include_metadata=True,
        include_values=False,
        )
    
    return result


# Prepare System Prompt
def get_system_prompt(result):
    """
    Esta función prepara el mensaje del sistema para el usuario, utilizando la información de la búsqueda en Pinecone.
    
    Args:
    - result: El resultado de la búsqueda en Pinecone
    
    Returns:
    - sys_prompt: El mensaje del sistema para el usuario
    """

    matched_info = ' '.join(item['metadata']['text'] for item in result['matches'])
    context = f"Information: {matched_info}"
    sys_prompt = f"""
    Instructions:
    - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
    - Utilize the context provided for accurate and specific information.
    Context: {context}
    """
    return sys_prompt


def sent_query_to_groq(sys_prompt, user_query, groq_client):
    """
    Esta función envía la consulta del usuario a Groq y devuelve la respuesta.
    
    Args:
    - sys_prompt: El mensaje del sistema para el usuario
    - user_query: La consulta del usuario
    - groq_client: El objeto de conexión a Groq
    
    Returns:
    - response: La respuesta del chatbot
    """

    # Define the query
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": user_query,
            }
        ],
        model="llama3-8b-8192",
    )
    # Get the response
    response = chat_completion.choices[0].message.content
    
    return response


def main():
    """
    Esta función es el punto de entrada principal de la aplicación. Configura el cliente de Groq, 
    la interfaz de Streamlit y maneja la interacción del chat.
    """

    print("Iniciando la aplicación...")

    # El título y mensaje de bienvenida de la aplicación Streamlit
    st.title("Chat CEIA de ejemplo")
    st.write("¡Hola! Este es un ejemplo de chatbot con memoria persistente gestionada programáticamente con Langchain, utilizando Groq")

    # Agregar opciones de personalización en la barra lateral
    st.sidebar.title('Personalización')
    conversational_memory_length = st.sidebar.slider('Longitud de la memoria conversacional:', 1, 10, value = 5)

    # Inicializar la memoria de la conversación
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="historial_chat", return_messages=True)

    user_question = st.text_input("Haz una pregunta:")

    # Variable de estado de la sesión
    if 'historial_chat' not in st.session_state:
        st.session_state.historial_chat=[]
    else:
        for message in st.session_state.historial_chat:
            memory.save_context(
                {'input': message['humano']},
                {'output': message['IA']}
            )

    groq_chat = groq_connection()
    embedding_model = embedder()

    # Si el usuario ha hecho una pregunta,
    if user_question:

        pinecone_index = "cv-index"

        # Buscar en Pinecone
        result = pinecone_search(user_question, embedding_model, pinecone_index, pinecone_connection())
        sys_prompt = get_system_prompt(result)
        response = sent_query_to_groq(sys_prompt, user_question, groq_chat)

        message = {'humano': user_question, 'IA': response}
        st.session_state.historial_chat.append(message)
        st.write("Chatbot:", response)


if __name__ == "__main__":
    main()