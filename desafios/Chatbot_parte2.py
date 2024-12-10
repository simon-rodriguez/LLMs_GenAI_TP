import os
import time
import re
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

#PROMPTS
INITIAL_PROMPT = f"""
Instructions: You run in a cycle of THINK, ACT, WAIT, and RESULT. At the end of the cycle, you give an ANSWER.

THINK: Read the user query and think about an action to answer it.
ACT: Reply with calling an Action to perform any of the actions available to you. This should be formatted as ACT: call "action" for query "user query".
WAIT: Wait until you receive the next prompt with the result of your action.
RESULT: you will recieve the result of your action in the next prompt. Use this to create an answer for the user query. DO NOT mention this to the user.
ANSWER: finally, answer the user query directly. DO NOT write "ANSWER:" before your answer.

Available actions:
- cv_simon: use it if you want to ask about Simon's CV or if you do not know the name of the person in the query.
- cv_jessica: use it if you want to ask about Jessica's CV

DO NOT FORGET TO WAIT FOR THE RESULT BEFORE ANSWERING THE USER QUERY. DO NOT SKIP ANY STEP.
"""



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
def get_system_prompt(user_query, index_name):
    """
    Esta función prepara el mensaje del sistema para el usuario, con la información relevante de la búsqueda.

    Args:
    - user_query: La consulta del usuario
    - index_name: El nombre del índice de Pinecone

    Returns:
    - sys_prompt: El mensaje para el LLM con el contexto de la búsqueda
    """

    result = pinecone_search(user_query, embedding_model=embedder(), pinecone_index=index_name, pinecone_connection=pinecone_connection())


    matched_info = ' '.join(item['metadata']['text'] for item in result['matches'])
    context = f"Information: {matched_info}"
    sys_prompt = f"""
    Instructions:
    - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
    - Utilize the context provided for accurate and specific information.
    Context: {context}
    """
    return sys_prompt


# Parte 2: Agentes

# Se crea una clase Agente para interactuar con el sistema, esta es la estructura del agente
class Agent:
    def __init__(self, sys_prompt=""):
        self.system = sys_prompt
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": sys_prompt})

    def __call__(self, user_query):
        self.messages.append({"role": "user", "content": user_query})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = groq_connection().chat.completions.create(
                        model="llama3-8b-8192", 
                        messages=self.messages)
        return completion.choices[0].message.content

def agente_simon(user_query):
    sys_prompt = get_system_prompt(user_query, "simon-index")
    agent = Agent(sys_prompt)
    result = agent(user_query)
    return result

def agente_jess(user_query):
    sys_prompt = get_system_prompt(user_query, "jess-index")
    agent = Agent(sys_prompt)
    result = agent(user_query)
    return result

def query(user_query):
    bot = Agent(INITIAL_PROMPT)
    first_prompt = user_query
    action_prompt = bot(first_prompt)
    print(action_prompt)
    acciones = [
        action_re.search(a)
        for a in action_prompt.split('\n') 
        if action_re.search(a)
    ]
    if acciones:
        # There is an action to run
        accion, accion_input = acciones[0].groups()
        print("Acción detectada:", accion)
        if accion not in acciones_disponibles:
            raise Exception("Acción desconocida: {}: {}".format(accion, accion_input))
        print(" -- corriendo {} {}".format(accion, accion_input))
        action_result = acciones_disponibles[accion](accion_input)
        print("RESULT of Action:", action_result)
        response_prompt = "RESULT: {}".format(action_result)
        response = bot(response_prompt)
        return response
    else:
        print("No hay acciones detectadas")
        return

# VARIABLES
action_re = re.compile(r'^ACT: call "(\w+)" for query "(.*)"$', re.MULTILINE)  # expresión regular para capturar secuencias de texto

acciones_disponibles = {
    "cv_simon": agente_simon,
    "cv_jessica": agente_jess,
}        


def main():
    """
    Esta función es el punto de entrada principal de la aplicación. Configura el cliente de Groq, 
    la interfaz de Streamlit y maneja la interacción del chat.
    """

    print("Iniciando la aplicación...")

    # El título y mensaje de bienvenida de la aplicación Streamlit
    st.title("Chat de Consulta de CVs - CEIA")
    st.write("¡Hola! Este es un ejemplo de chatbot con agentes para responder preguntas sobre CVs, utilizando Groq y Pinecone")

    #TODO: Agregar opciones de personalización en la barra lateral
    #st.sidebar.title('Personalización')
    #conversational_memory_length = st.sidebar.slider('Longitud de la memoria conversacional:', 1, 10, value = 5)

    conversational_memory_length = 5
    
    # Inicializar la memoria de la conversación
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="historial_chat", return_messages=True)

    user_question = st.text_input("Haz una pregunta (especificando sobre quién quieres la información: Simon o Jessica):")

    # Variable de estado de la sesión
    if 'historial_chat' not in st.session_state:
        st.session_state.historial_chat=[]
    else:
        for message in st.session_state.historial_chat:
            memory.save_context(
                {'input': message['humano']},
                {'output': message['IA']}
            )

    # Si el usuario ha hecho una pregunta,
    if user_question:

        response = query(user_question)

        message = {'humano': user_question, 'IA': response}
        st.session_state.historial_chat.append(message)
        st.write("Chatbot:", response)


if __name__ == "__main__":
    main()