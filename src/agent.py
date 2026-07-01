import os
import datetime
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from src.rag import get_retriever

# Setup LLM
llm = Ollama(model="llama3:latest", temperature=0.7)

# Load RAG retriever
retriever = get_retriever()

SYSTEM_PROMPT = """You are a friendly but rigorous Senior AI Engineering Interviewer. 
Your goal is to conduct a mock interview with the user.

IMPORTANT: The user is a non-native English speaker practicing for interviews. 
For every turn, you MUST do two things:
1. Provide a brief, gentle correction on their English grammar or vocabulary if they made a mistake (if their English was perfect, encourage them).
2. Respond to their technical answer, or ask the next technical question. 
If they ask you to repeat or say they don't understand, rephrase your question more simply.

Always be concise. Speak clearly and use accessible English without overly complex idioms.
If relevant to the current topic, use the following knowledge base context to ground your questions or feedback:
{context}

Respond in the following format:
[English Feedback] (Your grammar feedback here)
[Response] (Your technical response/question here)
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create a chain
chain = prompt | llm

# Manage conversation history manually
chat_history = ChatMessageHistory()
transcript_lines = []

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(user_input: str) -> str:
    """Generates the agent's response given user input."""
    # Retrieve context
    docs = retriever.invoke(user_input)
    context = format_docs(docs)
    
    # Generate response
    response = chain.invoke({
        "input": user_input,
        "history": chat_history.messages,
        "context": context
    })
    
    # Update memory
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response)
    
    # Update transcript
    transcript_lines.append(f"**User**: {user_input}\n")
    transcript_lines.append(f"**Agent**: {response}\n\n")
    
    return response

def save_transcript():
    """Saves the conversation transcript to the transcripts/ directory."""
    if not transcript_lines:
        return
        
    os.makedirs("transcripts", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("transcripts", f"interview_{timestamp}.md")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Mock Interview Transcript\n\n")
        f.writelines(transcript_lines)
        
    print(f"\n[Transcript saved to {filepath}]")
    return filepath
