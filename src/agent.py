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

SYSTEM_PROMPT = """You are a friendly, conversational Senior AI Engineering Interviewer conducting a mock voice interview. 
You must sound like a natural human speaking. Do NOT use labels like "English Feedback:", "Response:", or "Agent:". Do NOT use markdown formatting like **bold** because this text will be read aloud by a text-to-speech engine.

The candidate is a non-native English speaker. Follow these rules for every turn:
1. ONLY if they make a noticeable grammar or vocabulary mistake, gently weave a very quick correction into your natural spoken response (e.g., "By the way, a more natural way to say that is..."). If their English was fine, do not mention it.
2. Evaluate their technical answer conversationally, then smoothly transition into a follow-up or a new question.
3. Keep your responses relatively brief, just like a real spoken conversation. Don't give long monologues.
4. If they ask you to repeat or clarify, do so simply and patiently.

Use the following knowledge base topics to guide your questions:
{context}
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
