import os
import sys
import argparse
from src.audio import speak, listen_and_transcribe
from src.agent import generate_response, save_transcript

def main():
    print("==============================================")
    print("    AI Mock Interview Agent (English Practice)  ")
    print("==============================================\n")
    
    # Initial greeting
    greeting = "Hello! I am your AI Mock Interviewer. We are going to practice your English while doing a technical interview. Are you ready to begin?"
    speak(greeting)
    
    try:
        while True:
            # 1. Listen for user input
            user_input = listen_and_transcribe()
            if not user_input:
                continue
                
            print(f"\n[You]: {user_input}")
            
            # Check for exit commands
            if user_input.lower().strip('.!?,') in ['quit', 'exit', 'stop', 'goodbye', 'bye']:
                farewell = "Thank you for the interview practice. I have saved our transcript. Goodbye and good luck!"
                speak(farewell)
                break
                
            # 2. Agent thinks (LLM + RAG)
            print("\n[Agent is thinking...]")
            response = generate_response(user_input)
            
            # 3. Agent speaks
            speak(response)
            
    except KeyboardInterrupt:
        print("\nInterview interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Always save transcript on exit
        save_transcript()

if __name__ == "__main__":
    # Ensure working directory is correct
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
