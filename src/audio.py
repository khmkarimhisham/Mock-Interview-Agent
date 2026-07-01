import os
import sys
import tempfile
import asyncio
import speech_recognition as sr
from faster_whisper import WhisperModel
import edge_tts
from contextlib import contextmanager

# Hide pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

@contextmanager
def suppress_stderr():
    """Context manager to suppress C-level stderr output (like ALSA warnings)."""
    null_fd = os.open(os.devnull, os.O_RDWR)
    old_stderr_fd = os.dup(sys.stderr.fileno())
    try:
        os.dup2(null_fd, sys.stderr.fileno())
        yield
    finally:
        os.dup2(old_stderr_fd, sys.stderr.fileno())
        os.close(old_stderr_fd)
        os.close(null_fd)

# Initialize Whisper Model (running on CPU to avoid CUDA library setup issues)
print("Loading Whisper model...")
# Using 'base' model as it is fast and fairly robust.
with suppress_stderr():
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper model loaded.")

def listen_and_transcribe():
    """Listens to the microphone and transcribes the speech to text."""
    recognizer = sr.Recognizer()
    # Wait for 3 seconds of silence before considering the phrase complete
    recognizer.pause_threshold = 3.0
    
    # Adjust for ambient noise and listen (suppressing ALSA logs)
    with suppress_stderr():
        source = sr.Microphone()
        
        with source:
            sys.stdout.write("\n[Adjusting for ambient noise...]\n")
            sys.stdout.flush()
            recognizer.adjust_for_ambient_noise(source, duration=1)
            sys.stdout.write("[Listening... Speak now!]\n")
            sys.stdout.flush()
            
            try:
                # Listen until silence is detected
                audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=60)
                sys.stdout.write("[Processing audio...]\n")
                sys.stdout.flush()
            except sr.WaitTimeoutError:
                return ""
            
    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(audio_data.get_wav_data())
        temp_audio_path = temp_audio.name
        
    try:
        # Transcribe using faster-whisper
        with suppress_stderr():
            segments, info = whisper_model.transcribe(temp_audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        return text
    finally:
        # Clean up temp file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

async def _synthesize_and_play(text, voice="en-US-ChristopherNeural"):
    """Async helper to synthesize speech using edge-tts and play it."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
        
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_audio_path)
        
        # Play the audio using pygame (suppressing logs)
        with suppress_stderr():
            pygame.mixer.init()
            pygame.mixer.music.load(temp_audio_path)
            pygame.mixer.music.play()
            
            # Wait until the audio finishes playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            pygame.mixer.quit()
    finally:
        # Clean up temp file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def speak(text, voice="en-US-ChristopherNeural"):
    """Synchronous wrapper to speak text out loud."""
    # print the text as well for the user to read
    print(f"\n[Agent]: {text}")
    asyncio.run(_synthesize_and_play(text, voice))

if __name__ == "__main__":
    # Test Audio script
    speak("Hello! I am your AI mock interviewer. How are you doing today?")
    print("Now, say something...")
    text = listen_and_transcribe()
    print(f"You said: {text}")
    if text:
        speak(f"I heard you say: {text}")
