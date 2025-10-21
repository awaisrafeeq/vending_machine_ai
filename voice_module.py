from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
import numpy as np
import wave
import io
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models import Product  # Assuming this is defined in models.py
from database import DATABASE_URL  # Assuming this is set up in database.py

# Load environment variables from .env file
load_dotenv()
# Get the API key
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def speech_to_text(file_path):
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        )
    print(f"DEBUG: {response.text}")
    return response.text

def text_to_speech(text):
    # Generate speech
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    # Save the audio
    audio_path = "output.mp3"
    with open(audio_path, "wb") as f:
        f.write(response.content)
    # Convert and play the audio
    sound = AudioSegment.from_file(audio_path, format="mp3")
    play(sound)

if os.getenv("RENDER") != "true":
    import sounddevice as sd
else:
    sd = None  # or mock functions if needed

def record_audio(filename="user_audio.wav", duration=5, samplerate=44100):
    print("Recording... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    # Save the recording to a file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    print(f"Audio saved as {filename}")
    return filename  # Return the saved file path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a new database connection for each request."""
    try:
        engine = create_engine(DATABASE_URL)
        session = Session(engine)
        return session
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def get_available_categories() -> list:
    """Fetch all unique categories from the products table."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT DISTINCT category FROM products WHERE category IS NOT NULL")
        categories = cursor.fetchall()
        available_categories = [cat[0] for cat in categories if cat[0]]
        return available_categories
    except Exception as e:
        logger.error(f"Database error in get_available_categories: {e}")
        return []
    finally:
        conn.close()

def analyze_text_with_llm(transcribed_text: str) -> dict:
    """
    Send the transcribed text to LLM with a vending machine AI prompt,
    get description and category recommendation as text (no TTS).
    """

    available_categories = get_available_categories()
    if not available_categories:
        available_categories = ['protein', 'fat', 'junk']  # fallback

    system_prompt = (
        "You are a vending machine AI designed to engage with teens in a fun, interactive, and humorous way. "
        "Your personality is witty, slightly sassy, and full of science-backed facts about snacks and drinks. "
        "Analyze this input text and provide: "
        "1. A brief, fun description (1-2 sentences) "
        "2. Based on the text, recommend ONE category from this list: "
        f"{', '.join(available_categories)} "
        "Format your response EXACTLY as: "
        "[DESCRIPTION]: Your fun description here "
        "[CATEGORY]: single_category_name "
        "Keep it engaging and trendy!"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcribed_text}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300
        )

        content = response.choices[0].message.content
        description = ""
        category = ""

        for line in content.split('\n'):
            if '[DESCRIPTION]:' in line:
                description = line.split('[DESCRIPTION]:')[1].strip()
            elif '[CATEGORY]:' in line:
                category = line.split('[CATEGORY]:')[1].strip()

        # Fallback parsing if structured format isn't used
        if not description or not category:
            # Try to extract category from available categories
            for available_cat in available_categories:
                if available_cat.lower() in content.lower():
                    category = available_cat
                    break
            
            # If still no category found, use the first available category
            if not category:
                category = available_categories[0]
            
            # Clean up description
            description = content.replace(f"[CATEGORY]: {category}", "").replace(f"[DESCRIPTION]: ", "").strip()
            if not description:
                description = "Looking good! Ready for some snacks?"
        
        return {"description": description, "category": category}
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")