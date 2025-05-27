from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import tempfile
from typing import Dict, Any
import logging

from voice_module import speech_to_text, analyze_text_with_llm


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Vending Machine AI",
    description="AI-powered vending machine that analyzes user images and recommends products",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
api_key = os.getenv('OPENAI_API_KEY') or 'YOUR_OPENAI_API_KEY'  # Replace with your actual OpenAI API key
client = OpenAI(api_key=api_key)
client = OpenAI(api_key=api_key)

# Database connection function
def get_db_connection():
    """Create a new database connection for each request."""
    try:
        conn = sqlite3.connect('Newdb.db')
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def get_available_categories() -> list:
    """Fetch all unique categories from the products table."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM products WHERE category IS NOT NULL")
        categories = cursor.fetchall()
        available_categories = [cat[0] for cat in categories if cat[0]]
        return available_categories
    except sqlite3.Error as e:
        logger.error(f"Database error in get_available_categories: {e}")
        return []
    finally:
        conn.close()

def get_products_by_category(category: str) -> list:
    """Fetch all products for a given category."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name, price FROM products WHERE LOWER(category) = LOWER(?)", (category,))
        products = cursor.fetchall()
        return [{"name": product[0], "price": product[1]} for product in products]
    except sqlite3.Error as e:
        logger.error(f"Database error in get_products_by_category: {e}")
        return []
    finally:
        conn.close()

def validate_image_format(image_data: bytes) -> bool:
    """Validate if the image is in PNG or JPEG format."""
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            if img.format.lower() not in ['png', 'jpeg', 'jpg']:
                return False
            return True
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return False

def analyze_image_with_openai(image_data: bytes) -> tuple:
    """Analyze image using OpenAI's vision model and extract category recommendation."""
    
    # Get available categories from database
    available_categories = get_available_categories()
    if not available_categories:
        logger.error("No categories found in database")
        raise HTTPException(status_code=500, detail="No product categories available")
    
    # Encode image to base64
    image_b64 = base64.b64encode(image_data).decode('utf-8')
    
    # Create system prompt with available categories
    system_prompt = (
        "You are a vending machine AI designed to engage with teens in a fun, interactive, and humorous way. "
        "Your personality is witty, slightly sassy, and full of science-backed facts about snacks and drinks. "
        "Analyze the person in the image and provide: "
        "1. A brief, fun description of the person (1-2 sentences) "
        "2. Based on their appearance, age, style, or activity level, recommend ONE category from this list: "
        f"{', '.join(available_categories)} "
        "Format your response EXACTLY as: "
        "[DESCRIPTION]: Your fun description here "
        "[CATEGORY]: single_category_name "
        "Keep it engaging and trendy!"
    )
    
    # Prepare messages for API call
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this person and recommend a product category for them!"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        }
    ]
    
    try:
        # Make API call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        logger.info(f"OpenAI response: {content}")
        
        # Parse the response to extract description and category
        description = ""
        category = ""
        
        lines = content.split('\n')
        for line in lines:
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
        
        return description, category
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Vending Machine AI API!",
        "endpoints": {
            "/analyze-image": "POST - Upload an image for analysis",
            "/categories": "GET - Get all available product categories",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        categories = get_available_categories()
        return {
            "status": "healthy",
            "database": "connected",
            "categories_count": len(categories)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/categories")
async def get_categories():
    """Get all available product categories."""
    try:
        categories = get_available_categories()
        return {
            "categories": categories,
            "total": len(categories)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload an image and get AI-powered product recommendations.
    
    - **file**: Image file (PNG or JPEG format)
    
    Returns analysis with description, recommended category, and products.
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image file (PNG or JPEG)."
        )
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Validate image format
        if not validate_image_format(image_data):
            raise HTTPException(
                status_code=400,
                detail="Unsupported image format. Only PNG and JPEG images are supported."
            )
        
        # Check file size (limit to 10MB)
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image file too large. Maximum size is 10MB."
            )
        
        logger.info(f"Processing image: {file.filename}, size: {len(image_data)} bytes")
        
        # Analyze image with OpenAI
        description, recommended_category = analyze_image_with_openai(image_data)
        
        # Get products for the recommended category
        products = get_products_by_category(recommended_category)
        
        # Create response
        result = {
            "success": True,
            "filename": file.filename,
            "file_size_bytes": len(image_data),
            "analysis": {
                "description": description,
                "recommended_category": recommended_category,
                "products": products,
                "total_products": len(products)
            },
            "message": "Image analyzed successfully!"
        }
        
        logger.info(f"Analysis complete for {file.filename}: {recommended_category}")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing the image: {str(e)}"
        )

@app.get("/products/{category}")
async def get_products_by_category_endpoint(category: str):
    """Get all products for a specific category."""
    try:
        products = get_products_by_category(category)
        if not products:
            raise HTTPException(
                status_code=404,
                detail=f"No products found for category: {category}"
            )
        
        return {
            "category": category,
            "products": products,
            "total": len(products)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch products: {str(e)}")
    

# @app.post("/analyze-audio")
# async def analyze_audio(file: UploadFile = File(...)):
#     audio_data = await file.read()
#     # Save audio temporarily, or pass directly if you modify speech_to_text for bytes
#     temp_path = "/tmp/temp_audio.wav"
#     with open(temp_path, "wb") as f:
#         f.write(audio_data)
#     # Convert audio to text
#     transcribed_text = speech_to_text(temp_path)
#     # Analyze text with LLM
#     analysis_result = analyze_text_with_llm(transcribed_text)
#     return analysis_result    

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    audio_data = await file.read()
    # Use tempfile for cross-platform temp file creation
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data)
        temp_path = temp_audio.name
    try:
        # Convert audio to text
        transcribed_text = speech_to_text(temp_path)
        # Analyze text with LLM
        analysis_result = analyze_text_with_llm(transcribed_text)
        return analysis_result
    finally:
        # Clean up the temp file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

