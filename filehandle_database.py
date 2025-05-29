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
api_key = os.getenv('OPENAI_API_KEY') 
client = OpenAI(api_key=api_key)

DATABASE = 'Newdb.db'

# Database connection function
def get_db_connection():
    """Create a new database connection for each request."""
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row  # So that rows behave like dict
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

def get_products_by_category_data(category: str) -> list:
    """Fetch all products for a given category (used internally by AI analysis)."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name, price FROM products WHERE LOWER(category) = LOWER(?)", (category,))
        products = cursor.fetchall()
        return [{"name": product[0], "price": product[1]} for product in products]
    except sqlite3.Error as e:
        logger.error(f"Database error in get_products_by_category_data: {e}")
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

# ============= ENDPOINTS =============

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Vending Machine AI API!",
        "endpoints": {
            "/analyze-image": "POST - Upload an image for analysis",
            "/categories": "GET - Get all available product categories",
            "/products": "GET - Get all products",
            "/products/category/{category_name}": "GET - Get products by category",
            "/product/{product_id}": "GET - Get product by ID",
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

@app.get("/products")
async def get_all_products():
    """Get all products from the database."""
    conn = get_db_connection()
    try:
        products = conn.execute('SELECT * FROM Products').fetchall()
        
        result = []
        for product in products:
            product_dict = dict(product)
            # Convert JSON fields back to Python lists/dicts
            for field in ['images', 'flavor', 'ingredients']:
                if product_dict.get(field):
                    try:
                        product_dict[field] = json.loads(product_dict[field])
                    except json.JSONDecodeError:
                        product_dict[field] = []
                else:
                    product_dict[field] = []
            result.append(product_dict)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error fetching all products: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch products: {str(e)}")
    finally:
        conn.close()

@app.get("/products/category/{category_name}")
async def get_products_by_category(category_name: str):
    """Get all products for a specific category with full product details."""
    conn = get_db_connection()
    try:
        products = conn.execute('SELECT * FROM Products WHERE category = ?', (category_name,)).fetchall()
        
        if not products:
            raise HTTPException(status_code=404, detail=f"No products found in category '{category_name}'")

        result = []
        for product in products:
            product_dict = dict(product)
            for field in ['images', 'flavor', 'ingredients']:
                if product_dict.get(field):
                    try:
                        product_dict[field] = json.loads(product_dict[field])
                    except json.JSONDecodeError:
                        product_dict[field] = []
                else:
                    product_dict[field] = []
            result.append(product_dict)

        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching products by category: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch products: {str(e)}")
    finally:
        conn.close()

@app.get("/product/{product_id}")
async def get_product(product_id: int):
    """Get a specific product by ID."""
    conn = get_db_connection()
    try:
        product = conn.execute('SELECT * FROM Products WHERE id = ?', (product_id,)).fetchone()
        
        if product is None:
            raise HTTPException(status_code=404, detail="Product not found")

        product_dict = dict(product)
        for field in ['images', 'flavor', 'ingredients']:
            if product_dict.get(field):
                try:
                    product_dict[field] = json.loads(product_dict[field])
                except json.JSONDecodeError:
                    product_dict[field] = []
            else:
                product_dict[field] = []

        return JSONResponse(content=product_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching product by ID: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch product: {str(e)}")
    finally:
        conn.close()

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
        products = get_products_by_category_data(recommended_category)
        
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
        
@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    # Validate content type
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an audio file."
        )
    
    audio_data = await file.read()
    
    # Optional: file size limit
    if len(audio_data) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Audio file too large. Maximum size is 10MB."
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data)
        temp_path = temp_audio.name
    
    try:
        # Convert audio to text
        transcribed_text = speech_to_text(temp_path)
        
        # Get description and category tuple
        description, category = analyze_text_with_llm(transcribed_text)
        
        # Get products list by category
        products = get_products_by_category(category)
        
        response = {
            "success": True,
            "filename": file.filename,
            "file_size_bytes": len(audio_data),
            "transcription": transcribed_text,
            "analysis": {
                "description": description,
                "recommended_category": category,
                "products": products,
                "total_products": len(products)
            },
            "message": "Audio analyzed successfully!"
        }
        
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing the audio: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
