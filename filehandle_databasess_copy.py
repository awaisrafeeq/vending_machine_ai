from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import get_db
from models import Product
import json
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import tempfile
from typing import Dict, Any
import logging
from voice_module import speech_to_text, analyze_text_with_llm
from openai import OpenAI

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
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY is not set; OpenAI calls will fail.")
client = OpenAI(api_key=api_key)

def get_available_categories(db: Session) -> list:
    """Fetch all unique categories from the products table."""
    categories = db.query(Product.category).distinct().filter(Product.category.isnot(None)).all()
    return [cat[0] for cat in categories if cat[0]]

def get_products_by_category_data(db: Session, category: str) -> list:
    """Fetch all products for a given category (used internally by AI analysis)."""
    products = db.query(Product.name, Product.price).filter(Product.category.ilike(category)).all()
    return [{"name": p[0], "price": p[1]} for p in products]

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

def analyze_image_with_openai(image_data: bytes, db: Session) -> tuple:
    """Analyze image using OpenAI's vision model and extract category recommendation."""
    
    # Get available categories from database
    available_categories = get_available_categories(db)
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
        logger.error(f"Error calling OpenAI...(truncated 3003 characters)...")

@app.get("/")
async def welcome():
    return {"message": "Welcome to the Vending Machine AI API!", "endpoints": {
        "/analyze-image": "POST - Upload an image for analysis",
        "/categories": "GET - Get all available product categories",
        "/products": "GET - Get all products",
        "/products/category/{category_name}": "GET - Get products by category",
        "/product/{product_id}": "GET - Get product by ID",
        "/health": "GET - Health check"
    }}

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        categories = get_available_categories(db)
        return {"status": "healthy", "database": "connected", "categories_count": len(categories)}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/categories")
async def get_categories(db: Session = Depends(get_db)):
    categories = get_available_categories(db)
    return {"categories": categories, "total": len(categories)}

@app.get("/products")
async def get_all_products(db: Session = Depends(get_db)):
    products = db.query(Product).all()
    result = []
    for p in products:
        p_dict = {c.name: getattr(p, c.name) for c in p.__table__.columns}
        result.append(p_dict)
    return JSONResponse(content=result)

@app.get("/category")
async def get_products_by_category(name: str = Query(..., description="Category name"), db: Session = Depends(get_db)):
    products = db.query(Product).filter(Product.category.ilike(name)).all()
    if not products:
        raise HTTPException(status_code=404, detail=f"No products found in category '{name}'")

    result = []
    for product in products:
        product_dict = {c.name: getattr(product, c.name) for c in product.__table__.columns}
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

@app.get("/product")
async def get_product(id: int = Query(..., description="Product ID"), db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == id).first()
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")

    product_dict = {c.name: getattr(product, c.name) for c in product.__table__.columns}
    for field in ['images', 'flavor', 'ingredients']:
        if product_dict.get(field):
            try:
                product_dict[field] = json.loads(product_dict[field])
            except json.JSONDecodeError:
                product_dict[field] = []
        else:
            product_dict[field] = []

    return JSONResponse(content=product_dict)

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
        description, recommended_category = analyze_image_with_openai(image_data, db)

        # Get products for the recommended category
        products = get_products_by_category_data(db, recommended_category)
        
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
async def analyze_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
        products = get_products_by_category_data(db, category)
        print(file.filename, transcribed_text, description)
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