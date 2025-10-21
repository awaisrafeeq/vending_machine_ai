from sqlalchemy import Column, Integer, String, Float, Text
from sqlalchemy.dialects.postgresql import JSONB  # For JSON fields; falls back to Text in SQLite
from database import Base

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    price = Column(Float)
    images = Column(JSONB, default=list)  # List of image URLs
    rating = Column(Float, default=0.0)
    description = Column(Text)
    flavor = Column(JSONB, default=list)  # List of flavors
    stock = Column(Integer, default=0)
    ingredients = Column(JSONB, default=list)  # List of ingredients
    category = Column(String, index=True)