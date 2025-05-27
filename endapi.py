from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import sqlite3
import json

app = FastAPI()

DATABASE = 'Newdb.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # So that rows behave like dict
    return conn

#http://127.0.0.1:8000/products
# 1. Get all products
@app.get("/products")
async def get_all_products():
    conn = get_db_connection()
    products = conn.execute('SELECT * FROM Products').fetchall()
    conn.close()

    result = []
    for product in products:
        product_dict = dict(product)
        # Convert JSON fields back to Python lists/dicts
        for field in ['images', 'flavor', 'ingredients']:
            if product_dict.get(field):
                product_dict[field] = json.loads(product_dict[field])
            else:
                product_dict[field] = []
        result.append(product_dict)

    return JSONResponse(content=result)
 
#http://127.0.0.1:8000/products/category/Protein
# 2. Get products by category
@app.get("/products/category/{category_name}")
async def get_products_by_category(category_name: str):
    conn = get_db_connection()
    products = conn.execute('SELECT * FROM Products WHERE category = ?', (category_name,)).fetchall()
    conn.close()

    if not products:
        raise HTTPException(status_code=404, detail=f"No products found in category '{category_name}'")

    result = []
    for product in products:
        product_dict = dict(product)
        for field in ['images', 'flavor', 'ingredients']:
            if product_dict.get(field):
                product_dict[field] = json.loads(product_dict[field])
            else:
                product_dict[field] = []
        result.append(product_dict)

    return JSONResponse(content=result)

#http://127.0.0.1:8000/product/5
# 3. Get product by ID
@app.get("/product/{product_id}")
async def get_product(product_id: int):
    conn = get_db_connection()
    product = conn.execute('SELECT * FROM Products WHERE id = ?', (product_id,)).fetchone()
    conn.close()

    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")

    product_dict = dict(product)
    for field in ['images', 'flavor', 'ingredients']:
        if product_dict.get(field):
            product_dict[field] = json.loads(product_dict[field])
        else:
            product_dict[field] = []

    return JSONResponse(content=product_dict)
