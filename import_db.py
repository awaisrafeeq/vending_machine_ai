import json
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models import Base, Product
from database import DATABASE_URL

print(f"Using DATABASE_URL: {DATABASE_URL}")  # Debug print
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)  # Create tables

# Create a session
session = Session(engine)

try:
    # Load export with full path
    with open(r'C:\Users\Public\Projects\Eric_Procject\Code_file-completed runing code github\Code_file\products_export.json', 'r') as f:
        data = json.load(f)

    # Merge (update or insert) records
    updated_count = 0
    for item in data:
        product = Product(**item)
        session.merge(product)  # Updates if exists, inserts if new
        updated_count += 1

    # Commit the transaction
    session.commit()
    print(f"Data imported! {updated_count} records updated or inserted.")
except Exception as e:
    session.rollback()  # Rollback on error
    print(f"Error importing data: {e}")
finally:
    session.close()  # Ensure session is closed