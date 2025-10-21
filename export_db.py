# import sqlite3
# import json
# import os

# # Connect to SQLite
# conn_sqlite = sqlite3.connect('Newdb.db')
# cursor = conn_sqlite.cursor()

# # Fetch all products (adjust column names/order if your schema differs)
# cursor.execute('SELECT id, name, price, images, rating, description, flavor, stock, ingredients, category FROM Products')
# products = cursor.fetchall()

# # Convert to list of dicts (handle JSON fields)
# data = []
# for row in products:
#     product = {
#         'id': row[0],
#         'name': row[1],
#         'price': float(row[2]) if row[2] else 0.0,  # Ensure float for price
#         'images': json.loads(row[3]) if row[3] else [],
#         'rating': float(row[4]) if row[4] else 0.0,
#         'description': row[5] or '',
#         'flavor': json.loads(row[6]) if row[6] else [],
#         'stock': int(row[7]) if row[7] else 0,
#         'ingredients': json.loads(row[8]) if row[8] else [],
#         'category': row[9] or ''
#     }
#     data.append(product)

# # Save as JSON for easy import
# with open('products_export.json', 'w') as f:
#     json.dump(data, f, indent=2, default=str)

# print(f"Exported {len(data)} products to products_export.json")
# conn_sqlite.close()

import sqlite3

conn = sqlite3.connect('Newdb.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in DB:", tables)
conn.close()