from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import numpy as np

# Connect to Milvus
connections.connect("default", host="0.0.0.0", port="19530")

# Define a simple collection schema with a primary key
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, "A simple collection")

# Create a collection
collection_name = "test_collection"
collection = Collection(name=collection_name, schema=schema)

# Check if the collection exists
if collection.is_empty:
    print(f"Collection '{collection_name}' created successfully.")
else:
    print(f"Failed to create collection '{collection_name}'.")

# Insert some dummy data
vectors = np.random.random([10, 128]).astype(np.float32)
collection.insert([vectors])

# Check the number of entities
print(f"Number of entities in collection: {collection.num_entities}")
