from pymilvus import connections

def test_milvus_connection():
    try:
        connections.connect("default", host="0.0.0.0", port="19530")
        print("Successfully connected to Milvus!")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")

if __name__ == "__main__":
    test_milvus_connection()
