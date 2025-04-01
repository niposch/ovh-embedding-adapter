# original python code, later rewritten in Go
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

OVH_BATCH_API_URL = "https://bge-multilingual-gemma2.endpoints.kepler.ai.cloud.ovh.net/api/batch_text2vec"
OVH_TOKEN = os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN")

MAX_BATCH_SIZE = 10  # Max batch size for OVH API

@app.route("/v1/embeddings", methods=["POST"])
def get_embeddings():
    print("Received request for embeddings length:", len(request.data))
    try:
        data = request.json
        input_data = data.get("input", "")
        
        # Handle both string and array inputs from Continue
        if isinstance(input_data, list):
            texts = input_data
        else:
            texts = [input_data]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OVH_TOKEN}"
        }
        
        # Split texts into chunks of MAX_BATCH_SIZE
        all_embeddings = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i:i + MAX_BATCH_SIZE]
            print(f"Processing batch {i//MAX_BATCH_SIZE + 1}/{(len(texts) + MAX_BATCH_SIZE - 1)//MAX_BATCH_SIZE}, size: {len(batch)}")
            
            # Send batch request to OVH
            response = requests.post(
                OVH_BATCH_API_URL, 
                json=batch,
                headers=headers
            )
            
            if response.status_code == 200:
                # Add embeddings from this batch to our results
                all_embeddings.extend(response.json())
            else:
                return jsonify({
                    "error": f"Error from OVH API (batch starting at index {i}): {response.status_code}, response: {response.text}"
                }), 500
        
        # Format response to match OpenAI's format
        embeddings_results = []
        for i, embedding in enumerate(all_embeddings):
            embeddings_results.append({
                "embedding": embedding,
                "index": i,
                "object": "embedding"
            })
        
        return jsonify({
            "data": embeddings_results,
            "model": "ovh-embeddings",
            "object": "list",
            "usage": {
                "prompt_tokens": sum(len(str(t).split()) for t in texts),
                "total_tokens": sum(len(str(t).split()) for t in texts)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8000)
