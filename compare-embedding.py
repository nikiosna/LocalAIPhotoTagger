#!/usr/bin/env python3
import sys
import requests
import numpy as np

def get_embeddings(texts):
    url = "http://localhost:1234/v1/embeddings"
    response = requests.post(
        url,
        json={
            "model": "text-embedding-nomic-embed-text-v1.5",
            "input": texts
        }
    )
    response.raise_for_status()
    return [d["embedding"] for d in response.json()["data"]]

def absolute_distance(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.sum(np.abs(vec1 - vec2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} 'text1' 'text2'")
        sys.exit(1)

    text1, text2 = sys.argv[1], sys.argv[2]
    embeddings = get_embeddings([text1, text2])

    abs_dist = absolute_distance(embeddings[0], embeddings[1])

    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Absolute distance: {abs_dist:.4f}")
