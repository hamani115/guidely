import numpy as np

vector_a = np.array([0.9, 0.8, 0.1])
vector_b = np.array([0.8, 0.9, 0.1])
vector_c = np.array([0.1, 0.0, 0.9])

def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray):
    dot_product = np.dot(vector1, vector2)
    
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    return dot_product / (magnitude1 * magnitude2)

print("A comparted to B:")
print(cosine_similarity(vector_a, vector_b))

print("A comparted to C:")
print(cosine_similarity(vector_a, vector_c))

