from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Maximum sequence length:")
print(model.max_seq_length)

sentence_a = "The Master of Science in Cybersecurity prepares students for advanced study."

sentence_b = "Students can pursue graduate studies in cybersecurity."

sentence_c = "Chocolate cake should be baked in the oven."


embedding_a = model.encode(sentence_a)
embedding_b = model.encode(sentence_b)
embedding_c = model.encode(sentence_c)


similarity_ab = cos_sim(embedding_a, embedding_b)
similarity_ac = cos_sim(embedding_a, embedding_c)


print("A:")
print(sentence_a)

print("\nB:")
print(sentence_b)

print("\nC:")
print(sentence_c)

print("\nSimilarity A ↔ B:")
print(similarity_ab.item())

print("\nSimilarity A ↔ C:")
print(similarity_ac.item())