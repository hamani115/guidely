import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

text = "The Master of Science in Cybersecurity prepares students for advanced study."

tokens = encoding.encode(text)

print("Original text:")
print(text)

print("\nToken IDs:")
print(tokens)

print("\nNumber of tokens:")
print(len(tokens))

print("\nDecoded text:")
print(encoding.decode(tokens))
