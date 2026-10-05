from pathlib import Path

from services.file_utils import calculate_file_hash

pdf_path = Path("data/sample-docs/Information-Technology.pdf")

file_hash = calculate_file_hash(pdf_path)

print(f"File: {pdf_path.name}")
print(f"SHA-256: {file_hash}")
