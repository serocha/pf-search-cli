# CLI-only Python environment
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY run.py .

RUN mkdir -p /app/data /app/output

# Set the default command to show help
CMD ["python", "run.py", "help"] 