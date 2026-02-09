# DocScribe Chainlit - Dockerfile for Railway deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway uses PORT env var)
EXPOSE 8000

# No healthcheck in Dockerfile - let Railway handle it
# Chainlit serves on root path which returns HTML

# Run Chainlit
CMD ["sh", "-c", "chainlit run app.py --host 0.0.0.0 --port ${PORT:-8000}"]
