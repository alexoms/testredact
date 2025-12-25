FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyMuPDF, Playwright, and wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libmupdf-dev \
    # Playwright dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium

# Copy application code
COPY pdf_unredact.py .
COPY api.py .

# Expose port
EXPOSE 16250

# Run the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "16250"]
