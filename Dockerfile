FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SAM weights (optional - will download on first run if not present)
# RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Copy app
COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
