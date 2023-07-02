# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir dvc

# Copy the code
COPY . .

# Mount the logs directory
VOLUME /app/logs

# Expose the Logger UI server port
EXPOSE 8000

# Run the training script
CMD ["python", "train.py"]
