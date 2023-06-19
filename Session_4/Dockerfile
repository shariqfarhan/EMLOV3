# Base image
FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Start the development container
CMD ["/bin/bash"]
