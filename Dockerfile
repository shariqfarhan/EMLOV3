# # Use a base image with the desired OS and dependencies
# FROM python:3.8

# # Set the working directory in the container
# WORKDIR /app

# # Copy your project files into the container
# COPY . /app

# # Install Python dependencies using pip
# RUN pip install -e .

# # Specify the command to run your script
# CMD ["copper_cifar10_train data.num_workers=16"]

# Build Stage
FROM python:3.8-slim as build

WORKDIR /app

# Install your Python dependencies
COPY . /app
RUN pip install --no-dependencies -e .
RUN pip install torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install lightning[extra]>=2.0.0



# Set the entry point for your script
CMD ["copper_cifar10_train", "data.num_workers=16"]
