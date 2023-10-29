# FROM python:3.7.10-slim-buster

# RUN export DEBIAN_FRONTEND=noninteractive \
#     && echo "LC_ALL=en_US.UTF-8" >> /etc/environment \
#     && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
#     && echo "LANG=en_US.UTF-8" > /etc/locale.conf \
#     && apt update && apt install -y locales \
#     && locale-gen en_US.UTF-8 \
#     && rm -rf /var/lib/apt/lists/*

# RUN pip install \
#     torch==1.9.0+cpu \
#     torchvision==0.10.0+cpu \
#     torchaudio==0.9.0 \
#     -f https://download.pytorch.org/whl/torch_stable.html \
#     && rm -rf /root/.cache/pip

# ENV LANG=en_US.UTF-8 \
#     LANGUAGE=en_US:en \
#     LC_ALL=en_US.UTF-8

# Build Stage
FROM python:3.8-slim as build

WORKDIR /app

# Copy your Python code and requirements file
COPY . /app
COPY requirements.txt /app

# Install your Python dependencies from requirements.txt
RUN pip install --no-dependencies -e . \
    && pip install torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install lightning[extra]>=2.0.0 mlflow \
    && pip install -r requirements.txt \
    && pip install hydra-joblib-launcher --upgrade \
    && pip install --upgrade torch torchvision \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Expose port for MLflow UI (e.g., port 5000)
EXPOSE 5000

# Set the entry point for your script
CMD ["copper_train", " -m hydra/launcher=joblib hydra.launcher.n_jobs=1 experiment=vit model.net.patch_size=1,2,4,8,16 trainer.max_epochs=1 data.num_workers=0"]
