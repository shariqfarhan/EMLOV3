# Base Image - python 3.9
FROM python:3.9-slim-buster

WORKDIR /workspace
COPY .  .

# Install all requirements from requirements.txt
# Install custom library copper & to make it editable we use -e .

# torch==1.10.0+cpu

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir --no-dependencies torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir --no-dependencies numpy typing-extensions pillow && \
    pip install hydra-joblib-launcher --upgrade && \
    pip install --upgrade torch torchvision

# Expose port for MLFLOW UI
EXPOSE 5000

# Run Copper command
# Set the entry point for your script
CMD ["copper_train", " -m hydra/launcher=joblib hydra.launcher.n_jobs=3 experiment=vit model.net.patch_size=1,2,4 trainer.max_epochs=1 data.num_workers=0"]