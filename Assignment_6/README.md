# Leverage Docker for Experiment Tracking

We create a docker file to train ViT on CIFAR10 with different patch sizes.

The train.py script contains the code for training the ViT model with different patch sizes. 
The Dockerfile included in this repository sets up the environment and dependencies required to run the training script in a Docker container. 

Below are the 2 commands to be run to create the docker image and then training it.

```
docker build -t vit-cifar10 .
```

```
docker run -p 8000:8000 -v /logs:/app/logs vit-cifar10
```


Access the Logger UI:
Open a web browser and go to http://localhost:8000 (assuming the container is running on the local machine). The Logger UI provides real-time monitoring of the training progress and allows you to view the logs.

