# How to Use this repository Dockerfile

Run the below command to train the model on CIFAR data. This also mounts the data and the model repository for accessing the file generated in the docker command.

```
docker run -v ./data:/app/data -v ./model:/app/model cifar-training  python train.py
```
Run the below command to evaluate the model on CIFAR data. This accesses the mounted data from the data and the model folder.

```
docker run -v ./data:/app/data -v ./model:/app/model cifar-training  python eval.py
```
