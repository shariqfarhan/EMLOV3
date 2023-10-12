# CIFAR10 Training & Evaluation using Copper

Follow below steps for this

1. Pull Docker image from [Docker Hub](https://hub.docker.com/repository/docker/shariqfarhan/copper_cifar/general)

2. Use command
```
docker pull shariqfarhan/copper_cifar:latest
```

3. Run docker image with the below command
```
docker run -it shariqfarhan/copper_cifar:latest bash
```

4. Once inside the container run the below commands

## Lightning Template

```
copper_cifar10_train --help
```

examples

- `copper_cifar10_train data.num_workers=16`
- `copper_cifar10_eval data.num_workers=16`

## Development

Install in dev mode

```
pip install -e .
```

### Docker

If faced any challenges, upgrade torch & torchvision with the below command

```
pip install --upgrade torch torchvision
```

# Steps for Building Docker Image

1. Make a replica of the original base directory [Github Repo](https://github.com/satyajitghana/lightning-template/tree/master)
2. Make copies of various config files in according with CIFAR10 - [CIFAR10_data](https://github.com/shariqfarhan/EMLOV3/blob/master/configs/data/cifar10.yaml), [CIFAR10 model](https://github.com/shariqfarhan/EMLOV3/blob/master/configs/model/cifar10.yaml), [CIFAR10_train](https://github.com/shariqfarhan/EMLOV3/blob/master/configs/cifar10_train.yaml)
3. Similarly make changes to the files in copper - [CIFAR10_Data Module](https://github.com/shariqfarhan/EMLOV3/blob/master/copper/data/cifar10_datamodule.py), [CIFAR10 Lightning Module](https://github.com/shariqfarhan/EMLOV3/blob/master/copper/models/cifar10_module.py), [CIFAR10 Neural Network](https://github.com/shariqfarhan/EMLOV3/blob/master/copper/models/components/cifar10_dense_net.py)
4. Update the [CIFAR10 Train](https://github.com/shariqfarhan/EMLOV3/blob/master/copper/cifar10_train.py) script for training the model based on the above files
5. Evaluate the model based on [CIFAR10 Eval](https://github.com/shariqfarhan/EMLOV3/blob/master/copper/cifar10_eval.py)
6. The two files [CIFAR10 Train](https://github.com/shariqfarhan/EMLOV3/blob/master/configs/cifar10_train.yaml) & [CIFAR10 Test](https://github.com/shariqfarhan/EMLOV3/blob/master/configs/cifar10_test.yaml) are used to define the training & evaluation parameters
7. Finally update the [setup](https://github.com/shariqfarhan/EMLOV3/blob/master/setup.py) file with the relevant command scripts
```
"copper_cifar10_train = copper.cifar10_train:main",
"copper_cifar10_eval = copper.cifar10_eval:main",
```
8. Once the base files are built
9. Create the [Dockerfile](https://github.com/shariqfarhan/EMLOV3/blob/master/Dockerfile)
10. Run the below command
```
docker build -t copper_cifar:latest .
```
11. To run the above image
```
docker run -it copper_cifar:updated bash
```
12. Run below commands for training & evaluation within the docker image
```
- `copper_cifar10_train data.num_workers=16`
- `copper_cifar10_eval data.num_workers=16`
```
13. Publish the code to github using below command (if publishing from a remote location e.g. gitpod)
```
git add .
git commit -m "Session 4"
git push https://github.com/shariqfarhan/EMLOV3
```
14. To publish docker image to docker hub
```
docker login
docker image tag copper_cifar:latest shariqfarhan/copper_cifar:latest
docker push
```
Before docker push it's mandatory to rename image to username/image:tag format
