# CIFAR10 Training & Evalution using Copper

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
