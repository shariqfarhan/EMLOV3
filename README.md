# Data Version Control (DVC)

## Using DVC to Push & Pull Data

### Introduction

One of the main differences between DevOps & MLOps is shown in the image below. While DevOps involves only the code aspect, MLOps has 3 different aspects - Coda, Data, Configurations.
Git is great at identifying & managing changes in the code. MLOps has an added nuance of identifying changes in Data. Data could be in various forms - especially with advances in Neural networks. Data is now multimodal - audio, video, text, images etc. 

To address these challenges, DVC was created.

<img width="784" alt="image" src="https://github.com/shariqfarhan/EMLOV3/assets/57046534/18ddafc5-a805-4b0c-9c86-cc78181118a6">

### Functioning

To install dvc use the below commands
```
pip install dvc
brew install dvc 
```

When dvc is initialized DVC creates a file called data.dvc. Below is a sample data from the data.dvc file created once the dvc init command is run.

```
outs:
- md5: 32f48efaa3c0c25074d478c6de4853aa.dir
  size: 848866410
  nfiles: 24999
  hash: md5
  path: data

```
Git tracks changes to this file and DVC tracks changes to the individual files in the data folder.

### Options to Store Data

DVC Remotes can be stored in any of the below locations

1. Amazon S3
2. Microsoft Azure Blob Storage
3. Google Cloud Storage
4. SSH
5. HDFS
6. HTTP
7. Local files and directories outside the workspace


Once the above remotes are setup, the below commands can be used to push & pull data from the remotes.
In the below example, we set up a local remote not part of the workspace.

1. Add a local remote for storing files outside the original workspace:
    ```shell
    dvc remote add -d local /workspace/dvc-data
    ```

2. Add DVC Tracking to the 'data' folder:
    ```shell
    dvc add data
    ```

3. Push data to the local remote:
    ```shell
    dvc push -r local
    ```

4. To checkout the previous version of the code:
    ```shell
    git checkout xxxxx
    ```

5. For the corresponding checkout in Git, the data is checked out:
    ```shell
    dvc checkout
    ```




# Lightning Template
As shown in the below code. We add a new file [infer](https://github.com/shariqfarhan/EMLOV3/blob/assignment_5/copper/infer.py). Given an image path, read the image and display the model output.
This shows the show probabilities of top2 classes.

The input for the image path is passed through a [config file](https://github.com/shariqfarhan/EMLOV3/blob/assignment_5/configs/infer.yaml).
We add a parameter img_path which can be changed as needed.

### Sample Output 

For a given image, the sample output is as below
```
Class: Cat, Probability: 0.2012
Class: Dog, Probability: 0.0720
```

```
copper_train --help
```

examples

- `copper_train data.num_workers=16`
- `copper_train data.num_workers=16 trainer.deterministic=True +trainer.fast_dev_run=True`

## Development

Install in dev mode

```
pip install -e .
```

