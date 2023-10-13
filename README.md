# Data Version Control (DVC)

## Using DVC to Push & Pull Data

### Introduction

In MLOps, managing data is a critical aspect alongside code and configurations. With the increasing complexity and diversity of data types, such as audio, video, text, and images, handling data effectively is essential. Data Version Control (DVC) addresses these challenges.

![MLOps Aspects](https://github.com/shariqfarhan/EMLOV3/assets/57046534/18ddafc5-a805-4b0c-9c86-cc78181118a6)

### Functioning

To install DVC, you can use the following commands:

```
pip install dvc
brew install dvc
```

DVC initializes and creates a file called `data.dvc`, which tracks changes in the `data` folder. Below is a sample entry from the `data.dvc` file:

```yaml
outs:
- md5: 32f48efaa3c0c25074d478c6de4853aa.dir
  size: 848866410
  nfiles: 24999
  hash: md5
  path: data
```

Git tracks changes to the `data.dvc` file, while DVC tracks changes to individual files in the `data` folder.

### Options to Store Data

DVC remotes can be stored in various locations, including:

1. Amazon S3
2. Microsoft Azure Blob Storage
3. Google Cloud Storage
4. SSH
5. HDFS
6. HTTP
7. Local files and directories outside the workspace

Once remotes are set up, you can use the following commands to push and pull data from the remotes. In the example below, we set up a local remote that is not part of the workspace.

1. Add a local remote for storing files outside the original workspace:
   ```shell
   dvc remote add -d local /workspace/dvc-data
   ```

2. Add DVC tracking to the 'data' folder:
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
## Hyperparameter configuration




# Lightning Template

A new file [infer](https://github.com/shariqfarhan/EMLOV3/blob/assignment_5/copper/infer.py) is added. It takes an image path as input, reads the image, and displays the model output, showing the probabilities of the top two classes. The input image path can be configured in a [config file](https://github.com/shariqfarhan/EMLOV3/blob/assignment_5/configs/infer.yaml) by modifying the `img_path` parameter.

### Sample Output

For a given image, the sample output may look like this:

```
Class: Cat, Probability: 0.2012
Class: Dog, Probability: 0.0720
```
You can use the following command to run the inference script:

```
copper_infer
```

You can use the following command to run the training script:

```
copper_train --help
```

Examples:

- `copper_train data.num_workers=16`
- `copper_train data.num_workers=16 trainer.deterministic=True +trainer.fast_dev_run=True`

## Development

To install in development mode, use the following command:

```
pip install -e .
```


