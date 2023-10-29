In this session, we 




## Docker Image Cleanup

To delete all Docker images and reclaim valuable disk space, you can use the following command:

```bash
docker system prune -a
```

Docker may delete images, but the space they occupied is not immediately reclaimed. This command efficiently deletes all images and reclaims the associated space. It is particularly useful when dealing with images that consume a substantial amount of storage.

---

## Publishing Changes to a Remote Repository

When creating a replica of a GitHub repository and attempting to publish your changes, GitHub may encounter authentication errors. To track your work and continue building upon it, follow these steps:

1. Create a new branch
   ```
   git checkout -b <branch-name>
   ```

1. Add all the modified files to your local repository:

   ```bash
   git add .
   ```

2. Commit the changes with a descriptive message:

   ```bash
   git commit -m "Your commit message here"
   ```

3. Link your local repository to a remote repository by specifying a branch name and its corresponding repository:

   ```bash
   git remote add <branch-name> <repository-name>
   ```

   For example:

   ```bash
   git remote add assignment_6 https://github.com/shariqfarhan/EMLOV3
   ```

4. Push your local changes to the remote repository:

   ```bash
   git push
   ```

This process creates a new branch in the specified remote repository, enabling you to manage and expand upon your work seamlessly.

---


# Lightning Template

```
copper_train --help
```

examples

- `copper_train data.num_workers=16`
- `copper_train data.num_workers=16 trainer.deterministic=True +trainer.fast_dev_run=True`

### Cat vs Dog with ViT

```
find . -type f -empty -print -delete
```


```
copper_train experiment=cat_dog data.num_workers=16 +trainer.fast_dev_run=True
```

```
copper_train experiment=cat_dog data.num_workers=16
```

## Development

Install in dev mode

```
pip install -e .
```

### Docker

<docker-usage-instructions-here>
