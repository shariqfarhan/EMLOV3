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
