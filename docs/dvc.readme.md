# DVC Commands


```shell
# Remove all DVC-tracked outputs
dvc remove --outs

# Clean the cache
dvc gc -w

# Reproduce the pipeline
dvc repro

# Remove DVC lock files
rm -f .dvc/tmp/lock
rm -f .dvc/tmp/*.lock
rm -f .dvc/tmp/rwlock


# Clean DVC state
rm -f .dvc/state
rm -f .dvc/state-journal
rm -f .dvc/state-wal

# Reinitialize DVC
dvc init --no-scm

lsof | grep .dvc


# Remove DVC files and reinitialize
rm -rf .dvc
dvc init
dvc add [your-data-files]
dvc repro -f

# Cleam the pipeline
dvc remove --outs
dvc gc -w
```