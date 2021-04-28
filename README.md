# Docker

## Build
```
docker build . -t crypto_predictions
``` 

## Simple run
```
docker run -it --rm crypto_predictions
```

## Extra docker arguments
### Load models from /external_saved_models dir

```
 docker run -it --rm -v $(pwd)/external_saved_models:/app/saved_models crypto_predictions
```
