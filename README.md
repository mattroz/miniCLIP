# miniCLIP
Implementation of CLIP model with a reduced capacity. For self-educational purposes only.

![clip_summary](https://github.com/mattroz/miniCLIP/assets/8377365/f86721f8-1440-470e-9304-398754ef6d45)

This repo currently contains only CLIP-ResNet implementation, while in the original paper there are 5 ResNets and 3 ViTs models.
There was no intention to beat SotA or train a superior version of CLIP. This is just an attempt to understand the logic behind CLIP.

## Preliminary results
After training CLIP-ResNet50 for 10 epochs, the following results were obtained.

<p float="left">
  <img src="https://github.com/mattroz/miniCLIP/assets/8377365/498f8f63-c85e-4fc1-aa1c-f8282889dedc" width="49%" />
  <img src="https://github.com/mattroz/miniCLIP/assets/8377365/9c483de2-16a1-4ca4-910e-60cffb48b626" width="49%" />
</p>

<p float="left">
  <img src="https://github.com/mattroz/miniCLIP/assets/8377365/3d79455f-f530-4b89-8c49-2ea6b1c0002e" width="49%" />
  <img src="https://github.com/mattroz/miniCLIP/assets/8377365/5f96c92f-eb8d-4f98-94fe-806f16216dca" width="49%" />
</p>

As can be seen, the results are not great, but the model is definetely trying to stick closer to correct pairs. 

## Example usage
### Train
To run the training, you should first download the COCO dataset and provide paths to annotations and images for both `train` and `val` in a config (check example [here](https://github.com/mattroz/miniCLIP/blob/main/configs/clip_base.yaml)).
After that, run:
```
python tools/train.py --path_to_config=configs/clip_base.yaml --path_to_log=logs/
```

This will create directory structure under the `logs/` directory for each run separately (aka experiment directories):
```
logs/
  |--{experiment_name}/
      |--artifacts/
      |--checkpoints/
      |--train.log
      |--{experiment_name}.yaml               
```
Under the `logs/{experiment_name}/artifacts/` a `training_progress.log` will be saved, containing losses for train and validation.
Each training run generates an overrided config and saves it under the `logs/{experiment_name}/` directory.

### Plot similarity matrices
To plot similarity matrices on validation dataset, run:
```
python tools/plot_similarities.py --path_to_config=logs/{experiment_name}/{experiment_name}.yaml \
                                  --path_to_ckpt=logs/{experiment_name}/checkpoints/some_ckpt.pth \
                                  --n_pairs=8 \
                                  --n_matricies=5
```

Here, `n_matricies` denotes number of similarity matrices to create, and `n_pairs` denotes number of image-text pairs to include into each similarity matrix.
All the similarity matrices will be saved under `logs/{experiment_name}/artifacts/`.
