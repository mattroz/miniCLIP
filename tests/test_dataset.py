import torch
import pytest

from pathlib import Path

from torchvision import transforms
from torch.utils.data import DataLoader

from src.data.loader import CocoDataset
from src.data.tokenizer import Tokenizer
from src.data.utils import collate_fn
from src.utils import load_yaml


@pytest.mark.parametrize("resize", [(112, 112), (224, 224), (448, 448), (1024, 1024)])
def test_dataset(resize):
    config = load_yaml(Path("configs/base_config.yaml").resolve(strict=True))
    path_to_annotation = Path(config["data"]["path_to_annotation"])
    path_to_images = Path(config["data"]["path_to_images"])

    _transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    tokenizer = Tokenizer(bos_str="<|startoftext|>", eos_str="<|endoftext|>", tokenizer_name="gpt2")

    dataset = CocoDataset(path_to_annotation, path_to_images, _transforms)
    
    for i in [10, 100, 1000, 10000]:
        image, annotation = dataset[i]

        assert isinstance(image, torch.Tensor)
        assert isinstance(annotation, str)

        assert image.shape == (3, resize[0], resize[1])

        assert annotation == tokenizer.decode(tokenizer.encode(annotation))


@pytest.mark.parametrize("resize, batch_size, sequence_length", [((112, 112), 64, 128), ((224, 224), 32, 64), ((448, 448), 16, 64), ((1024, 1024), 8, 64)])
def test_loader(resize, batch_size, sequence_length):
    config = load_yaml(Path("configs/base_config.yaml").resolve(strict=True))
    path_to_annotation = Path(config["data"]["path_to_annotation"])
    path_to_images = Path(config["data"]["path_to_images"])

    _transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    tokenizer = Tokenizer(bos_str="<|startoftext|>", eos_str="<|endoftext|>", tokenizer_name="gpt2")
    dataset = CocoDataset(path_to_annotation, path_to_images, _transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    for idx, batch in enumerate(loader):
        if idx == 5:
            break
        images, captions = batch

        assert isinstance(images, tuple)
        assert isinstance(captions, tuple)

        assert len(images) == batch_size
        assert len(captions) == batch_size

        encoded_text = tokenizer.encode_batch(captions, max_length=sequence_length)
        decoded_text = tokenizer.decode_batch(encoded_text, supress_special_tokens=True)

        assert list(captions) == decoded_text