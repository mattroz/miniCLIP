import cv2
import torch

from pathlib import Path
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    def __init__(self, path_to_annotation: Path, path_to_images: Path, transforms: list = None, tokenizer = None):
        self.annotation_file = path_to_annotation
        self.image_dir = path_to_images
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.coco = COCO(self.annotation_file)

    def __len__(self):
        return len(self.coco.anns)

    def load_annotation(self, idx: int):
        return list(self.coco.anns.values())[idx]

    def load_image_meta(self, id: int):
        return self.coco.loadImgs([id])[0]
    
    def load_image(self, id: int):
        image_meta = self.load_image_meta(id)
        path_to_image = Path(self.image_dir, image_meta["file_name"])
        
        image = cv2.imread(str(path_to_image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def __getitem__(self, idx):
        annotation = self.load_annotation(idx)
        image = self.load_image(annotation["image_id"])

        if self.transforms:
            image = self.transforms(image)
        
        #annotations_encoded = torch.tensor(self.tokenizer.encode(annotation["caption"]), dtype=torch.long)
        
        return image, annotation["caption"] #, annotations_encoded


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    import tiktoken
    from torchvision import transforms
    from src.utils import load_yaml

    config = load_yaml(Path("configs/base_config.yaml").resolve(strict=True))
    path_to_annotation = Path(config["data"]["path_to_annotation"])
    path_to_images = Path(config["data"]["path_to_images"])

    _transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = CocoDataset(path_to_annotation, path_to_images, _transforms, tokenizer)
    image, annotation = dataset[15000]
    
    image_to_save = image.permute(1, 2, 0).numpy() * 255
    image_to_save = image_to_save.astype("uint8")
    image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.jpg", image_to_save)

    print(image.shape)
    print(annotation.shape)
    print(annotation)
    print(tokenizer.decode(annotation.tolist()))