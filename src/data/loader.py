import cv2

from pathlib import Path
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    def __init__(self, path_to_annotation: Path, path_to_images: Path, transforms: list = None):
        self.annotation_file = path_to_annotation
        self.image_dir = path_to_images
        self.transforms = transforms
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
        
        # Handle uppercase-only annotations
        annotation["caption"] = annotation["caption"][0] + annotation["caption"][1:].lower()

        return image, annotation["caption"]