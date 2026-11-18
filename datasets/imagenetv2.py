import os
from .utils import Datum, DatasetBase, listdir_nohidden

from .imagenet import ImageNet

template = ["itap of a {}.",
                        "a bad photo of the {}.",
                        "a origami {}.",
                        "a photo of the large {}.",
                        "a {} in a video game.",
                        "art of the {}.",
                        "a photo of the small {}."]

class ImageNetV2(DatasetBase):
    """ImageNetV2.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenetv2"

    def __init__(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        image_dir = "imagenetv2-matched-frequency-format-val"
        self.image_dir = os.path.join(self.dataset_dir, image_dir)
        self.template = template
        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(test=data) 

    def read_data(self, classnames):
        image_dir = self.image_dir
        ordered_wnid_keys = list(classnames.keys())
        folders = listdir_nohidden(image_dir, sort=True)
        items = []

        for label, folder in enumerate(folders):

            # Map numeric folder name to WNID key, then to classname
            if folder.isdigit():
                index = int(folder)

                try:
                    wnid_key = ordered_wnid_keys[index]
                    classname = classnames[wnid_key]

                except (IndexError, KeyError):
                    print(f"Warning: Skipping folder {folder}. Index/Key mismatch.")
                    continue
            else:
                continue

            imnames = listdir_nohidden(os.path.join(image_dir, folder))

            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items