from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, labeled_images, transform):
        self.transform = transform
        self.labeled_images = labeled_images

    def __len__(self):
        return len(self.labeled_images)

    def __getitem__(self, idx):
        image = Image.open(self.labeled_images[idx][1])
        print(image)
        image = self.transform(image)
        label = self.labeled_images[idx][0]

        return image, label
