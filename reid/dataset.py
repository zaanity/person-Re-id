from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class ImageReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None, return_paths=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_paths = return_paths
        self.samples = []
        for pid_dir in sorted(self.root_dir.iterdir()):
            if not pid_dir.is_dir(): continue
            pid = int(pid_dir.name)
            for img_file in pid_dir.glob('*.*'):
                self.samples.append((img_file, pid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.return_paths:
            return img, pid, str(path)
        return img, pid
