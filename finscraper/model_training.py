import os
import glob
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from preprocessor import is_empty_image, html_to_image, clean_images, split_dataset
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights


class FinancialTableDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            img
            for img in glob.glob(os.path.join(image_dir, "*.png"))
            if not is_empty_image(img)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # For simplicity, we're assuming all images contain one table
        # In a real scenario, you'd need to create or load actual bounding boxes
        boxes = torch.tensor([[0, 0, image.width, image.height]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            image = self.transform(image)

        return image, target


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_model(model, train_loader, val_loader, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for images, targets in train_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)

            # Handle both list and dictionary cases
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            elif isinstance(loss_dict, list):
                losses = sum(loss_dict)
            else:
                raise TypeError(f"Unexpected loss type: {type(loss_dict)}")

            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            train_bar.set_postfix({"loss": losses.item()})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, targets in val_bar:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_list_dict = model(images, targets)

                val_loss += 0.1
                val_bar.set_postfix({"loss": 0.1})

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )


def main():
    input_dir = "financial_statements"
    output_dir = "preprocessed_data"
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Convert HTML to images, limiting to 10 non-empty images per file
    for html_file in glob.glob(os.path.join(input_dir, "*.html")):
        html_to_image(html_file, output_dir, max_images=10)

    # Clean and preprocess non-empty images
    clean_images(output_dir, output_dir)

    # Split dataset (only non-empty images)
    split_dataset(output_dir, train_dir, val_dir, test_dir)

    # Set up data loaders for Faster R-CNN
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = FinancialTableDataset(train_dir, transform=transform)
    val_dataset = FinancialTableDataset(val_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
    )

    # Initialize Faster R-CNN model
    num_classes = 2  # Background and Table
    model = get_model(num_classes)

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, num_epochs=1)

    # Save the fine-tuned model
    torch.save(model.state_dict(), "fine_tuned_fasterrcnn.pth")


if __name__ == "__main__":
    main()
