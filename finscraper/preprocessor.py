import os
import glob
from PIL import Image
import imgkit
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from PIL import ImageEnhance
import numpy as np
import cv2


def is_empty_image(img_path, min_text_area=0.01, min_lines=3):
    # Read the image
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to preprocess the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours for text regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate total area of text regions
    total_text_area = sum(cv2.contourArea(contour) for contour in contours)
    image_area = img.shape[0] * img.shape[1]
    text_area_ratio = total_text_area / image_area

    # Count lines using Hough Line Transform
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    num_lines = 0 if lines is None else len(lines)

    # Check if the image has enough text area and lines
    return text_area_ratio < min_text_area or num_lines < min_lines


def html_to_image(html_file, output_dir, max_images=10):
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")

    processed_images = 0
    for i, table in enumerate(tables):
        if processed_images >= max_images:
            break
        img_filename = f"{os.path.basename(html_file)[:-5]}_table_{i}.png"
        img_path = os.path.join(output_dir, img_filename)
        imgkit.from_string(str(table), img_path)

        if not is_empty_image(img_path):
            print(f"Converted table {i} from {html_file} to {img_path}")
            processed_images += 1
        else:
            os.remove(img_path)
            print(f"Skipped content-less image from table {i} in {html_file}")


def clean_images(image_dir, output_dir, target_size=(800, 600)):
    for img_file in glob.glob(os.path.join(image_dir, "*.png")):
        if not is_empty_image(img_file):
            with Image.open(img_file) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize image
                img = img.resize(target_size, Image.LANCZOS)

                # Enhance contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)

                # Save processed image
                output_path = os.path.join(output_dir, os.path.basename(img_file))
                img.save(output_path)
                print(f"Cleaned and saved: {output_path}")
        else:
            print(f"Skipped cleaning empty image: {img_file}")


def split_dataset(
    image_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)
):
    all_images = [
        img
        for img in glob.glob(os.path.join(image_dir, "*.png"))
        if not is_empty_image(img)
    ]

    train_images, test_val_images = train_test_split(
        all_images, train_size=split_ratio[0], random_state=42
    )
    val_images, test_images = train_test_split(
        test_val_images,
        train_size=split_ratio[1] / (split_ratio[1] + split_ratio[2]),
        random_state=42,
    )

    for img in train_images:
        os.rename(img, os.path.join(train_dir, os.path.basename(img)))
    for img in val_images:
        os.rename(img, os.path.join(val_dir, os.path.basename(img)))
    for img in test_images:
        os.rename(img, os.path.join(test_dir, os.path.basename(img)))

    print(
        f"Split dataset: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test"
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


if __name__ == "__main__":
    main()
