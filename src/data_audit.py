from pathlib import Path
from collections import Counter
from PIL import Image
from statistics import mean, median
import matplotlib.pyplot as plt
import hashlib

DATASET_DIR = Path("dataset")
SPLITS = ["train", "test", "unclean"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}
MIN_IMAGE_SIZE = 32


def count_images_in_class(class_dir):
    """Count image files directly inside a class directory."""
    return sum(
        1
        for file_path in class_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def audit_split(split_dir):
    """Audit images in one dataset split."""
    class_counts = Counter()

    invalid_files = []
    too_small_files = []
    widths = []
    heights = []
    image_formats = Counter()

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for file_path in sorted(class_dir.iterdir()):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            result = inspect_image(file_path)

            if not result["valid"]:
                invalid_files.append(file_path)
                continue

            class_counts[class_dir.name] += 1

            image_formats[result["format"]] += 1

            widths.append(result["width"])
            heights.append(result["height"])

            if result["too_small"]:
                too_small_files.append(file_path)

    return {
        "class_counts": class_counts,
        "invalid_files": invalid_files,
        "too_small_files": too_small_files,
        "widths": widths,
        "heights": heights,
        "image_formats": image_formats,
    }


def inspect_image(file_path):
    """Inspect one image file for readability and dimensions."""
    try:
        with Image.open(file_path) as image:
            width, height = image.size

            is_too_small = width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE

            return {
                "valid": True,
                "format": image.format,
                "width": width,
                "height": height,
                "too_small": is_too_small,
            }
    except Exception as error:
        return {
            "valid": True,
            "format": None,
            "width": None,
            "height": None,
            "too_small": False,
            "error": str(error),
        }


def get_image_hash(file_path):
    """Create a SHA-256 hash from image pixel content."""
    try:
        with Image.open(file_path) as image:
            image = image.convert("RGB")

            hash_object = hashlib.sha256()
            hash_object.update(image.size.__repr__().encode())
            hash_object.update(image.tobytes())

            return hash_object.hexdigest()
    except Exception:
        return None


def find_duplicates(dataset_dir):
    """Find duplicate images across train, test, and unclean."""

    hash_map = {}

    for split_name in SPLITS:
        splt_dir = dataset_dir / split_name

        for class_dir in sorted(splt_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for file_path in sorted(class_dir.iterdir()):
                if not file_path.is_file():
                    continue

                image_hash = get_image_hash(file_path)

                if image_hash is None:
                    continue

                image_info = {
                    "split": split_name,
                    "class": class_dir.name,
                    "path": str(file_path),
                }

                hash_map.setdefault(image_hash, []).append(image_info)

    duplicate_groups = {
        image_hash: files for image_hash, files in hash_map.items() if len(files) > 1
    }

    return duplicate_groups


def report_duplicates(dataset_dir):
    """Report duplicate groups and classify their relationships."""

    duplicate_groups = find_duplicates(dataset_dir)

    print("\n--- DUPLICATE ANALYSIS ---")

    if not duplicate_groups:
        print("No duplicate images found.")
        return

    print(f"Duplicate groups found: {len(duplicate_groups)}")

    category_counts = {
        "train-test": 0,
        "train-unclean": 0,
        "test-unclean": 0,
        "label-conflict": 0,
        "other": 0,
    }

    for group_number, (image_hash, files) in enumerate(
        duplicate_groups.items(), start=1
    ):
        splits = {item["split"] for item in files}
        classes = {item["class"] for item in files}

        if "train" in splits and "test" in splits:
            category = "train-test"

        elif "train" in splits and "unclean" in splits and len(classes) > 1:
            category = "label-conflict"

        elif "train" in splits and "unclean" in splits:
            category = "train-unclean"

        elif "test" in splits and "unclean" in splits and len(classes) > 1:
            category = "label-conflict"

        elif "test" in splits and "unclean" in splits:
            category = "test-unclean"

        else:
            category = "other"

        category_counts[category] += 1

        print(f"\nGroup {group_number}")
        print(f"Category: {category}")
        print(f"Hash: {image_hash}")

        for file_info in files:
            print(
                f"  [{file_info['split']}] {file_info['class']} -> {file_info['path']}"
            )

    print("\nDuplicate Summary")

    for category, count in category_counts.items():
        print(f"{category:16}: {count}")


def show_sample_images(split_dir, samples_per_class=2):
    """Display labelled sample images from each class."""
    class_dirs = [
        directory for directory in sorted(split_dir.iterdir()) if directory.is_dir()
    ]

    images = []

    for class_dir in class_dirs:
        image_files = [
            file_path
            for file_path in sorted(class_dir.iterdir())
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for file_path in image_files[:samples_per_class]:
            images.append((file_path, class_dir.name))

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    axes = axes.flatten()

    for ax, (file_path, class_name) in zip(axes, images):
        with Image.open(file_path) as image:
            ax.imshow(image)

        ax.set_title(class_name)
        ax.axis("off")

    for ax in axes[len(images) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("Traffic Vehicle Dataset Audit")
    print("=" * 60)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR.resolve()}")

    print(f"\nDataset path: {DATASET_DIR.resolve()}")

    for split_name in SPLITS:
        split_dir = DATASET_DIR / split_name

        if not split_dir.exists():
            print(f"\n[WARNING] Missing split: {split_name}")
            continue

        audit_result = audit_split(split_dir)
        class_counts = audit_result["class_counts"]

        print(f"\n--- {split_name.upper()} ---")

        for class_name, count in class_counts.items():
            print(f"{class_name:12s}: {count}")

        print(f"Total        : {sum(class_counts.values())}")

        print(f"Invalid files: {len(audit_result['invalid_files'])}")
        print(f"Too small    : {len(audit_result['too_small_files'])}")

        print("Formats      :", dict(audit_result["image_formats"]))

        widths = audit_result["widths"]
        heights = audit_result["heights"]
        print(
            f"Width        : min={min(widths)}, "
            f"max={max(widths)}, "
            f"mean={mean(widths):.1f}, "
            f"median={median(widths):.1f}"
        )

        print(
            f"Height       : min={min(heights)}, "
            f"max={max(heights)}, "
            f"mean={mean(heights):.1f}, "
            f"median={median(heights):.1f}"
        )

    train_dir = DATASET_DIR / "train"
    # show_sample_images(train_dir)

    report_duplicates(DATASET_DIR)


if __name__ == "__main__":
    main()
