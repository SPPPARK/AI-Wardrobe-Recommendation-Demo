import os

DATA_DIR = "data/images"
LABEL_FILES = [
    "data/labels/texture/fabric_ann.txt",
    "data/labels/texture/pattern_ann.txt",
    "data/labels/shape/shape_anno_all.txt"
]
TARGET_LABEL = (1, 1, 7)  # Type + Color + Pattern（Edit）

def load_labels(files):
    label_dict = {}
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                img, l1, l2, l3 = parts
                if img not in label_dict:
                    label_dict[img] = []
                label_dict[img].append((int(l1), int(l2), int(l3)))
    return label_dict

def filter_by_label(label_dict, target):
    matches = []
    for img, labels in label_dict.items():
        for triplet in labels:
            if triplet == target:
                matches.append(img)
                break
    return matches

if __name__ == "__main__":
    print("📦 Loading Labels...")
    label_dict = load_labels(LABEL_FILES)

    print("🎯 Label Matching...:")
    matched_images = filter_by_label(label_dict, TARGET_LABEL)

    print(f"\n✅ Find {len(matched_images)} Image, Show Top 10:\n")
    for img in matched_images[:10]:
        print("-", img)
