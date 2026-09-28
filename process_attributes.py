import os
import sys
import csv
# pyrefly: ignore [missing-import]
import numpy as np

def main():
    repo_root = os.getcwd()
    cub_dir = os.path.join(repo_root, "datasets/CUB_200_2011")
    
    # Check paths
    images_txt = os.path.join(cub_dir, "images.txt")
    class_labels_txt = os.path.join(cub_dir, "image_class_labels.txt")
    split_txt = os.path.join(cub_dir, "train_test_split.txt")
    attributes_txt = os.path.join(cub_dir, "attributes.txt")
    image_attribs_txt = os.path.join(cub_dir, "attributes/image_attribute_labels.txt")
    
    for path in [images_txt, class_labels_txt, split_txt, attributes_txt, image_attribs_txt]:
        if not os.path.exists(path):
            print(f"[ERROR] Required dataset file not found: {path}")
            sys.exit(1)
            
    # 1. Parse split information
    print("Parsing splits and metadata...")
    img_split = {}
    train_count = 0
    test_count = 0
    with open(split_txt, "r") as f:
        for line in f:
            idx, is_train = map(int, line.strip().split())
            img_split[idx] = "train" if is_train == 1 else "test"
            if is_train == 1:
                train_count += 1
            else:
                test_count += 1
            
    # 2. Parse image paths
    img_paths = {}
    with open(images_txt, "r") as f:
        for line in f:
            idx, path = line.strip().split()
            idx = int(idx)
            img_paths[idx] = path
            
    # 3. Parse class labels (0-indexed internally for mapping, but we save 1-indexed class_id in CSV)
    img_classes = {}
    class_to_imgs = {c: [] for c in range(200)}
    with open(class_labels_txt, "r") as f:
        for line in f:
            img_id, class_id = map(int, line.strip().split())
            class_id = int(class_id)
            img_classes[img_id] = class_id
            # Build training lists for majority voting
            if img_split[img_id] == "train":
                class_to_imgs[class_id - 1].append(img_id)
                
    # 4. Load attribute names
    print("Loading attribute names...")
    attr_names = {}
    with open(attributes_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            idx = int(parts[0])
            name = parts[1]
            attr_names[idx - 1] = name # 0-indexed
            
    # Official 0-indexed 112 selected indices from Koh et al.
    koh_indices = [
        1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 
        51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101, 
        106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 
        153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194, 
        196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 
        238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277, 
        283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311
    ]
    
    # 5. Load raw annotations (is_present, certainty)
    print("Loading raw annotations...")
    num_images = 11788
    num_attributes = 312
    attribute_labels_all = {img_id: [0] * num_attributes for img_id in range(1, num_images + 1)}
    attribute_certainties_all = {img_id: [0] * num_attributes for img_id in range(1, num_images + 1)}
    
    with open(image_attribs_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            img_id = int(parts[0])
            attr_idx = int(parts[1]) - 1
            a = int(parts[2])
            cert = int(parts[3])
            attribute_labels_all[img_id][attr_idx] = a
            attribute_certainties_all[img_id][attr_idx] = cert
            
    # 6. Apply CBM majority voting protocol on training split
    print("Calculating class-level majority voting...")
    class_attr_count = np.zeros((200, num_attributes, 2))
    for img_id in range(1, num_images + 1):
        if img_split[img_id] == "train":
            class_label = img_classes[img_id] - 1
            certainties = attribute_certainties_all[img_id]
            labels = attribute_labels_all[img_id]
            for attr_idx in range(num_attributes):
                a = labels[attr_idx]
                cert = certainties[attr_idx]
                if a == 0 and cert == 1: # not visible
                    continue
                class_attr_count[class_label][attr_idx][a] += 1
                
    class_attr_min_label = np.argmin(class_attr_count, axis=2)
    class_attr_max_label = np.argmax(class_attr_count, axis=2)
    equal_count = np.where(class_attr_min_label == class_attr_max_label)
    class_attr_max_label[equal_count] = 1 # Resolve ties with 1
    
    # 7. Generate and write CSV rows
    output_csv = os.path.join(cub_dir, "processed_attributes.csv")
    print(f"Writing dataset to: {output_csv}")
    
    selected_attr_names = [attr_names[idx] for idx in koh_indices]
    header = ["image_id", "image_path", "class_id", "split"] + selected_attr_names
    
    total_rows = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for img_id in sorted(img_paths.keys()):
            class_id = img_classes[img_id]
            class_label = class_id - 1
            img_path = img_paths[img_id]
            split = img_split[img_id]
            
            # Get 112 class-level majority voted binary attributes
            class_attrs = class_attr_max_label[class_label, koh_indices]
            
            row = [img_id, img_path, class_id, split] + list(class_attrs)
            writer.writerow(row)
            total_rows += 1
            
    print("Dataset saved successfully.")
    
    # 8. Verification and Printing required stats
    total_attributes = len(selected_attr_names)
    
    print("\n--- Statistics ---")
    print(f"Total images     : {total_rows}")
    print(f"Train images     : {train_count}")
    print(f"Test images      : {test_count}")
    print(f"Total attributes : {total_attributes}")
    
    # Verify that every image has exactly 112 concept labels
    # Verify by reading back and checking columns
    cols_count = len(header)
    concept_cols_count = cols_count - 4
    assert concept_cols_count == 112, f"Error: Header has {concept_cols_count} concept columns instead of 112!"
    assert total_rows == 11788, f"Error: Total rows is {total_rows} instead of 11788!"
    print("Verification passed: Every image row has exactly 112 concept labels.")

if __name__ == "__main__":
    main()
