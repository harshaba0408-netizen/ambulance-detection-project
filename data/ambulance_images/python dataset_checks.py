# pair_and_validate.py
import os, glob, itertools
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(ROOT, "data.yaml")

def load_yaml(p):
    import yaml
    return yaml.safe_load(open(p))

def list_files(img_dir):
    imgs = sorted([os.path.basename(p) for p in glob.glob(os.path.join(img_dir, "*.*")) if p.lower().endswith(('.jpg','.jpeg','.png'))])
    return imgs

def corresponding_label_path(img_path):
    # if img is .../images/name.jpg -> label under same parent replacing 'images'->'labels' and .jpg->.txt
    if os.sep + "images" + os.sep in img_path:
        return img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit('.',1)[0] + '.txt'
    else:
        # fallback: sibling labels folder
        return os.path.join(os.path.dirname(img_path), '..', 'labels', os.path.basename(img_path).rsplit('.',1)[0] + '.txt')

def scan_split(split_name, img_dir, nc):
    print(f"\n=== Checking split: {split_name} ===")
    imgs = sorted([p for p in glob.glob(os.path.join(img_dir, "*.*")) if p.lower().endswith(('.jpg','.jpeg','.png'))])
    print("Images found:", len(imgs))
    expected_labels = []
    missing_labels = []
    for img in imgs:
        lbl = corresponding_label_path(img)
        if not os.path.exists(lbl):
            missing_labels.append(os.path.basename(img))
        else:
            expected_labels.append(lbl)
    # find label files present in labels dir
    labels_dir = img_dir.replace(os.sep + "images", os.sep + "labels") if "images" in img_dir else os.path.join(os.path.dirname(img_dir), "labels")
    found_label_files = sorted([p for p in glob.glob(os.path.join(labels_dir, "*.txt"))]) if os.path.exists(labels_dir) else []
    # filter out README and roboflow text files that are not per-image labels
    filtered_labels = [p for p in found_label_files if not os.path.basename(p).lower().startswith('readme')]
    # unmatched labels: those in labels folder that don't match any image (by basename)
    img_basenames = set([os.path.splitext(os.path.basename(x))[0] for x in imgs])
    unmatched_labels = [p for p in filtered_labels if os.path.splitext(os.path.basename(p))[0] not in img_basenames]
    print("Label files found (filtered):", len(filtered_labels))
    print("Missing label files for images:", len(missing_labels))
    if missing_labels:
        print(" Examples of images missing labels (first 10):", missing_labels[:10])
    print("Labels with no matching image:", len(unmatched_labels))
    if unmatched_labels:
        print(" Examples of unmatched label files (first 10):")
        for p in unmatched_labels[:10]:
            print(" ", p)
    # quick format check for up to 50 label files
    problems = []
    for lbl in filtered_labels[:50]:
        with open(lbl,'r') as f:
            lines = [l.strip() for l in f if l.strip()]
            for i,ln in enumerate(lines,1):
                parts = ln.split()
                if len(parts) != 5:
                    problems.append((lbl, i, "bad_len", parts))
                    continue
                try:
                    cls = int(float(parts[0]))
                    if cls < 0 or cls >= nc:
                        problems.append((lbl,i,"bad_class",cls))
                    vals = list(map(float, parts[1:]))
                    if any(v < 0 or v > 1 for v in vals):
                        problems.append((lbl,i,"out_of_range",vals))
                except Exception as e:
                    problems.append((lbl,i,"parse_error",str(e)))
    print("Checked up to 50 label files for format problems. Problems found:", len(problems))
    for p in problems[:10]:
        print(" ", p)
    return {
        "n_images": len(imgs),
        "n_filtered_label_files": len(filtered_labels),
        "n_missing_labels": len(missing_labels),
        "missing_examples": missing_labels[:10],
        "n_unmatched_labels": len(unmatched_labels),
        "unmatched_examples": unmatched_labels[:10],
        "format_problems_count": len(problems),
        "format_problems_examples": problems[:10]
    }

def main():
    data = load_yaml(DATA_YAML)
    nc = int(data['nc'])
    splits = {
        'train': data['train'],
        'val': data.get('val') or data.get('valid'),
        'test': data.get('test')
    }
    report = {}
    for k,v in splits.items():
        if not v:
            print(f"\nSkipping {k} (not present in data.yaml)")
            continue
        report[k] = scan_split(k, v, nc)
    print("\n=== SUMMARY ===")
    for k in report:
        print(k, report[k])
    print("\nIf you have many 'missing_labels', either move label files into the expected labels/ folders or re-download the YOLO export from Roboflow and place it under", ROOT)

if __name__ == '__main__':
    main()
