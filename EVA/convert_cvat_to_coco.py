import json
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path

def parse_points(points_str):
    points = []
    for pt in points_str.split(';'):
        x, y = map(float, pt.split(','))
        points.extend([x, y])
    return points

def get_bbox(points):
    xs = points[0::2]
    ys = points[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return [xmin, ymin, xmax - xmin, ymax - ymin]

def convert():
    base_dir = Path('/home/orchvate/CAR_TRAIN1/EVA')
    yaml_path = base_dir / 'unified_label_mapping.yaml'
    xml_path = base_dir / 'GROUND/annotations.xml'
    out_path = base_dir / 'GROUND/coco_annotations.json'

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    final_cats = config['final_categories']
    cat_name_to_id = {v: k for k, v in final_cats.items()}
    cvat_config = config['source_1_cvat_xml']
    direct_map = cvat_config['direct_mapping']
    pos_map = cvat_config['position_split_mapping']

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": k, "name": v} for k, v in final_cats.items()]
    }

    tree = ET.parse(xml_path)
    root = tree.getroot()

    ann_id = 1
    for img_elem in root.findall('.//image'):
        img_id = int(img_elem.get('id'))
        file_name = Path(img_elem.get('name')).name  # Extract just the filename
        width = int(img_elem.get('width'))
        height = int(img_elem.get('height'))
        
        coco['images'].append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        for poly in img_elem.findall('polygon'):
            raw_label = poly.get('label')
            position = None
            
            for attr in poly.findall('attribute'):
                if attr.get('name') == 'Position':
                    position = attr.text.strip() if attr.text else None
                    break
            
            mapped_name = None
            if raw_label in direct_map:
                mapped_name = direct_map[raw_label]
            elif raw_label in pos_map:
                if position in pos_map[raw_label]:
                    mapped_name = pos_map[raw_label][position]
                else:
                    print(f"Warning: Unexpected position '{position}' for label '{raw_label}' in image {file_name}")
            else:
                print(f"Warning: Unknown label '{raw_label}' in image {file_name}")

            if not mapped_name:
                continue
                
            if mapped_name not in cat_name_to_id:
                print(f"Warning: Mapped name '{mapped_name}' not in final categories.")
                continue

            points_str = poly.get('points')
            segmentation = parse_points(points_str)
            bbox = get_bbox(segmentation)
            area = bbox[2] * bbox[3]  # simple bbox area for COCO

            coco['annotations'].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_name_to_id[mapped_name],
                "segmentation": [segmentation],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1

    with open(out_path, 'w') as f:
        json.dump(coco, f, indent=2)
        
    print(f"Converted {len(coco['images'])} images and {len(coco['annotations'])} annotations.")
    print(f"Output saved to {out_path}")

if __name__ == '__main__':
    convert()
