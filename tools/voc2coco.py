import glob
import json
import os
import xml.etree.ElementTree as ET


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_files, json_file):
    json_dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "supercategory": "component",
                "id": 1,
                "name": "cell"}
        ]
    }

    annotation_id = 1
    for index, xml_file in enumerate(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))

        image_id = index + 1
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        json_dict["images"].append(
            {
                "file_name": filename,
                "height": height,
                "width": width,
                "id": image_id,
            }
        )

        for obj in get(root, "object"):
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text)
            ymin = int(get_and_check(bndbox, "ymin", 1).text)
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            W = xmax - xmin
            H = ymax - ymin
            json_dict["annotations"].append(
                {
                    "image_id": image_id,
                    "bbox": [xmin, ymin, W, H],
                    "area": W * H,
                    "iscrowd": 0,
                    "category_id": 1,
                    "id": annotation_id,
                    "segmentation": [],
                }
            )
            annotation_id += 1

    print(f'annotation_nums/image_nums: {annotation_id - 1}/{index + 1}')
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotation to COCO format."
    )
    parser.add_argument("xml_dir", help="Directory path to xml files.", type=str)
    parser.add_argument("json_file", help="Output COCO format json file.", type=str)
    args = parser.parse_args()
    xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))

    # If you want to do train/test split, you can pass a subset of xml files to convert function.
    print("Number of xml files: {}".format(len(xml_files)))
    convert(xml_files, args.json_file)
    print("Success: {}".format(args.json_file))
