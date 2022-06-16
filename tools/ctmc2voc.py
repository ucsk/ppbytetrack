import codecs
import os
import shutil

import click
from tqdm import tqdm


def get_LengthWidthHeight(_seq_dir):
    with open(os.path.join(_seq_dir, 'seqinfo.ini'),
              mode='r',
              encoding='utf-8') as f:
        seq_info = f.readlines()
    name = str(seq_info[1][5:].strip())
    seqLenth = int(seq_info[4][10:])
    imWidth = int(seq_info[5][8:])
    imHeight = int(seq_info[6][9:])
    return name, seqLenth, imWidth, imHeight


def generate_xml(gt_path, xml_save_dir,
                 filename, save_name,
                 width, height):
    with open(gt_path, mode='r', encoding='utf-8') as f:
        gt_lines = f.readlines()

    current_gt_bbox = []
    for line in gt_lines:
        line = line.strip().split(',')
        if int(line[0]) != int(filename):
            continue

        x_min, y_min, w, h = int(line[2]), int(line[3]), int(line[4]), int(line[5])
        if w < 3 or h < 3:
            continue

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width - 1, x_min + w)
        y_max = min(height - 1, y_min + h)

        current_gt_bbox.append((x_min, y_min, x_max, y_max))

    with codecs.open(os.path.join(xml_save_dir, f'{save_name}.xml'),
                     mode='w',
                     encoding='utf-8') as xml:
        xml.write('<?xml version="1.0" encoding="UTF-8"?>')
        xml.write('<annotation>')
        xml.write('<folder>JPEGImages</folder>')
        xml.write(f'<filename>{save_name}.jpg</filename><size><width>{width}</width><height>{height}</height>')
        xml.write('<depth>3</depth></size>')
        for x_min, y_min, x_max, y_max in current_gt_bbox:
            xml.write('<object>')
            xml.write('<name>cell</name>')
            xml.write('<pose>Unspecified</pose>')
            xml.write('<truncated>0</truncated>')
            xml.write('<difficult>0</difficult>')
            xml.write('<bndbox>')
            xml.write(f'<xmin>{x_min}</xmin><ymin>{y_min}</ymin><xmax>{x_max}</xmax><ymax>{y_max}</ymax>')
            xml.write('</bndbox>')
            xml.write('</object>')
        xml.write('</annotation>\n')


def get_frame_offset(dataset_name: str):
    dataset_name = dataset_name.split('-run')[0]
    counts = {
        "3T3": 5,
        "A-10": 4,
        "A-549": 1,
        "APM": 3,
        "BPAE": 4,
        "CRE-BAG2": 2,
        "CV-1": 2,
        "LLC-MK2": 5,
        "MDBK": 5,
        "MDOK": 5,
        "OK": 4,
        "PL1Ut": 3,
        "RK-13": 2,
        "U2O-S": 2
    }
    return 5 + counts[dataset_name]


def split2voc(save_root_dir: str, save_dir_name: str, offset: int,
              file_name_list: list, seq_dir: str, seq_name: str, img_dir: str,
              imWidth: int, imHeight: int):
    save_root_dir = os.path.normpath(save_root_dir)
    img_save_dir = os.path.join(save_root_dir, save_dir_name, 'JPEGImages')
    xml_save_dir = os.path.join(save_root_dir, save_dir_name, 'Annotations')
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(xml_save_dir, exist_ok=True)

    cnt = 0
    for img_name in tqdm(file_name_list, desc=seq_dir):
        file_name = str(img_name.split('.')[0])
        if int(file_name) % offset != 0:
            continue

        save_name = f'{seq_name}_{file_name}'
        shutil.copy(src=os.path.join(img_dir, img_name),
                    dst=os.path.join(img_save_dir, f'{save_name}.jpg'))

        generate_xml(gt_path=os.path.join(seq_dir, 'gt', 'gt.txt'),
                     xml_save_dir=xml_save_dir,
                     filename=file_name,
                     save_name=save_name,
                     width=imWidth,
                     height=imHeight)

        cnt += 1

    return cnt


def mot2voc(seq_dir: str, save_dir: str):
    seq_dir = os.path.normpath(seq_dir)
    save_dir = os.path.normpath(save_dir)

    seq_name, seqLength, imWidth, imHeight = get_LengthWidthHeight(seq_dir)

    img_dir = os.path.join(seq_dir, 'img1')
    img_name_list = sorted(os.listdir(img_dir))
    assert len(img_name_list) == seqLength

    split_len = int(len(img_name_list) * 0.25)
    train_list = img_name_list[split_len:]
    val_list = img_name_list[:split_len]

    train_cnt = split2voc(save_root_dir=save_dir,
                          save_dir_name='train',
                          offset=get_frame_offset(seq_name),
                          file_name_list=train_list,
                          seq_dir=seq_dir,
                          seq_name=seq_name,
                          img_dir=img_dir,
                          imWidth=imWidth,
                          imHeight=imHeight)

    val_cnt = split2voc(save_root_dir=save_dir,
                        save_dir_name='val',
                        offset=get_frame_offset(seq_name),
                        file_name_list=val_list,
                        seq_dir=seq_dir,
                        seq_name=seq_name,
                        img_dir=img_dir,
                        imWidth=imWidth,
                        imHeight=imHeight)

    return seq_name.split('-run')[0], train_cnt, val_cnt


@click.command()
@click.option('--src_root', type=click.Path(exists=True), prompt='Training set folder path for CTMCCVPR20')
@click.option('--dst_root', type=click.Path(), prompt='Save path of VOC format data')
def main(src_root, dst_root):
    src_root = os.path.normpath(src_root)
    dst_root = os.path.normpath(dst_root)

    results = {}
    dataset_list = os.listdir(src_root)
    for seq_name in dataset_list:
        name, train_cnt, val_cnt = mot2voc(seq_dir=os.path.join(src_root, seq_name),
                                           save_dir=dst_root)
        if results.get(name) is None:
            results[name] = [0, 0]
        results[name][0] += train_cnt
        results[name][1] += val_cnt

    print(results)


if __name__ == '__main__':
    main()
