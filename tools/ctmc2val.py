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


def mot2val(seq_dir: str, save_dir: str):
    name, seqLength, imWidth, imHeight = get_LengthWidthHeight(seq_dir)
    img_dir = os.path.join(seq_dir, 'img1')
    img_name_list = sorted(os.listdir(img_dir))
    assert len(img_name_list) == seqLength
    split_len = int(len(img_name_list) * 0.25)
    val_list = img_name_list[:split_len]
    end_frame = int(val_list[-1].split('.')[0])

    img_save_dir = os.path.join(save_dir, 'img1')
    gt_save_dir = os.path.join(save_dir, 'gt')
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(gt_save_dir, exist_ok=True)

    for img_name in tqdm(val_list, desc=f'{seq_dir} -> {img_save_dir}'):
        shutil.copy(src=os.path.join(img_dir, img_name),
                    dst=os.path.join(img_save_dir, img_name))

    with open(os.path.join(seq_dir, 'gt', 'gt.txt'),
              mode='r', encoding='utf-8') as rf:
        gt_lines = rf.readlines()

    with open(os.path.join(gt_save_dir, 'gt.txt'),
              mode='w', encoding='utf-8') as wf:
        for line in gt_lines:
            frame, uid, x, y, w, h, score, _, _ = line.strip().split(',')
            if int(frame) <= end_frame:
                if int(w) < 3 or int(h) < 3:
                    continue
                wf.write(f'{int(frame)},{uid},{x},{y},{w},{h},{score},-1,-1\n')


@click.command()
@click.option('--src_root', type=click.Path(exists=True), prompt='Training set folder path for CTMCCVPR20')
@click.option('--dst_root', type=click.Path(), prompt='The path to save the MOT validation set')
def main(src_root, dst_root):
    src_root = os.path.normpath(src_root)
    dst_root = os.path.normpath(dst_root)

    dataset_list = os.listdir(src_root)
    for seq_name in dataset_list:
        mot2val(seq_dir=os.path.join(src_root, seq_name),
                save_dir=os.path.join(dst_root, seq_name))


if __name__ == '__main__':
    main()
