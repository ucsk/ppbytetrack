## Prepare the environment

We use Python-3.7 and PaddlePaddle-2.2.2 framework to implement multi-cell tracking, if you are interested in this, please install [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en /2.2/install/index_en.html).

The next code or command comes from the Jupyter environment.

```bash
!git clone https://github.com/ucsk/ppbytetrack.git
%cd ppbytetrack
!pip install -r requirements.txt
!python setup.py install
```

## Prepare the dataset

We first convert the MOT format data set to COCO format, and the converted COCO format data is used as the training set of the cell detector (in the process of format conversion, we also converted to VOC format by the way, you can ignore this VOC folder).

```bash
!python tools/ctmc2voc.py \
    --src_root ../data/CTMCCVPR20/train \
    --dst_root dataset/voc

!python tools/voc2coco.py dataset/voc/train/Annotations dataset/coco/train.json
!python tools/voc2coco.py dataset/voc/val/Annotations dataset/coco/val.json

!cp -r dataset/voc/train/JPEGImages dataset/coco/train
!cp -r dataset/voc/val/JPEGImages dataset/coco/val
```

This part is to extract the validation set from the CTMC dataset to prove the effectiveness of the tracker on the validation set.


```bash
!python tools/ctmc2val.py \
    --src_root ../data/CTMCCVPR20/train \
    --dst_root dataset/mot/val
```

## Model training

We use a 4 card Tesla V100 (32 GB) for training. You can modify the batch size (default 20) in `configs/ppyoloe/_base_/ppyoloe_reader`, at the same time, please adjust the learning rate (default 0.00625) in `configs/ppyoloe/_base_/optimizer_300e`, the modification formula is: $lr_{new} = lr_{default} * (batch\ size_{new} * GPU\ number_{new}) / (batch\ size_{default} * GPU\ number_{default}) $.

```bash
# 4 card training (default)
!python -m paddle.distributed.launch \
    --gpus 0,1,2,3 tools/train.py \
    -c configs/ppyoloe/ppyoloe_crn_l_300e.yml \
    --vdl_log_dir=vdl_log/ppyoloe_crn_l_300e \
    --eval --amp --fleet
```

```bash
# single card training
!python tools/train.py \
    -c configs/ppyoloe/ppyoloe_crn_l_300e.yml \
    --vdl_log_dir=vdl_log/ppyoloe_crn_l_300e \
    --eval --amp
```

## Model evaluation

The validation set optimal model was selected for cell detection.

```bash
!python tools/eval.py \
    -c configs/ppyoloe/ppyoloe_crn_l_300e.yml \
    -o weights=output/ppyoloe_crn_l_300e/best_model.pdparams \
    --classwise
```

## Model prediction

Combine detector and tracker to achieve multi-cell tracking and generate MOT-Challenge submission file (test set part).

```bash
!python tools/eval_mot.py \
    -c configs/mot/bytetrack/bytetrack_ppyoloe_val.yml \
    --output_dir=output/val/bytetrack_ppyoloe \
    --scaled=True
```

```bash
!python tools/eval_mot.py \
    -c configs/mot/bytetrack/bytetrack_ppyoloe_test.yml \
    --output_dir=output/test/bytetrack_ppyoloe \
    --scaled=True
```

## Valid-set evaluation

We have used the time-series top 25% of the CTMC dataset for tracker parameter optimization.

```bash
!python tools/eval_ctmc.py
```

## Reference

-   [CTMC: Cell Tracking with Mitosis Detection Dataset Challenge](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w57/Anjum_CTMC_Cell_Tracking_With_Mitosis_Detection_Dataset_Challenge_CVPRW_2020_paper.pdf)

-   [PP-YOLOE: An evolved version of YOLO](https://arxiv.org/abs/2203.16250)
-   [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
-   [PaddleDetection: Object detection and instance segmentation toolkit based on PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection)

