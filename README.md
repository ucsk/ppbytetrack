## Introduction

Our method is in the tracking-by-detection paradigm, the detector uses the anchor-free model PP-YOLOE, and the tracker uses ByteTrack, so our method can be optimized separately and only needs to train the detector for cell detection.

We only used the CTMC-v1 dataset as the training of the cell detection model, but used COCO pretrained weights.

We take the first 25% of the time series of each sequence in the CTMC-v1 dataset as the validation set and the last 75% as the training set. In particular, for the training of the detector, we use frame sampling to obtain 9191 (6911+2280) images on the divided dataset for training and validation (this greatly improves the training efficiency of the cell detector). For the local evaluation of the tracker, we use the full validation set.

We believe that there are still many areas for improvement in this approach, but the simple, unpretentious steps further demonstrate the effectiveness of our ideas.

## Prepare the environment

We use Python-3.7 and PaddlePaddle-2.2.2 framework to implement multi-cell tracking, if you are interested in this project, please install [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/2.2/install/index_en.html) first.

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
# For single-card training,
# please adapt the learning-rate and batch-size first.
!python tools/train.py \
    -c configs/ppyoloe/ppyoloe_crn_l_300e.yml \
    --vdl_log_dir=vdl_log/ppyoloe_crn_l_300e \
    --eval --amp
```

## Model evaluation

We choose the highest weight of COCO-mAP on the validation set (2280 images) as the final model.

```bash
!python tools/eval.py \
    -c configs/ppyoloe/ppyoloe_crn_l_300e.yml \
    -o weights=output/ppyoloe_crn_l_300e/best_model.pdparams \
    --classwise
```

## Model prediction

For the trained detector weights, we configure the path of the weights on the tracker file (`configs/mot/bytetrack/*.yml`, `det_weights:`).

When making predictions on the CTMC-v1 test set, be careful to modify the path to the dataset (`configs/mot/bytetrack/_base_/ctmc_test.yml`, `dataset_dir`).

```bash
# Generate the MOT-Challenge format file of the validation set.
!python tools/eval_mot.py \
    -c configs/mot/bytetrack/bytetrack_ppyoloe_val.yml \
    --output_dir=output/val/bytetrack_ppyoloe \
    --scaled=True
```

```bash
# Generate the MOT-Challenge format file of the test set.
!python tools/eval_mot.py \
    -c configs/mot/bytetrack/bytetrack_ppyoloe_test.yml \
    --output_dir=output/test/bytetrack_ppyoloe \
    --scaled=True
```

When it was time to submit it to the server, we copied the GT annotation files in the training set and mixed the prediction results of the test set, and passed the server's review (because the server needs 86 cell sequence files, which includes the training set and test set).

## Valid-set evaluation

After completing the generation of the cell tracking result file, we compare the result file of the validation set with the GT annotation to verify the local reliability of the method. Here, the parameters of the tracker can be slightly adjusted.

```bash
!python tools/eval_ctmc.py
```

## Reference

-   [CTMC: Cell Tracking with Mitosis Detection Dataset Challenge](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w57/Anjum_CTMC_Cell_Tracking_With_Mitosis_Detection_Dataset_Challenge_CVPRW_2020_paper.pdf)

-   [PP-YOLOE: An evolved version of YOLO](https://arxiv.org/abs/2203.16250)
-   [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
-   [PaddleDetection: Object detection and instance segmentation toolkit based on PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection)

