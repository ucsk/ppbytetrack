## Prepare the environment

```bash
%cd CTMC
!pip install -r requirements.txt
!python setup.py install
```

## Prepare the dataset

```bash
!python tools/ctmc2voc.py \
    --src_root ../data/CTMCCVPR20/train \
    --dst_root dataset/voc

!python tools/voc2coco.py dataset/voc/train/Annotations dataset/coco/train.json
!python tools/voc2coco.py dataset/voc/val/Annotations dataset/coco/val.json

!cp -r dataset/voc/train/JPEGImages dataset/coco/train
!cp -r dataset/voc/val/JPEGImages dataset/coco/val
```


```bash
!python tools/ctmc2val.py \
    --src_root ../data/CTMCCVPR20/train \
    --dst_root dataset/mot/val
```

## Model training

```bash
!python -m paddle.distributed.launch \
    --gpus 0,1,2,3 tools/train.py \
    -c configs/ppyoloe/ppyoloe_crn_l_300e.yml \
    --vdl_log_dir=vdl_log/ppyoloe_crn_l_300e \
    --eval --amp --fleet
```

```bash
!python tools/train.py \
    -c configs/ppyoloe/ppyoloe_crn_l_300e.yml \
    --vdl_log_dir=vdl_log/ppyoloe_crn_l_300e \
    --eval --amp
```

## Model evaluation

```bash
!python tools/eval.py \
    -c configs/ppyoloe/ppyoloe_crn_l_300e.yml \
    -o weights=output/ppyoloe_crn_l_300e/best_model.pdparams \
    --classwise
```

## Model prediction

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

```bash
!python tools/eval_ctmc.py
```

