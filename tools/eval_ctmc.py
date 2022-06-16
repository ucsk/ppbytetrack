import os

import motmetrics as mm
from motmetrics.io import Format

metrics = list(mm.metrics.motchallenge_metrics)
val_list = os.listdir('output/val')

for val_dir in val_list:
    seqs = os.listdir('dataset/mot/val')
    accs = []
    names = []
    for seq in seqs:
        gt_file = f"dataset/mot/val/{seq}/gt/gt.txt"
        ts_file = f"output/val/{val_dir}/mot_results/{seq}.txt"

        gt = mm.io.loadtxt(gt_file, fmt=Format.MOT15_2D, min_confidence=0.5)
        dt = mm.io.loadtxt(ts_file, fmt=Format.MOT15_2D)

        accs.append(mm.utils.compare_to_groundtruth(gt, dt, 'iou', distth=0.5))
        names.append(seq)

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)
    summary = summary.loc[['OVERALL']]
    df = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(val_dir)
    print(df)
    print()
