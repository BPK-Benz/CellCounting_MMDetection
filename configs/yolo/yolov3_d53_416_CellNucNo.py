_base_ = './yolov3_d53_mstrain-608_273e_coco.py'


# 1. dataset settings
# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('object', )

img_scale = (int(1360/4*3), int(1024/4*3))
img_norm_cfg = dict(
    mean=[25.526, 0.386, 52.850], std=[53.347, 9.402, 53.172], to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (416, 416)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5 ),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

base = "/share/NAS/Benz_Cell/cellLabel-main/"
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file= base+'Coco_File/Cell_TrainNuc_April.json',
        img_prefix= base,
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file= base+'Coco_File/Cell_TestNuc_April.json',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file= base+'Coco_File/Cell_TestNuc_April.json',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    )
)

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.8, weight_decay=0.0005)
# optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)

# 2. model settings
# We also need to change the num_classes in head to match the dataset's annotation
# https://github.com/open-mmlab/mmdetection/blob/1376e77e6ecbaad609f6003725158de24ed42e84/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')
        ),

    bbox_head=dict(
        num_classes=len(classes),
        ),
        
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)
        ),

    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=300
        )
    )



load_from="pretrained_models/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth"
resume_from = None
workflow = [('train', 1),('val', 1)]
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook'),])

device = 'cuda'
runner = dict(type='EpochBasedRunner', max_epochs=30)
evaluation = dict(interval=1,metric='bbox', save_best='bbox_mAP')
work_dir='./work_dirs/New_OCT/YoloV3_DarkNet_CellNuc'