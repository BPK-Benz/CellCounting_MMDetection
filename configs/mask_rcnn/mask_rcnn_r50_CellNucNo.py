_base_ = './mask_rcnn_r50_fpn_2x_coco.py'

# 1. dataset settings
# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('object', )

img_scale = (int(1360/4*3), int(1024/4*3))
# img_scale = (int(1360/2), int(1024/2))
img_norm_cfg = dict(
    mean=[25.526, 0.386, 52.850], std=[53.347, 9.402, 53.172], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal','vertical'] ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels','gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor')),

]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal','vertical'] ),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

base = "/share/NAS_DATASETS/Benz_Cell/cellLabel-main/"
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
# 2. model settings
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        depth=50,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50'
                      )
    ),

    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes)) 
    ),

    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.5),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5))                     
)
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)

load_from="pretrained_models/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth"
workflow = [('train', 1),('val', 1)]
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook'),])

device = 'cuda'
runner = dict(type='EpochBasedRunner', max_epochs=30)
evaluation = dict(interval=1,metric='bbox', save_best='bbox_mAP')
work_dir='./work_dirs/New_OCT/MaskRCNN_R50_CellNuc'