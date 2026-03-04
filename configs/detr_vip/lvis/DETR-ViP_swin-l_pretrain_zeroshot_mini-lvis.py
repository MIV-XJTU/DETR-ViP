_base_ = '../DETR-ViP_swin-l_pretrain_coco.py'
backend_args = None

model = dict(
    bbox_head=dict(num_classes=1204),
    test_cfg=dict(
        max_per_img=300,
        chunked_size=40,
)
)

dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomSamplingNegPosToList', padding=True, padding_len=365),
    dict(type='MapTextToEmbedding'),
    dict(type='RandomSamplingVisualPrompt', select_mode='training'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'text_prompt_labels', 'dataset_mode', 'text_prompts', 
                   'visual_prompts', 'visual_prompt_labels', 'visual_prompt_types'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=backend_args,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MapTextToEmbedding', text_cache_file='cache/vocabulary/lvis_vocabulary.pkl'),
    dict(type='RandomSamplingVisualPrompt', select_mode='test'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'text_prompt_labels', 'text_prompts',
                   'visual_prompts', 'visual_prompt_labels', 'visual_prompt_types'))
]

support_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=backend_args,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomSamplingVisualPrompt', select_mode='support'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'text_prompt_labels', 'visual_prompts', 'visual_prompt_labels', 'visual_prompt_types'))
]

lvis_od_dataset = dict(
    type='ODVGStableDataset',
    data_root=data_root,
    ann_file='annotations/lvis_v1_train_od.json',
    label_map_file='annotations/lvis_v1_label_map.json',
    # text_cache_file='cache/vocabulary/lvis_vocabulary.pkl',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[lvis_od_dataset]))

support_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='cache/support/lvis_sup.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=support_pipeline,
        backend_args=backend_args,
        return_classes=True))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    support_dataloader=support_dataloader,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/lvis_v1_minival.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline,
        backend_args=backend_args,
        return_classes=True))

test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root +
    'annotations/lvis_v1_minival.json')
test_evaluator = val_evaluator

val_cfg = dict(type='VisualGenericValLoop', val_mode='text')
# val_cfg = dict(type='VisualGenericValLoop', val_mode='visual-generic')

# test_cfg = dict(type='VisualGenericTestLoop', test_mode='text')
test_cfg = dict(type='VisualGenericTestLoop', test_mode='visual-generic')
# test_cfg = dict(type='VisualGenericTestLoop', test_mode='visual-interactive')

# load_from = "work_dirs/detr-ViP_swin-l_pretrain_obj365-goldg_fre-neg-list_encoder-sup-fuse/epoch_12.pth"