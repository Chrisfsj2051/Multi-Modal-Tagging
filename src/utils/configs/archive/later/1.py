2021-07-05 10:53:10,737 - mmdet - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.7.10 | packaged by conda-forge | (default, Feb 19 2021, 16:07:37) [GCC 9.3.0]
CUDA available: True
GPU 0,1: Tesla P40
CUDA_HOME: None
GCC: gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
PyTorch: 1.6.0
PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) oneAPI Math Kernel Library Version 2021.2-Product Build 20210312 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v1.5.0 (Git Hash e2ac1fac44c5078ca927cb9b90e1b3066a0b2ed0)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 10.1
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_37,code=compute_37
  - CuDNN 7.6.3
  - Magma 2.5.2
  - Build settings: BLAS=MKL, BUILD_TYPE=Release, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_VULKAN_WRAPPER -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, USE_CUDA=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_STATIC_DISPATCH=OFF,

TorchVision: 0.7.0
OpenCV: 4.5.2
MMCV: 1.2.4
MMCV Compiler: GCC 7.3
MMCV CUDA Compiler: 10.1
RepoVersion: f5aaf46
------------------------------------------------------------

2021-07-05 10:53:13,514 - mmdet - INFO - Distributed training: True
2021-07-05 10:53:16,381 - mmdet - INFO - Config:
checkpoint_config = dict(interval=1000)
evaluation = dict(interval=1000)
seed = 1
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = []
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.01, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8000, 9000])
runner = dict(type='IterBasedRunner', max_iters=10000)
modal_used = ['image', 'video', 'text', 'audio']
model = dict(
    type='SingleBranchModel',
    pretrained='torchvision://resnet50',
    key='image',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    head=dict(
        type='SingleSEHead',
        in_dim=2048,
        gating_reduction=8,
        out_dim=1024,
        dropout_p=0.8,
        cls_head_config=dict(
            type='ClsHead',
            in_dim=1024,
            out_dim=82,
            loss=dict(type='MultiLabelBCEWithLogitsLoss', loss_weight=8)),
        norm_cfg=dict(type='SyncBN')))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='CutOut',
        n_holes=3,
        cutout_ratio=[(0.05, 0.05), (0.1, 0.1), (0.07, 0.07)]),
    dict(
        type='CutOut',
        n_holes=1,
        cutout_ratio=[(0.2, 0.2), (0.15, 0.15), (0.13, 0.13)]),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'Shear',
            'prob': 0.5,
            'level': 1
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 2
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 3
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 4
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 5
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 6
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 7
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 8
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 9
        }], [{
            'type': 'Shear',
            'prob': 0.5,
            'level': 10
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 1
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 2
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 3
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 4
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 5
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 6
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 7
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 8
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 9
        }], [{
            'type': 'Rotate',
            'prob': 0.5,
            'level': 10
        }]]),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['image', 'meta_info', 'gt_labels'])
]
val_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='BertTokenize', bert_path='pretrained/bert', max_length=256),
    dict(type='Pad', video_pad_size=(300, 1024), audio_pad_size=(300, 128)),
    dict(type='Resize', size=(224, 224)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['image', 'meta_info'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='TaggingDataset',
        ann_file='dataset/tagging/GroundTruth/datafile/train.txt',
        label_id_file='dataset/tagging/label_super_id.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='CutOut',
                n_holes=3,
                cutout_ratio=[(0.05, 0.05), (0.1, 0.1), (0.07, 0.07)]),
            dict(
                type='CutOut',
                n_holes=1,
                cutout_ratio=[(0.2, 0.2), (0.15, 0.15), (0.13, 0.13)]),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 1
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 2
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 3
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 4
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 5
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 6
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 7
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 8
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 9
                }], [{
                    'type': 'Shear',
                    'prob': 0.5,
                    'level': 10
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 1
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 2
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 3
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 4
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 5
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 6
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 7
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 8
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 9
                }], [{
                    'type': 'Rotate',
                    'prob': 0.5,
                    'level': 10
                }]]),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='BertTokenize',
                bert_path='pretrained/bert',
                max_length=256),
            dict(
                type='Pad',
                video_pad_size=(300, 1024),
                audio_pad_size=(300, 128)),
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375]),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['image', 'meta_info', 'gt_labels'])
        ]),
    val=dict(
        type='TaggingDataset',
        ann_file='dataset/tagging/GroundTruth/datafile/val.txt',
        label_id_file='dataset/tagging/label_super_id.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(
                type='BertTokenize',
                bert_path='pretrained/bert',
                max_length=256),
            dict(
                type='Pad',
                video_pad_size=(300, 1024),
                audio_pad_size=(300, 128)),
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375]),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['image', 'meta_info'])
        ]),
    test=dict(
        type='TaggingDataset',
        ann_file='dataset/tagging/GroundTruth/datafile/test_2nd.txt',
        label_id_file='dataset/tagging/label_super_id.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(
                type='BertTokenize',
                bert_path='pretrained/bert',
                max_length=256),
            dict(
                type='Pad',
                video_pad_size=(300, 1024),
                audio_pad_size=(300, 128)),
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375]),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['image', 'meta_info'])
        ]))
norm_cfg = dict(type='SyncBN')
work_dir = './work_dirs/_image_id38_2gpu'
gpu_ids = range(0, 2)