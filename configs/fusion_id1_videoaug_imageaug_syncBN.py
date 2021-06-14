_base_ = 'fusion_id1_videoaug_imageaug.py'

load_from = 'pretrained/text3_audio1_video1_image34.pth'

norm_cfg = dict(type='SyncBN')

model = dict(branch_config=dict(
    video=dict(norm_cfg=norm_cfg),
    audio=dict(norm_cfg=norm_cfg),
),
             head_config=dict(image=dict(norm_cfg=norm_cfg),
                              video=dict(norm_cfg=norm_cfg),
                              text=dict(norm_cfg=norm_cfg),
                              audio=dict(norm_cfg=norm_cfg),
                              fusion=dict(norm_cfg=norm_cfg)))
