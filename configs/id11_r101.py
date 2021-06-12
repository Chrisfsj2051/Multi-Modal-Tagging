_base_ = 'id11.py'

model = dict(pretrained=dict(image='torchvision://resnet101'),
             branch_config=dict(image=dict(depth=101)))
