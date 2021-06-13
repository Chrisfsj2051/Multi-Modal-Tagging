_base_ = 'id11.py'

model = dict(modal_used=['image'],
             pretrained=dict(image='torchvision://resnet50'))
