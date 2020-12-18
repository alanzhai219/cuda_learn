import torch
import torchvision
from torchvision.ops.boxes import nms

ld = torch.load('nms_db')

boxes = ld['boxes']
score = ld['score']
thres = 0.7

keep = nms(boxes, score, thres)

print('FINISH')
