# ttf

tracking traffic light

# Data

https://drive.google.com/file/d/1mVIt3WfnTe1E0H8KZlIDxG_JaBrth5Ej/view?usp=sharing

# Install

pip install -r requirements.txt 

# Run

python3 demo.py 'path to video'

# Train

python3 train.py --data ../data.yaml --epochs 300 --weights yolov5s.pt --batch-size 16 --img 640 --cache ram --hyp ./runs/evolve/exp2/hyp_evolve.yaml

# hup.yaml

lr0: 0.01117
lrf: 0.01062
momentum: 0.95704
weight_decay: 0.00053
warmup_epochs: 3.145
warmup_momentum: 0.84833
warmup_bias_lr: 0.11466
box: 0.06022
cls: 0.51602
cls_pw: 0.92867
obj: 1.0444
obj_pw: 0.92248
iou_t: 0.2
anchor_t: 3.7003
fl_gamma: 0.0
hsv_h: 0.01411
hsv_s: 0.61574
hsv_v: 0.36388
degrees: 0.0
translate: 0.12281
scale: 0.55119
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 0.87277
mixup: 0.0
copy_paste: 0.0
anchors: 2.0


