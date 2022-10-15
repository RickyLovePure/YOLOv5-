## 1. 安装与导入依赖

安装PyTorch
https://pytorch.org/get-started/locally/


```python
!pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102
```

克隆YOLOv5仓库


```python
!git clone https://github.com/ultralytics/yolov5
```

安装YOLOv5仓库


```python
!cd yolov5 & pip install -r requirements.txt
```

导包


```python
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
```

## 2. 加载YOLOv5模型
借助PyTorch https://pytorch.org/hub/ultralytics_yolov5/


```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
```

## 3. 使用YOLOv5进行图像物体识别
配置图像地址


```python
imgs = ['https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg']
```

识别并输出结果


```python
results = model(imgs)
print(results)
```

展示识别图像


```python
%matplotlib inline
plt.imshow(results.render()[0])
```

常用属性


```python
results.pandas().xyxy[0]
```

## 4. 利用OpenCV实现实时物体识别


```python
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## 5. 训练定制模型
导包


```python
import uuid
import os
import time
```

参数配置


```python
IMAGE_PAHT = os.path.join('data', 'images')
labels = ['vup', 'vup-glasses-on']
num_imgs = 20
```

采集图像数据集


```python
cap = cv2.VideoCapture(2)

for label in labels:
    print(f'---------\ncollecting image for {label}\n----------')
    time.sleep(5)
    
    for i in range(num_imgs):
        print(f'collecting {label}.{i + 1}')
        
        ret, frame = cap.read()
        
        image_name = os.path.join(IMAGE_PAHT, label + '.' + str(uuid.uuid1()) + '.jpg')
        cv2.imwrite(image_name, frame)
        cv2.imshow('Data Collecting...', frame)
        time.sleep(2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

克隆labelImg


```python
!git clone https://github.com/heartexlabs/labelImg
```

安装labelImg依赖并进行配置


```python
!pip install pyqt5 lxml --upgrade
```


```python
!cd labelImg && pyrcc5 -o libs/resources.py resources.qrc
```

使用labelImg进行数据标注
```python
python3 labelImg.py
```

配置`dataset.yml`并训练模型


```python
!cd yolov5 && python train.py --img 640 --batch 10 --epochs 500 --data dataset.yml --weights yolov5s.pt
```

## 6. 使用定制模型


```python
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp3/weights/best.pt', force_reload=True)
```


```python
cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
