# Augmentation in Semantic Segmentation of SAR Images

SpaceNet6 데이터셋을 활용하여 SAR 영상의 건물 Segmentation 할 때의 Data Augmentation 성능 변화 실험

### 참고 논문:
Wangiyana, S.; Samczyński, P.; Gromek, A. Data Augmentation for Building Footprint Segmentation in SAR Images: An Empirical Study. Remote Sens. 2022, 14, 2012. [https://doi.org/10.3390/rs14092012](https://doi.org/10.3390/rs14092012)

### Backbone
- FPN

### Encoder
- EfficientNet B4

### Training Parameters
- Batch size: 32
- Epochs: 60
- Learning rate: 32e-4

### Scheduler
- Cosine Anealing Scheduler

### Geometric Augmentation 성능 변화

| Augmentation        | Train IoU | Valid IoU |
| ------------------- | --------- | --------- |
| Baseline            | 0.7900    | 0.3330    |
| Horizontal Flip     | 0.6662    | 0.3783    |
| Vertical Flip       | 0.6834    | 0.3629    |
| Rotation 90         | 0.5523    | 0.3620    |
| Fine Rotation [-10, 10] | 0.7101    | 0.3631    |
| ShearX [-10, 10]    | 0.7358    | 0.3535    |
| ShearY [-10, 10]    | 0.7434    | 0.3532    |
| Random Erasing      | 0.7292    | 0.3468    |

### Pixel Augmentation 성능 변화

| Augmentation   | Train IoU | Valid IoU |
| -------------- | --------- | --------- |
| Baseline       | 0.7900    | 0.3330    |
| Sharpening     | 0.7789    | 0.3213    |
| CLAHE          | 0.7565    | 0.0010    |
| Gaussian Noise | 0.7696    | 0.3315    |
| Speckle Noise  | 0.7194    | 0.3422    |

### 종합 Augmentation 성능 변화

| Augmentation   | Train IoU | Valid IoU |
| -------------- | --------- | --------- |
| Baseline       | 0.7900    | 0.3330    |
| Light Pixel    | 0.7192    | 0.3341    |
| Light Geometry | 0.6264    | 0.3914    |
| Heavy Geometry | 0.5969    | 0.4036    |
| Combination    | 0.6017    | 0.3734    |

## Using

### 데이터셋 경로 설정
x_train_dir = r'dataset/x_train'
y_train_dir = r'dataset/y_train'
x_valid_dir = r'dataset/x_valid'
y_valid_dir = r'dataset/y_valid'

### Training 명령어
python seg_aug_train.py --dataset [Augmentation]

### Augmentation List
- "baseline"
- "Horizontal Flip"
- "Vertical Flip"
- "Rotation90"
- "Fine Rotation [-10, 10]"
- "ShearX [-10, 10]"
- "ShearY [-10, 10]"
- "Random Erasing"
- "Motion Blur"
- "Sharpening"
- "CLAHE"
- "Gaussian Noise"
- "Speckle Noise"
- "Light Pixel"
- "Light Geometry"
- "Heavy Geometry"
- "Combination"
