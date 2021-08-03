# URP_PD
2021 summer URP. Pedestrian Detection

SSD에 Segmentation Loss 추가하여 Multi Task Learning (SDS: Simultaneous Detection & Segmentation)

## Todo
- [x] Conv feature 별 Segmentation Loss 비율 확인해보고 weight 설정 고민해보기
  - 학습은 제대로 되고있었는듯... segmentation mask 만들 때 SSD 기본으로 적용되어있던 data augmentation 고려하지 못함
- [ ] faster-rcnn 에서는 모든 크기의 box를 한 feature map에서 검출하지만 ssd는 각 feature map 별로 검출하는 box 크기가 다르다. 그렇기 때문에 segmentation mask 를 만들 때 해당 feature map에서 검출되지 않을 것은 뺴야될 것 같다. IOU 생각해서 합리적으로 계산 해보자. 범위 안에 들지 않는 object 는 0(배경) 으로 할지 -1(무시: CrossEntropy 계산에서 제외 가능한 듯 ignore_index) 로 할지
  - 큰 개선이 필요하기 때문에 별도의 파일로 다시 만드는게 좋을 듯
- [x] 애초에 segmentation 자체가 제대로 이루어지지 않는듯... 3*3 conv 를 segmentation layers 에 추가해야하나...? segmentation loss 를 출력해보며 제대로 학습이 되고 있는 것인지 확인
  - 논문에 segmentation layer 를 깊게 쌓지 않는 이유에 대해서 자세히 나온다

## Reference
- [Code Base](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)
### Paper
- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
- [Illuminating Pedestrians via Simultaneous Detection & Segmentation](https://arxiv.org/abs/1706.08564)

## Question
- [x] convolution, fully-connected layer 에서 bias 를 사용할 때와 사용하지 않을 때의 차이
  - https://excelsior-cjh.tistory.com/180
  - https://stackoverflow.com/questions/45134831/is-bias-necessarily-need-at-colvolution-layer
  - https://github.com/KaimingHe/deep-residual-networks/issues/10
  - https://stackoverflow.com/questions/51959507/does-bias-in-the-convolutional-layer-really-make-a-difference-to-the-test-accura
- [x] missrate, false positive per image
  - https://stackoverflow.com/questions/57511274/understanding-miss-rate-vs-fppi-metric-for-object-detection

## Problem
1. loss 가 nan이 되는 현상
- https://velog.io/@0hye/PyTorch-Nan-Loss-%EA%B2%80%EC%B6%9C-%EB%B0%A9%EB%B2%95
- https://powerofsummary.tistory.com/165


## Pseudo GT
> Segmentation Loss 적용을 위해 필요...

### 1
- Box 모양 그대로 Segmentation GT 생성

## Baseline

### 1
- 코딩 실수로 decay_lr 이 적용되어야 하는 시점에서 오류가 났음
- transform: resize + to_tensor + normalize
- epoch 80 정도에서 train_loss = 0.2 정도까지 떨어졌으나 val loss = 2 이상으로 높았다. 이후 val loss 가 높아질 조짐이 보여서 overfitting 이 의심된다.
- SGD

### 2 (1)
- 코딩 실수로 decay_lr 이 적용되어야 하는 시점에서 오류가 났음
- transform: resize + to_tensor + normalize
- 30 epoch 정도 이후 loss 가 nan 이 되는 현상 발생
- loss 가 완만하게 떨어지고 있었으므로 발산 등은 아닌 것 같다.
- Adam
- Miss Rate : 43.63% (epoch 25)

### 3 (2)
- baseline2 에서 lr = 5e-4 로 줄이고 다시 해봄
- transform: resize + to_tensor + normalize

### 4 - Miss Rate: 29.03%, Recall: 0.8329809725158562 (epoch 130)
- 리더보드 baseline 성능 원복을 위해 SSD tutorial 의 data augmentation 다시 적용 및 validation 으로 나누지 않고 진행
- lr 는 5e-4 로 줄어든 상태로 진행


### 5 (4)
- 4와 같은 세팅에서 train - validation set 으로 구분해주었다.
- 이를 바탕으로 추후 성능 비교해보자

### 6 (5)
- SGD -> Adam
- 나머지는 5번과 같은 세팅

### 7 (6)
- L1Loss -> SmoothL1Loss
- 나머지는 6번과 같은 세팅

### 8
- epochs 300
- 나머지는 7과 같은 세팅

### 9 (7) - Miss Rate: 29.18%, Recall: 0.8028571428571428
- validation으로 나누지 않고 전체 train set 으로 학습
- 나머지 7과 동일

### 10 (9) - Miss Rate: 27.72%, Recall: 0.8092198581560284
- epoch 120

### 11 (9) - Miss Rate: 28.55%, Recall: 0.8131241084165478
- epoch 100
> 8~11 의 그래프를 확인했을 때 epoch가 증가한다고 loss 가 크게 떨어지지 않고 decay_lr 이 적용되는 첫 번째 지점에서만 어느 정도 폭으로 한번 감소 후 더 이상 감소하지 않는다. 9~11 점수를 봐도 크게 차이가 없는 것을 알 수 있다(이정도 차이는 발생할 수 있다). 그렇기 때문에 빠른 실험을 위해 **epoch 100을 사용하자**

### 12 (11) - Miss Rate: 28.25%%, Recall: 0.8172804532577904
> MR(all): 27.93, MR(day): 33.82, MR(night): 16.06
- BatchNorm 적용

### 13 (12) - Miss Rate: 35.52%, Recall: 0.7742175856929955
> MR(all): 36.34 MR(day): 33.86 MR(night): 42.52
- visible image 사용

### 14 (12)
> MR(all): 34.75 MR(day): 35.27 MR(night): 33.55
- visible + lwir image 사용

### 15 (13)
- epoch 1e-3

### 16 (12)
- evaluate every 10 epochs

### 17 (13)
- evaluate every 10 epochs

### 18 - Miss Rate: 29.03%%, Recall: 0.8101983002832861
> MR(all): 29.02 MR(day): 33.77 MR(night): 19.17
- segmentation infusion layers 적용
- usages_seg_feats = [True, True, False, False, False, False]
- total_seg_loss 의 평균 사용
- 추가적인 적용 X

### 19 (18) - Miss Rate: 26.09%, Recall: 0.8295615275813296
> MR(all): 26.33 MR(day): 31.38 MR(night): 15.89
- total_seg_loss 의 합 사용 (평균 X)

### 20 (19)
> MR(all): 27.72 MR(day): 32.78 MR(night): 16.59
- usages_seg_feats = [True, True, True, False, False, False]

### 21 (20)
> MR(all): 27.12 MR(day): 33.33 MR(night): 14.15
- segmentation 에 3*3 conv 추가
```python
# seg_infusion_layer = nn.Conv2d(in_channels, self.n_classes, kernel_size=1)
seg_infusion_layer = nn.Sequential(
    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
    nn.Conv2d(in_channels, self.n_classes, kernel_size=1)
)
```

### 22 (19)
- usages_seg_feats = [True, True, True, True, True, True]

### 23 (19)
- usages_seg_feats = [True, False, False, False, False, False]

### 24 (23)
> conv_feats 을 통해 얻은 torch.mean(seg_loss)을 이용하고 있었다 -> 이로 인해 loss가 너무 작아 학습이 제대로 이루어지지 않은 듯
- segmentation loss 가 잘 떨어지고 있는지 확인

### 25 (24)
- torch.sum(seg_loss) 으로 변경

### 26
- 앞에서는 segmentation gt 를 생성할 때 image 들의 transform 을 고려하지 못 했다. 이를 해결함.