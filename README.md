# URP_PD
2021 summer URP. Pedestrian Detection

## Todo
- [x] optimizer: SGD -> Adam
- [ ] FocalLoss
- [ ] ground truth 의 scale 과 aspect ratio 분석 후 default box 다시 세팅해보기
- [ ] vgg16_bn 을 통해 VGGbnBase 만들어보기
- [ ] ResNetBase 만들어 보기
- [ ] DataAugmentation 적용

## Question
- [ ] convolution, fully-connected layer 에서 bias 를 사용할 때와 사용하지 않을 때의 차이
## Problem
1. loss 가 nan이 되는 현상
- https://velog.io/@0hye/PyTorch-Nan-Loss-%EA%B2%80%EC%B6%9C-%EB%B0%A9%EB%B2%95
- https://powerofsummary.tistory.com/165

## Baseline
### 1
- 코딩 실수로 decay_lr 이 적용되어야 하는 시점에서 오류가 났음
- transform: resize + to_tensor + normalize
- epoch 80 정도에서 train_loss = 0.2 정도까지 떨어졌으나 val loss = 2 이상으로 높았다. 이후 val loss 가 높아질 조짐이 보여서 overfitting 이 의심된다.

### 2
- 코딩 실수로 decay_lr 이 적용되어야 하는 시점에서 오류가 났음
- transform: resize + to_tensor + normalize
- 30 epoch 정도 이후 loss 가 nan 이 되는 현상 발생
- loss 가 완만하게 떨어지고 있었으므로 발산 등은 아닌 것 같다.
- Miss Rate : 43.63% (epoch 25)

### 3
- baseline2 에서 lr = 5e-4 로 줄이고 다시 해봄
- transform: resize + to_tensor + normalize

### 4
- 리더보드 baseline 성능 원복을 위해 SSD tutorial 의 data augmentation 다시 적용 및 validation 으로 나누지 않고 진행
- lr 는 5e-4 로 줄어든 상태로 진행
- Miss Rate: 29.03% (epoch 130)

### 5
- 4와 같은 세팅에서 train - validation set 으로 구분해주었다.
- 이를 바탕으로 추후 성능 비교해보자

### 6
- SGD -> Adam
- 나머지는 5번과 같은 세팅

### 7
- L1Loss -> SmoothL1Loss
- 나머지는 6번과 같은 세팅
