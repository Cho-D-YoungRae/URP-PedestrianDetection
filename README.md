# URP_PD
2021 summer URP. Pedestrian Detection

## Todo
- [x] optimizer: SGD -> Adam
- [ ] lwir + visible_L, lwir + visible_RGB
- [ ] lwir, visible 데이터 확인.
  - lwir 만 사용할 것이 아니라 모두 사용할 이유를 찾자
- [ ] 예측한 데이터 확인
  - 어떤 데이터를 제대로 예측하지 못 했는지 확인하자
- [ ] FocalLoss
- [ ] ground truth 의 scale 과 aspect ratio 분석 후 default box 다시 세팅해보기
- [ ] vgg16_bn 을 통해 VGGbnBase 만들어보기
- [ ] ResNetBase 만들어 보기
- [ ] DataAugmentation 적용

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
- BatchNorm 적용
> 

### 13 (12)
- visible image 사용

### 14 (12)
- visible + lwir image 사용

### 15 (7)
- DataAugmentation 적용 X
- train-val set
- VGG16bn