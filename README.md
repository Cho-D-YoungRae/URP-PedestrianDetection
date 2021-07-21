# URP_PD
2021 summer URP. Pedestrian Detection

## Todo
- [ ] optimizer: SGD -> Adam
- [ ] ground truth 의 scale 과 aspect ratio 분석 후 default box 다시 세팅해보기
- [ ] vgg16_bn 을 통해 VGGbnBase 만들어보기
- [ ] ResNetBase 만들어 보기
- [ ] DataAugmentation 적용
## Question
- [ ] convolution, fully-connected layer 에서 bias 를 사용할 때와 사용하지 않을 때의 차이

## Baseline
### 1
- 코딩 실수로 decay_lr 이 적용되어야 하는 시점에서 오류가 났음
- epoch 80 정도에서 train_loss = 0.2 정도까지 떨어졌으나 val loss = 2 이상으로 높았다. 이후 val loss 가 높아질 조짐이 보여서 overfitting 이 의심된다.

### 2
- 코딩 실수로 decay_lr 이 적용되어야 하는 시점에서 오류가 났음
- 30 epoch 정도 이후 loss 가 nan 이 되는 현상 발생
- loss 가 완만하게 떨어지고 있었으므로 발산 등은 아닌 것 같다.

## Problem
1. loss 가 nan이 되는 현상
- https://velog.io/@0hye/PyTorch-Nan-Loss-%EA%B2%80%EC%B6%9C-%EB%B0%A9%EB%B2%95
- https://powerofsummary.tistory.com/165
   
