## FeatherNets for [Face Anti-spoofing Attack Detection Challenge@CVPR2019](https://competitions.codalab.org/competitions/20853#results)[1]

## The detail in orign paper：[FeatherNets: Convolutional Neural Networks as Light as Feather for Face Anti-spoofing](https://arxiv.org/pdf/1904.09290)

# My extend of Multi-Modal Face Spoof Model
| Model | Multi-Modal FeatherNetB | Multi-Modal FeatherNetB | Multi-Modal FeatherNetB |
| --- | --- | --- | --- |
 | Fusion | concat + linear mapping | concat + 3 resnet block + adaptive pooling | cdcn fusion(last 2 layers cdc convolution) |
| input | 224 x 224 | 224 x 224 | 224 x 224 |
| Total params | 1.06MB | 45.28MB | 3.86MB |
| Number of FLOPs | 296.87MFlops | 400.21MFlops | 434.21MFlops |
| Inference Time | 13ms | 16ms | 14ms |

# Results on the validation set (Experiment - Multi-Modal FeatherNetB Fusion)
- Batch: 32
- Train on Nvidia-2080 SUPER 8G
- CPU: Intel(R) Core(TM) i5-9400F CPU @ 2.90GHz
- Train Set: CASIA-SURF CEFA (4@1 + 4@2 + 4@3 Train)
- Test Set: CASIA-SURF CEFA (4@1 + 4@2 + 4@3 Test)

| Fusion Method | Acc(%) | EER(%) | TPR@FPR=10E-2(%) | TPR@FPR=10E-3(%) | APCER(%) | BPCER(%) | ACER(%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Featue-Level | 88.321 | 0.109524 | 0.301429  | 0.075714  | 0.148571 | 0.02142 | 0.085 |
| PipeNet Fusion | 97.321 | 0.007143 | 0.994286 | 0.981429  | 0.034286 | 0.004286 | 0.01928 |
| CDCN Fusion | 98.929 | 0.004286 |  0.997143  | 0.988571 | 0.01381 | 0.001429 | 0.007619 |

# Compare to paper original model
| Metric / Train hours | Acc(%) | EER(%) | TPR@FPR=10E-2(%) | TPR@FPR=10E-3(%) | APCER(%) | BPCER(%) | ACER(%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FeatherNetA | 98.607 | 0.008571 | 0.991429 | 0.978571 | 0.017619 | 0.002857 | 0.0102 |
| FeatherNetB | 99.07 | 0.004 | 0.997 | 0.99 | 0.011 | 0.0028 | 0.0071 |
| PipeNet Fusion | 97.321 | 0.007143 | 0.994286 | 0.981429 | 0.034286 | 0.004286 | 0.01928 |
| CDCN Fusion | 98.929 | 0.004286 |  0.997143  | 0.988571 | 0.01381 | 0.001429 | 0.007619 |

* The performance of Pipenet Fusion is catching up with that of the single-modal FeatherNetB in terms of accuracy.
* The performance of CDCN Fusion is better than that of Pipenet Fusion and even surpasses that of single-modal FeatherNetA.


# Our Pretrained Models(model checkpoints)
Link:https://pan.baidu.com/s/1vlKePiWYFYNxefD9Ld16cQ 
Key:xzv8

decryption key: OTC-MMFD-11846496
[Google Dirve](https://drive.google.com/open?id=1F_du_iarTepKKYgXpk_cJNGRb34rlJ5c)


# Prerequisites

##  install requeirements
```
conda env create -n env_name -f env.yml
```


# Train the model

### Download pretrained models(trained on ImageNet2012)
download [fishnet150](https://pan.baidu.com/s/1uOEFsBHIdqpDLrbfCZJGUg) pretrained model from [FishNet150 repo](https://github.com/kevin-ssy/FishNet)(Model trained without tricks )

download [mobilenetv2](https://drive.google.com/open?id=1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR) pretrained model from [MobileNet V2 repo](https://github.com/tonylins/pytorch-mobilenet-v2),or download from here,link: https://pan.baidu.com/s/11Hz50zlMyp3gtR9Bhws-Dg password: gi46 
**move them to  ./checkpoints/pre-trainedModels/**


### 1.train FishNet150

> nohup python main.py --config="cfgs/fishnet150-32.yaml" --b 32 --lr 0.01 --every-decay 30 --fl-gamma 2 >> fishnet150-train.log &
###  2.train MobileNet V2

> nohup python main.py --config="cfgs/mobilenetv2.yaml" --b 32 --lr 0.01 --every-decay 40 --fl-gamma 2 >> mobilenetv2-bs32-train.log &

Commands to train the model:
####  3Train MobileLiteNet54
```
python main.py --config="cfgs/MobileLiteNet54-32.yaml" --every-decay 60 -b 32 --lr 0.01 --fl-gamma 3 >>FNet54-bs32-train.log
```
####  4Train MobileLiteNet54-SE
```
python main.py --config="cfgs/MobileLiteNet54-se-64.yaml" --b 64 --lr 0.01  --every-decay 60 --fl-gamma 3 >> FNet54-se-bs64-train.log
```
#### 5Train FeatherNetA
```
python main.py --config="cfgs/FeatherNetA-32.yaml" --b 32 --lr 0.01  --every-decay 60 --fl-gamma 3 >> MobileLiteNetA-bs32-train.log
```
#### 6Train FeatherNetB
```
python main.py --config="cfgs/FeatherNetB-32.yaml" --b 32 --lr 0.01  --every-decay 60 --fl-gamma 3 >> MobileLiteNetB-bs32--train.log

```

#### 7Train FeatherNetB (My Custom)
```
python main.py --config="cfgs/FeatherNet-modal-fusion.yaml" --b 32 --lr 0.01  --every-decay 60 --fl-gamma 3 >> FeatherNet-modal-fusion-bs32--train.log

```


## How to create a  submission file
example:
> python main3modal.py --config="cfgs/mobilenetv2.yaml" --resume ./checkpoints/mobilenetv2_bs32/_4_best.pth.tar --val True --val-save True

# Transform to onnx
```
python Feather_pytorch_2_onnx.py
```
```
python Feather_pytorch_3modal_onnx.py
```

# Ensemble 

### for validation
```
run EnsembledCode_val.ipynb
```
### for test
```
run EnsembledCode_test.ipynb
```
**notice**:Choose a few models with large differences in prediction results
