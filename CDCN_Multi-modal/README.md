# [CDCN_Multi-modal](https://docs.google.com/presentation/d/1FkZ9dRnFEa5QCz-b2-X05wJCHu0KLcUD4RgGdar81_0/edit?usp=sharing)
In these repository invloves 3 experminent.
1. Multi-Modal CDCNs(rgb/ir/depth)
2. Multi-Modal CDCNs(rgb/depth)
3. Multi-Modal CDCNs(rgb/depth) + remove mid-block in modal backbone

# Dataset
* Using CASIA-surf cefa protocol dataset 4@1, 4@2, 4@3 from Multi-modal Challenge@CVPR2020
* And I also using some script in tools/fordataset folder, to merge Cross-ethnicity & PAI attack data into `b_train.txt` and `b_val.txt`
* Test data using surfing tech provide 3-modal test data
    - **[2020 Anti Spoofing Data——Paper Attack](http://www.surfing.ai/Datasets/226.html)**
    - **[2020 Anti Spoofing Data——3D Mask Attack](http://www.surfing.ai/Datasets/224.html)**
    - **[2020 Anti Spoofing Data——Collecting Real Person](http://www.surfing.ai/Datasets/223.html)**
* ![image](https://user-images.githubusercontent.com/24097516/228152092-4bebe5ac-7238-4b0d-b138-b5dfd9a7dab9.png)

# Demo
* Simple demo using MCDCN test in surfing tech data
![image](https://user-images.githubusercontent.com/24097516/228152527-aead04e0-43d6-4a36-9abd-c30ef5dd4b7f.png)

# Result
| Format | GPU / CPU | Inference Time(s) | Image size |
| --- | --- | --- | --- |
| Pytorch (pkl) | NVIDIA GeForce RTX 2080 SUPER | 0.83~0.87 | 256x256 |
| Pytorch (pkl) | Intel(R) Core(TM) i5-9400F CPU @ 2.90GHz | 379s | 256x256 |
| Python (onnx) | Intel(R) Core(TM) i5-9400F CPU @ 2.90GHz | 47~49 | 256x256 |

| Metric | DataType | APCER(%) | BPCER(%) | ACER(%) | Accuracy
(%) | Thres |
| --- | --- | --- | --- | --- | --- | --- |
| Multi-Modal CDCN | Merge Multi-ethnicity&PAI | 0 | 100 | 50 | 85.7 | 0.9 |
| Multi-Modal CDCN | Merge Multi-ethnicity&PAI | 14.81 | 77.77 | 53.7 | 76.19 | 0.7 |
| Multi-Modal CDCN | Merge Multi-ethnicity&PAI | 31.48 | 44.44 | 53.7 | 66.66 | 0.5 |

* Due to the difference in depth attributes between the surfing tech dataset and the casia-cefa dataset, there are many false detections relying on the module of the depth pattern.
