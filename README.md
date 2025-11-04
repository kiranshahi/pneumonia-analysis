
### Weakly Supervised Pneumonia Localization from Chest X-Rays Using Deep Neural Network and Grad-CAM Explanations

Train, evaluate, and visualize Grad-CAM for binary chest X-ray classification (Normal vs Pneumonia) with **consistent preprocessing**, **patient-wise split**, and **multi-model ensembling**.

## Project layout
```
pneumonia-analysis/
  src/
    __init__.py
    data.py
    model.py
    train.py
    eval.py
    gradcam.py
    train_many.py
    ensemble_eval.py
    utils.py
  requirements.txt
```
#### Localized result using Grad-CAM 

![Result on Chest X-ray dataset.](docs/img/Grad-CAM.png)

*Results on the Chest X-ray dataset[^1].*

## 💾 Pretrained Weights

You can download pretrained model checkpoints from the following links:

| Model | Parameters (M) | Download |
|:--|:--:|:--:|
| ResNet-18 | 11.5 | [⬇️ Download](https://drive.google.com/file/d/1is3ZML1WfCcVYwr-gVXH2K7UH4rNkK99/view?usp=sharing) |
| ResNet-50 | 24.6 | [⬇️ Download](https://drive.google.com/file/d/18rlu9IR4z9gE-Joj21l1kdvE1YEVFqWn/view?usp=sharing) |
| DenseNet-121 | 7.5 | [⬇️ Download](https://drive.google.com/file/d/1CQRg1RAKDWV8u-DmJyWz5-ye7IY5RvhU/view?usp=sharing) |
| EfficientNet-B0 | 4.7 | [⬇️ Download](https://drive.google.com/file/d/1mZuTbLOkByszvrFXvq4vt5JKLZUXH2kL/view?usp=sharing) |
| MobileNet-V2 | 2.9 | [⬇️ Download](https://drive.google.com/file/d/1boPL1gcIKd82yFBYVqz44TKFlNxhfIqf/view?usp=sharing) |
| MobileNet-V3 | 4.9 | [⬇️ Download](https://drive.google.com/file/d/1Ooud467Gr9EzTAyscsg6QjsPPv2QB-70/view?usp=sharing) |
| Vision Transformer (ViT-B/16) | 86.2 | [⬇️ Download](https://drive.google.com/file/d/1sqWzbNPRKyNySK9ftWCQNItuYmY6VO0y/view?usp=sharing) |

Each `.pth` file contains the best model checkpoint (based on validation ROC-AUC) trained on the Chest X-Ray dataset with consistent preprocessing and patient-wise data split.


#### Citation
```
@misc{shahi2025weaklysupervisedpneumonialocalization,
      title={Weakly Supervised Pneumonia Localization from Chest X-Rays Using Deep Neural Network and Grad-CAM Explanations}, 
      author={Kiran Shahi and Anup Bagale},
      year={2025},
      eprint={2511.00456},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.00456}, 
}
```

[^1]: https://www.sciencedirect.com/science/article/pii/S0092867418301545