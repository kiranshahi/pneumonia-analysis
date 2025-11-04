
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

[^1]: https://www.sciencedirect.com/science/article/pii/S0092867418301545