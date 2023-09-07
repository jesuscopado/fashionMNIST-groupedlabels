# Fashion MNIST - Grouped labels
This project aims to build a multi-class fashion classifier using the FashionMNIST dataset. The classifier categorizes items into five different classes: Upper part, Bottom part, One piece, Footwear, and Bags.

## Features
- Multi-class classification using MobileNetV3 small architecture.
- Data preprocessing and exploration.
- Metrics evaluation: Accuracy, Precision, Recall, and F1 Score.
- CI/CD Pipeline Plan and SQL Query in separate documentation files (under the dir `docs/`).

NOTE: While the MobileNetV3 small architecture is effective for this task, it may be considered overkill given the simplicity of the problem at hand. The model quickly converges to optimal results, often within just one epoch, suggesting that a simpler architecture might be sufficient.

## Installation
To clone this repository, simply run:

```bash
git clone https://github.com/jesuscopado/FashionMNIST-GroupedLabels.git
```

## Usage

### Jupyter Notebook
To view and execute the notebook (`notebooks/FashionMNIST_Grouped_labels.ipynb`) locally:

1. Install Jupyter Notebook if not already installed: `pip install notebook`.
2. Navigate to the repository directory and run: `jupyter notebook`.
3. Open the notebook containing exploratory data analysis and model training.

Alternatively, you can upload the Jupyter notebook to Google Colab for execution.

### Training and Evaluation Scripts
In addition to the notebook, you can also run the model training and evaluation using Python scripts.

- For training, navigate to the `src/` folder and run:
```bash
python train.py [arguments]
```

- For evaluation, navigate to the `src/` folder and run:
```bash
python eval.py [arguments]
```

NOTE: Evaluating on CPU a model trained on GPU could cause issues at this state.

## Testing
Unit tests can be found under the `test/` directory. To run the tests:

```bash
python -m unittest
```

## Documentation
- For complete details on CI/CD Plan and SQL Query, please refer to the `docs/` folder.
- Notebook contains the data exploration and model building process.
