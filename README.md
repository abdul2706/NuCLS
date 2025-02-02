# NuCLS: A scalable crowdsourcing, deep learning approach and dataset for nucleus classification, localization and segmentation

High-resolution mapping of cells and tissue structures provides a foundation for developing interpretable machine-learning models for computational pathology. Deep learning algorithms can provide accurate mappings given large numbers of labeled instances for training and validation. Generating adequate volume of quality labels has emerged as a critical barrier in computational pathology given the time and effort required from pathologists. In this paper we describe an approach for engaging crowds of medical students and pathologists that was used to produce a dataset of over 220,000 annotations of cell nuclei in breast cancers. We show how suggested annotations generated by a weak algorithm can improve the accuracy of annotations generated by non-experts and can yield useful data for training segmentation algorithms without laborious manual tracing. We systematically examine interrater agreement and describe modifications to the MaskRCNN model to improve cell mapping. We also describe a technique we call Decision Tree Approximation of Learned Embeddings (DTALE) that leverages nucleus segmentations and morphologic features to improve the transparency of nucleus classification models. The annotation data produced in this study are freely available for algorithm development and benchmarking at: [https://sites.google.com/view/nucls](https://sites.google.com/view/nucls).

## Setup

1. Create python environment and cd to it's directory then activate it.

    ```cmd
    >> python -m venv NuCLS-env
    >> cd NuCLS-env
    >> Scripts\activate
    ```

2. Clone this repo and cd to it's directory.

    ```cmd
    >> git clone https://github.com/abdul2706/NuCLS
    >> cd NuCLS
    ```

3. Create following necessary folders inside NuCLS directory.

    ```cmd
    >> mkdir "data/tcga-nucleus/v4_2020-04-05_FINAL_CORE/CORE_SET/"
    >> mkdir "data/tcga-nucleus/v4_2020-04-05_FINAL_CORE/CORE_SET/QC/"
    >> mkdir "results"
    >> mkdir "results/tcga-nucleus"
    >> mkdir "results/tcga-nucleus/models"
    >> mkdir "results/tcga-nucleus/models/v4_2020-04-05_FINAL_CORE_QC"
    ```

4. Download dataset (QC.zip) from [drive](https://drive.google.com/file/d/1k350VQeegN5hMxRK9Vpc65fdLe3wsqYy/view?usp=sharing) link.

5. Unzip QC.zip inside following path: **data/tcga-nucleus/v4_2020-04-05_FINAL_CORE/CORE_SET/QC/** inside NuCLS directory. After unzip, make sure you have **csv, mask, rgbs, rgbs_colorNormalized and train_test_splits** directories inside QC directory.

6. Install necessary packages.

    ```cmd
    >> pip install -r requirements.txt
    ```

7. Follow [HistomicsTK](https://github.com/CancerDataScience/HistomicsTK) instructions to install it.

8. Run following command to train model.

    ```cmd
    >> python train.py
    ```

## Updates

2021-05-20: fixed path related issues and other errors, formatted the code, added the sqlite files, added training code and it's notebook, updated README.md
