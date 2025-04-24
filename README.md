# PPG Signal Analysis
A Python-based pipeline for processing photoplethysmography (PPG) signals from RGB channels to extract cardiac metrics and classify cardiac rhythms, such as sinus rhythm and atrial fibrillation (AFib).

## Overview
This project processes RGB PPG signals captured from mobile optical sensors. It includes:​
- Signal pre-processing (interpolation, filtering)
- Peak detection
- Cardiac metric computation (HR, rMSSD, SDNN, pNN50)
- Rule-based Rhythm classification (Regular vs. Irregular)​

## Files

``` bash
.
├── data                # data directory, put data here.
├── libs                # core modules
│   ├── data_processor.py           # data preprocess and feature extraction
│   └── rhythm_classifier.py        # rhythm classiier class module
├── scripts             # main scripts
│   ├── main_compare_metrics.py     # main script to compare metrics
│   ├── main_plot_preprocess.py     # main script to plot signals
│   └── main_rhythm_classifier.py   # main script to run rhythm classifier
├── tests               # unit test scripts
│   ├── data                        # data directory for unit test
│   ├── test_all.bash               # bash script to run all unit tests
│   ├── test_data_loader.py         # test for data loading functions
│   ├── test_data_processor.py      # test for data processor
│   └── test_rhythm_classifier.py   # test for rhythm classifier
├── utils               # utility modules
│   └── data_loader.py            # Data loading functions
├── README.md           # This file
└── requirements.txt    # python required libraries
```

## Setup environment

``` shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Unit test

``` shell
bash tests/test_all.bash
```
or run individually
```
python tests/test_data_loader.py
python tests/test_data_processor.py
python tests/test_rhythm_classifier.py
```

## Signal visualisation (before and after preprocessing)
This script is to process the preprocessing for signals and create time series chart to show before and after preprocessing signals. The images will be stored in `res/raw_vs_filtered/` directory.
``` shell
python scripts/main_plot_preprocess.py
```

## Computed metrics comarison with expected result
This script is to compute cardiac metrics and compare with the expected result. Images will be stored in `res/metrics/` directory.
``` shell
python scripts/main_compare_metrics.py
```

## Rhythm classifier
This script is to predict Sinus rhythm or atrial fibrillation (AFib) from signals. The result will be stored in `res/classification/` directory.
``` shell
python scripts/main_rhythm_classifier.py
```
