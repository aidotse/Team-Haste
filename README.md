# HASTE Team: Adipocyte Cell Challenge

In this repository we provide the code base of our solution to the challenge hosted by AI Sweden and Astra Zeneca (https://www.ai.se/en/news/challenge-can-machine-learning-replace-potential-cytotoxic-labeling-cell-cultures). The content and structure of the repo is given by the following: 

```sh
.
├── README.md
├── ai_haste
│   ├── data                : python scripts for loading the datasets required
│   ├── loss                : for loss functions not included in PyTorch
│   ├── model               : python scripts for the various neural networks
│   ├── test                : scripts requied when running the models in test mode
│   ├── train               : scripts requied when running the models in train mode
│   └── utils               : utility functions (e.g. for converting the images to numpy arrays for faster data loading)
├── config
│   ├── train               : .json files for running the models to reconstruct the three fluorescence channels
│   │   ├── nuclei          
│   │   ├── lipids
│   │   └──  cytoplasm
│   ├── test                : .json files for running the models in test mode for the three fluorescence channels
│   │   ├── nuclei
│   │   ├── lipids
│   │   └──  cytoplasm
├── docker                  : files for Docker image construction and running
└── exp_stats               : .csv files for image statistics and train/test splits 
    
```
