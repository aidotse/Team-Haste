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
## Docker image
On AZ server:

### 1. Build the docker image if it doesn't exist
`docker build -t 3h4m/haste .`

### 2. Run training with docker
`./run_with_docker.sh -c config.json`

### 3. Run the shell with docker
`docker run --shm-size=24gb -it --rm -v /home/group5/:/workspace 3h4m/haste bash`

---

### Install docker
`curl -sSL https://get.docker.com/ | sh`

### Pull the image 
`sudo docker pull 3h4m/haste`

### Connect to the kernel
`sudo docker run -it 3h4m/haste bash`

---
``
## Running inference

To run the inference you run ai_haste as a module while specifying which config file to use. All configs for inference are located in the "config/test" folder for the different magnifications. The module will run inference on all models (1 model per channel) at the same time and write the result in a pre-specified output folder.

To begin, download the models and put them in the "models" folder. Then edit each config file in $config/test$ and specify the data path and the output path.

config
│── "test"             
│   ├── "output_folder":"output" <--- Path to output folder goes here
│   │
│── "data"             
│   ├── "folder":"" <--- Path to data folder goes here

Inference is then executed by:
`python3 -m ai_haste -c config_test_20x.json` <- for 20x
`python3 -m ai_haste -c config_test_40x.json` <- for 40x
`python3 -m ai_haste -c config_test_60x.json` <- for 60x
