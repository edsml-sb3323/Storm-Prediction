# The Day After Tomorrow - Ciaran Team 

## Predicting tropical storm behaviour through Deep Learning

### Description 

The GitHub package provides a tool for predicting Storm behaviour. It can generate future storm evolution images over time while also predicting the associated wind speed at the same timestep. This can help provide better emergency services response, Early Warning and Evacuation & Emergency Response Planning​. In addition such a tool helps with overall Risk Mitigation, Infrastructure Protection, Agricultural Planning and Scientific Research.

Note that the model works best on predicting storms evolution in the Atlantic and Pacific Oceans as the models were exclusively trained on data from these oceans. However, It is possible for the user to retrain the models with storms from new oceans in order to create a more robust model (including Indian Ocean for example).

### Installation Instructions for usage on Google Colab:
- Open the repository with your Google Colab Account. No environment.yml file or environment creation is needed when choosing this method.
  
- The files are primarily meant to be ran on Google, however it can also be ran locally by changing path names / directly reading the data from the data folder provided.

### Installation Instructions for local usage
 
- Clone the repository from github using the following command:
 
  ```bash
 
    git clone [https://github.com/ese-msc-2023/ads-deluge-severn](https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-ciaran)
 
- Install the environment.yml file using the following command:
 
  ```bash
 
    conda env create -f environment.yml
 
- It will be named 'storm'. You can activate it using the following command:
 
  ```bash
 
    conda activate storm
 
The environment.yml file contains all the packages required to run the tool.

 ### Software Installation Guide

1. Install Python
   - Ensure Python 3.11 is installed on your system. If not, download and install it from [python download website](https://www.python.org/downloads/).
2. Set Up a Conda Environment
   - Create a new Conda environment using the provided environment.yml file:
   ```bash
   conda env create -f environment.yml
   conda activate storm
   ```
   - This environment includes needed packages such as numpy, scipy, pandas, folium, scikit-learn ,matplotlib and ipywidgets.

3. Verify Installation
   - Ensure that all components are correctly installed and functioning.

### User instructions

#### Image Generation

1. To create your own generated images:
   
- Go to the solutions folder
  
- From there, open the image generation folder

- Follow the instructions of the notebook 

2. To get wind predictions:

- Go to solutions folder

- From there, open the wind prediction folder

- Follow the instructions on the notebook

### Testing

The tool includes several tests, which can be  uses to check its operation on your system. They can be found in the test folder and include the following tests:

test_storm_dataset: test the class call StormDataset

test_model: test the model EncoderDecoderConvLSTM

test_utils: test any functions inside utils.py

you can use the tests in test folder, by going to:

```bash
cd /path/test
```

With [pytest](https://doc.pytest.org/en/latest) installed, these can be run with

```bash
python -m unittest test.test_model
```

Alternatively, you could use Github actions which have been set up.

### Reading list/ Resources

[1]“Papers with Code - ResNet Explained,” paperswithcode.com. https://paperswithcode.com/method/resnet
‌
[2]“What is dtype(‘O’), in pandas?,” Stack Overflow. https://stackoverflow.com/questions/37561991/what-is-dtypeo-in-pandas (accessed Feb. 02, 2024).

[3]“Is it possible to do a CNN for classification using both images and and a dataframe using Python?,” Stack Overflow. https://stackoverflow.com/questions/76930135/is-it-possible-to-do-a-cnn-for-classification-using-both-images-and-and-a-datafr (accessed Feb. 02, 2024).
‌
[4]“Datasets & DataLoaders — PyTorch Tutorials 1.11.0+cu102 documentation,” pytorch.org. https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
‌
[5]L. Liu, X. Liu, J. Gao, W. Chen, and J. Han, “Understanding the Difficulty of Training Transformers,” arXiv:2004.08249 [cs, stat], Sep. 2020, Available: https://arxiv.org/abs/2004.08249
‌
[6]“predict-wind-speeds-of-tropical-storms-rank-70.ipynb · master · DrivenData Competitions / Wind-dependent Variables Predict Wind Speeds of Tropical Storms · GitLab,” GitLab, Feb. 04, 2021. https://gitlab.com/drivendata-competitions/wind-dependent-variables-predict-wind-speeds-of-tropical-storms/-/blob/master/predict-wind-speeds-of-tropical-storms-rank-70.ipynb?ref_type=heads (accessed Feb. 02, 2024).

[7]T. Glazer, “How to Use Deep Learning to Predict Tropical Storm Wind Speeds - Benchmark,” DrivenData Labs, Dec. 08, 2020. https://drivendata.co/blog/predict-wind-speeds-benchmark/ (accessed Feb. 02, 2024).

[8]“Leaving Google Colab,” colab.research.google.com. https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fdrivendata.co%2Fblog%2Fpredict-wind-speeds-benchmark%2F (accessed Feb. 02, 2024).
‌
‌
‌
‌

https://chat.openai.com/share/b1095dc8-3b40-4e94-8688-e0ad846434f0 

https://chat.openai.com/share/42d73282-9a07-48f5-98da-ea9515e976b1 

https://chat.openai.com/share/9bfe7fea-3a03-4121-a7da-8a65bde0c045 

https://chat.openai.com/share/2f54c480-c5ee-44ac-8c04-082d88de1301 

https://chat.openai.com/share/ca25c216-9a8f-440a-ae0b-334756f70f0f  

https://chat.openai.com/share/0ce3de92-2309-43fe-be2a-84312c6a802a 
 
https://chat.openai.com/share/065ed735-a8cb-4e3f-ade6-e7ec2772707e 


## Authors:
Made with ❤️ by: Shrreya Behll, Mayeul Godinot, Nicholas Teo, Yuxuan Wen, Sihan Cao, Geyu Ji, Khunakorn limpsapapkasiphol, Oscar Dong
