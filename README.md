# fashion-compatibility

This repository contains a Pytorch fashion compatibility model.This Pytorch implementation is built on the [mvasil's fashion-compatibility](https://github.com/mvasil/fashion-compatibility) and [rxtan2's Learning-Similarity-Conditions](https://github.com/rxtan2/Learning-Similarity-Conditions).There are some differences between the implementations. In particular, this Pytorch version support

- uses Resnet50 to extract the image's generalized feature.

- has a mask branch to get the cloth's shape feature.

- uses the whole outfit and multiple negative sample to calculate margin ranking loss.


## Requirments

pytorch==1.2.0

torchvision==0.2.1

numpy==1.16.4

PIL==6.2.0


## Installation

1. Clone this repository
    
       git clone https://github.com/zuoxiang95/fashion-compatibility.git
    
2. Get the train data

   There are two kinds of data.
  
   **Polyvore data:**
  
   **Alibaba's dida data:**
  
3. Train your model

    You can begin to train your own model with the command:
        
        python main.py --name 'your own training name'
        
4. Test your model
  
       python test_in_polyvore.py