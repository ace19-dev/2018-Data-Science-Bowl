# Kaggle Competitions

## 2018-Data-Science-Bowl
- [Find the nuclei in divergent images to advance medical discovery](https://www.kaggle.com/c/data-science-bowl-2018)

## Description
- The 2018 Data Science Bowl offers our most ambitious mission yet: create an algorithm to automate nucleus detection.
- By automating nucleus detection, you could help unlock cures faster—from rare disorders to the common cold.
- Identifying the cells’ nuclei is the starting point for most analyses because most of the human body’s 30 trillion cells contain a nucleus full of DNA, 
the genetic code that programs each cell.
- teams will work to automate the process of identifying nuclei, which will allow for more efficient drug testing, 
shortening the 10 years it takes for each new drug to come to market. 
- Teams will create a computer model that can identify a range of nuclei across varied conditions. 

## Evaluation
[https://www.kaggle.com/c/data-science-bowl-2018#evaluation](https://www.kaggle.com/c/data-science-bowl-2018#evaluation)

## Submission File
- In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. 
- you will submit pairs of values that contain a start position and a run length. 
E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).
- The competition format requires a space delimited list of pairs. 
For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. 
The pixels are one-indexed and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.
- The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.
- The file should contain a header and have the following format. 
Each row in your submission represents a single predicted nucleus segmentation for the given ImageId.
```
ImageId,EncodedPixels  
0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5,1 1  
0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac,1 1  
0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac,2 3 8 9  
etc...
```

## About
- The Data Science Bowl, presented by Booz Allen and Kaggle, is the world’s premier data science for social good competition.
- The Data Science Bowl brings together data scientists, technologists, domain experts, 
and organizations to take on the world’s challenges with data and technology. 
- During a 90-day period, participants, either alone or working in teams, 
gain access to unique data sets to develop algorithms that address a specific challenge. 

## Timeline
- April 9th, 2018 - Entry deadline and Team merger deadline.
- April 11, 2018 - Stage one deadline and stage two data release. Your model must be finalized and uploaded to Kaggle by this deadline.
- April 16, 2018 - Final submission deadline.

## Reference
- [Generic U-Net Tensorflow implementation for image segmentation](https://github.com/jakeret/tf_unet) - hong, sean
- [A concise code for training and evaluating Unet using tensorflow+keras](https://github.com/zizhaozhang/unet-tensorflow-keras) -
- [Implementation of Segnet, FCN, UNet and other models in Keras](https://github.com/divamgupta/image-segmentation-keras) -
- [Dilated U-net](https://chuckyee.github.io/cardiac-segmentation/)
- https://github.com/preritj/segmentation


## Quick Start
- Firstly, you can make ground truth mask.
  - python make_ground_truth.py
- Second
  - python train.py



## Idea
- check image and label file
- Augumentation (by using elastice depormation -> flip, bright etc.) (ing)
- using ensemble (hong) / 
- modify hyper-parameter
- Change network to Dilated U-net (sungil)

