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

## Evaluation (editing)
- This competition is evaluated on the mean average precision at different intersection over union (IoU) thresholds. 
- at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.
- A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. 
- A false positive indicates a predicted object had no associated ground truth object.
- A false negative indicates a ground truth object had no associated predicted object.
- The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold.
- Lastly, the score returned by the competition metric is the mean taken over the individual average precisions of each image in the test dataset.

## Submission File
- In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. 
- you will submit pairs of values that contain a start position and a run length. 
E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).
- The competition format requires a space delimited list of pairs. 
For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. 
The pixels are one-indexed and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.
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
- April 9th, 2018 - Entry deadline. Team merger deadline.
- April 11, 2018 - Stage one deadline and stage two data release. Your model must be finalized and uploaded to Kaggle by this deadline.
- April 16, 2018 - Final submission deadline.