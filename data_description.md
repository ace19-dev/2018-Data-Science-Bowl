## data description
- This dataset contains a large number of segmented nuclei images. 
- The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). 
- The dataset is designed to challenge an algorithm's ability to generalize across these variations.
- Each image is represented by an associated ImageId.
- Files belonging to an image are contained in a folder with this ImageId. Within this folder are two subfolders:
  - images contains the image file.
  - masks contains the segmented masks of each nucleus. This folder is only included in the training set. Each mask contains one nucleus. 
  Masks are not allowed to overlap (no pixel belongs to two masks).
- As with any human-annotated dataset, you may find various forms of errors in the data.
- You may manually correct errors you find in the training set. 
- The dataset will not be updated/re-released unless it is determined that there are a large number of systematic errors.
- The masks of the stage 1 test set will be released with the release of the stage 2 test set.

## File descriptions
- /stage1_train/* - training set images (images and annotated masks)
- /stage1_test/* - stage 1 test set images (images only, you are predicting the masks)
- /stage2_test/* (released later) - stage 2 test set images (images only, you are predicting the masks)
- stage1_sample_submission.csv - a submission file containing the ImageIds for which you must predict during stage 1
- stage2_sample_submission.csv (released later) - a submission file containing the ImageIds for which you must predict during stage 2
- stage1_train_labels.csv - a file showing the run-length encoded representation of the training images. 
This is provided as a convenience and is redundant with the mask image files.