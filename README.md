# Open Sleep-Wake Scorer

This project is dedicated to building a machine-learning based classifier for sleep-wake classification for mouse model of alzheimer's disease. It is an ongoing effort to achieve a satisfiable performance with balanced accuracy across 3 classes.

We implemented 3 machine learning models and 2 deep learning models for sleep staging on our collected dataset. We report the results in our written report. For reimplementation, please see the corresponding notebook to find the our pipelines. cnn_transformer.ipynb contains the cnn+transformer model proposed in our report. SVM_RF.ipynb contains the support vector machine and random forest models as the baseline. The multinomial_logitic_regression.R is the R script for multinomial regression. 

 Due to the size the dataset, the data used by these pipelines are not included in this repository. We therefore share the data using google links:

 raw data: https://drive.google.com/file/d/1_qGBp_afq7QhovnS-PD1F91FSF4dkeMH/view?usp=drive_link
 processed data: https://drive.google.com/file/d/1vKHwSEf8NdUHPI6p1rAffCSR82Xv9Yzh/view?usp=drive_link

Our notebooks contain the necessary codes to replicate the confusion matrices presented in the report. Simply downloading the dataset and running the notebook should be sufficient.