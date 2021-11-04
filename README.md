# EEG_Alcoholic-vs-Control
Alcoholic vs Control classification using EEG signals

## Paper acceptance
Paper was presented in [WSPML 2019](http://www.wspml.org/2019.html) in Bangkok. Best student paper award was given to this paper.
https://dl.acm.org/doi/10.1145/3387168.3389119

## Abstract
A lot of advancements have been made in the field of Brain Computer Interfaces (BCI) using machine and deep learning. This paper presents a novel preprocessing technique to process Electroencephalography (EEG) signals in time domain. The proposed methodology, (Peak Visualization Method) (PVM) is based on selecting peaks with distinctive width and height range in order to perform better classification. PVM uses multiple machine learning techniques such as Random Forest, Logistic Regression and Support Vector Machine (SVM), Naive Bayes in order to find the most discriminating ranges. Moreover, selected range peaks are further used to compute features like indices of peaks, prominence of peak, contour heights, relative maxima, relative minima, local maxima and local minima. The extracted features were used as training and test data for a competitive 5-fold cross-validated analysis with Long Short Term Memory (LSTM) network. A publicly available EEG dataset for alcoholic and non-alcoholic classification was used to compare the proposed technique with state of the art EEG-NET deep learning model. In order to visualize the generalized performance of proposed system we use award winning dimensionality reduction technique t-Distributed Stochastic Neighbor Embedding (t-SNE) on the features extracted by EEGLSTM and show how our model's activations are classified in alcoholic and non-alcoholic categories. The reduced features are visualized into two dimensions. Features extracted using PVM gives an average accuracy of 90% on 5 folds improving the current state of the art EEG-NET, which manages to achieve an average accuracy of 88% on this dataset.

## Task formulation
Using machine and deep learning technologies, the task is to classify EEG samples between alcoholic or control people

## Introduction

- This is the repository for keeping track of results and things done
- The datasets are not provided here
- Each folder has it's own markdown readme for further explanation
