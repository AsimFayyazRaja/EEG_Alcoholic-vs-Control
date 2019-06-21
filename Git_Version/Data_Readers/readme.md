## Introduction

- EEG samples are given for different subjects
- Multiple trials are done per subject
- The shape of data is (11097,64,256)
    * 11097 is the number of EEG recordings taken from each subject's each trial
    * Data is recorded from 64 electrodes
    * In each samples, each electrode gives 256 values i.e. sampling frequency is 256

- In this section data is taken from files and converted to pickle files of numpy arrays
