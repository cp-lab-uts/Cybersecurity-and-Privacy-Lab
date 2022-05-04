# Fairness Regularizers in Semi-supervised Learning

Authors: **Tao Zhang**, Tianqing Zhu, Kun Gao, Wanlei Zhou

This is part of code for my paper 'Fairness in Graph-based Semi-supervised Learning'. 

Paper Link: https://arxiv.org/abs/2009.06190


## Introduction
Fairness in machine learning has received considerable attention. However, most studies on fair learning focus on either supervised learning or unsupervised learning. Very few consider semi-supervised settings. Yet, in reality, most machine learning tasks rely on large datasets that contain both labeled and unlabeled data.
Recent  study  has  proved  that  increasing  the  size  oftraining  (labeled)  data  will  promote  the  fairness  criteria  withmodel  performance  being  maintained.  In  this  work,  we  aim  toexplore  a  more  general  case  where  quantities  of  unlabeled  data are provided, indeed leading to a new form of learning paradigm,namely  fair  semi-supervised  learning.  Taking  the  popularity  ofgraph-based  approaches  in  semi-supervised  learning,  we  study this  problem  both  on  conventional  label  propagation  methodand  graph  neural  networks,  where  various  fairness  criteria  canbe  flexibly  integrated.  Our  developed  algorithms  are  provedto  be  non-trivial  extensions  to  the  existing  supervised  modelswith  fairness  constraints.  More  importantly,  we  theoretically demonstrate the source of discrimination in fair semi-supervised learning  problems  via  bias,  variance  and  noise  decomposition. Extensive  experiments  on  real-world  datasets  exhibit  that  ourmethods achieve a better trade-off between classification accuracy and  fairness  than  the  compared  baselines.

## Contribution
First, we conduct the study of algorithmic fairness in thesetting of graph-based SSL, including graph-based regularizations and graph neural networks. These approaches enable the use of unlabeled data to achieve a better trade-off between fairness and accuracy.

Second, we  propose  algorithms  to  solve  optimizationproblems  when  disparate  impact  and  disparate  mistreat-ment are integrated as fairness metrics in the graph-based regularization.

Third, we consider different cases of fairness constraintson labeled and unlabeled data. This helps us understandthe impact of unlabeled data on model fairness, and howto control the fairness level in practice.

Forth, we theoretically analyze the sources of discrimina-tion in SSL to explain why unlabeled data help to reacha  better  trade-off.  We  conduct  extensive  experiments  tovalidate the effectiveness of our proposed methods.


## Method 1
Our first approach,  fair  semi-supervised  margin  classifiers  (FSMC),  isformulated  as  an  optimization  problem,  where  the  objectivefunction  includes  a  loss  for  both  the  classifier  and  label propagation,  and  fairness  constraints  over  labeled  and  unlabeled  data.  Classification  loss  is  to  optimize  the  accuracy  oftraining result; label propagation loss is to optimize the label predictions on unlabeled data; the fairness constraint is to lead optimization towards to a fairness direction.  In  this  way,  labeled  and unlabeled data are used to achieve a better trade-off between accuracy and fairness.

## Method 2
Our second approach,  fair graph neural networks (FGNN), is built with GNNs, where the loss function includes classification loss and fairness loss. Classification loss optimizes the classification accuracy over all labeled data, and fairness loss enforces fairness over labeled data and unlabeled data.  GNN  models  combine  graph  structures  and  features, and  our  method  allows  GNN  models  to  distribute  gradient information from the classification loss and fairness loss. Thus, fair representations of nodes with labeled and unlabeled datacan be learned to achieve the ideal trade-off between accuracy and fairness.

## Requirements
Python 3.6<br>
Pytorch 1.2<br>
Pandas 1.2.0<br>
Numpy 1.18.0<br>

## Getting started

Fairness regularizers and data pre-processing are given in the utils.py.

Graph neural network is defined in the model.py and layer.py.

The training process is in the train.py. 

Fairness metrics.py is used to evaluate discrimination level with demographic parity and equal opportunity.

The following is an example to execute train.py.

python train.py --lr=0.005 --fare=1 --fair_metric=1 --alpha=0.5 --num_unlabel=400 --num_labeled=1000 

## Datasets

Three dataset are used in this paper, and the links are given in the following.

Bank dataset: https://archive.ics.uci.edu/ml/datasets/bank+mar

Health dataset: https://foreverdata.org/1015/index.html

Titanic dataset: https://www.kaggle.com/c/titanic/data

## Evaluation 
From these experiments, we can obtain some conclusions. 1) The proposed framework can make use of unlabeled data to achieve a better trade-off between accuracy and discrimination. 2) Under the fairness metric of disparate impact, the fairness constraint on mixed labeled and unlabeled
data generally has the best trade-off between accuracy and discrimination. Under the fairness metric of disparate mistreatment, the fairness constraint on labeled data is used to achieve the trade-off between accuracy and discrimination. 3) More unlabeled data generally helps to make a better compromise between accuracy and discrimination. 4) Model choice can affect the trade-off between accuracy and discrimination. Our experiments show that SVM is more friendly to achieve a better trade-off than LR.
