Practical Machine Learning Project
==================================



## Introduction

Goal of this project is to analyze data about personal activity using monitoring devices. Such data are used for reasons like finding patterns in behavior or improving one's health. In this study we use data measured from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The purpose is to build a machine learning algorithm to predict the manner in which a person does an exercise.

## Data Processing

The data for this project come from [http://groupware.les.inf.puc-rio.br/har]. Two data sets were used in order to build the classification algorithm, a [training set](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) set and a [test set](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). The target variable describing the manner in which each person did the exercise is the "classe" variable. Classe was converted to a factor variable with values "A", "B", "C", "D" and "E". The data set contains 159 other variables that can be used as predictors. In order to conclude to a final data set for building the machine learning algorithm, the first step was to exclude variables with no predictive value. First, we removed variables related to the structure of the data (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window). Next, values with high percentage of missing values were discarded to end up to a data set consisting of 53 variables including classe.


```r
testing <- read.csv("pml-testing.csv", stringsAsFactors=FALSE)  # load test set
testing$user_name <- factor(testing$user_name)
testing$new_window <- factor(testing$new_window)

training <- read.csv("pml-training.csv", stringsAsFactors=FALSE)# load training set
training$user_name <- factor(training$user_name)
training$new_window <- factor(training$new_window)
training$classe <- factor(training$classe)

training2 <- training[,which(colSums(!is.na(training))/nrow(training)>0.5)]  # exclude variables with high percentage of missing data
training3 <- subset(training2, select=-c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))	# exclude variables with no predictive power
training4 <- subset(training3, select=-c(kurtosis_yaw_belt, skewness_yaw_belt, amplitude_yaw_belt, kurtosis_picth_dumbbell, kurtosis_yaw_dumbbell, skewness_yaw_dumbbell, amplitude_yaw_dumbbell, kurtosis_yaw_forearm, skewness_yaw_forearm, amplitude_yaw_forearm, kurtosis_roll_belt, kurtosis_picth_belt, skewness_roll_belt, skewness_roll_belt.1, max_yaw_belt, min_yaw_belt, kurtosis_roll_arm, kurtosis_picth_arm, kurtosis_yaw_arm, skewness_roll_arm, skewness_pitch_arm, skewness_yaw_arm, kurtosis_roll_dumbbell, skewness_roll_dumbbell, skewness_pitch_dumbbell, max_yaw_dumbbell, min_yaw_dumbbell, kurtosis_roll_forearm, kurtosis_picth_forearm, skewness_roll_forearm, skewness_pitch_forearm, max_yaw_forearm, min_yaw_forearm))	# exclude variables with missing values
```

## Classification Algorithm

In order to build the classification algorithm, we split the original training set in a train and a test set and left the original test set to be used for validation. The split was performed keeping 70% in the train set. A series of classification algorithms were applied using the caret package, including classification trees, linear discriminant analysis, boosted trees and random forests. The performance of each algorithm was evaluated with cross-validation using the test set. Random forest algorithm outperformed the other three algorithms with an accuracy of 99.27% on the test set. Boosted trees also had a high accuracy on the test set equal to 95.72%, while linear discriminant analysis and classification trees performed poorly with 69.82% and 55.23% accuracy respectively.


```r
set.seed(123)
inTrain <- createDataPartition(y=training4$classe, p=0.7, list=FALSE)
train <- training4[inTrain,]
test <- training4[-inTrain,]
set.seed(123)
modFit <- train(classe ~ ., method="rpart", data=train) # fit tree
confusionMatrix(test$classe, predict(modFit, test))
set.seed(123)
modFitLDA <- train(classe ~ ., method="lda", data=train)# fit linear discriminant analysis
confusionMatrix(test$classe, predict(modFitLDA, test))
set.seed(123)
modFitGBM <- train(classe ~ ., method="gbm", data=train)# fit boosted trees
confusionMatrix(test$classe, predict(modFitGBM, test))
set.seed(123)
modFitRF <- train(classe ~ ., method="rf", data=train)  # fit random forest
```

Based on these results, the algorithm that was chosen to fit best our purpose was random forests with a very low expected out of sample error, equal to 0.73%. Due to the high accuracy of the model on both the train and test set, which implies the absence of overfitting, no preprocessing like PCA or correlation analysis was considered necessary before the application of the algorithm. The confusion matrix and some statistics of the algorithm are presented below.


```r
confusionMatrix(test$classe, predict(modFitRF, test))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    7 1130    1    1    0
##          C    0    8 1015    3    0
##          D    0    0   17  946    1
##          E    0    0    2    1 1079
## 
## Overall Statistics
##                                        
##                Accuracy : 0.993        
##                  95% CI : (0.99, 0.995)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.991        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.992    0.981    0.995    0.999
## Specificity             1.000    0.998    0.998    0.996    0.999
## Pos Pred Value          0.999    0.992    0.989    0.981    0.997
## Neg Pred Value          0.998    0.998    0.996    0.999    1.000
## Prevalence              0.285    0.194    0.176    0.162    0.184
## Detection Rate          0.284    0.192    0.172    0.161    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.998    0.995    0.989    0.996    0.999
```

The final random forest algorithm was further applied on the separate [test set](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) which included 20 test cases and predicted all 20 cases correctly, validating the accuracy of our model. 

## Conclusions

In this project we presented a machine learning algorithm for predicting the way of performing barbell lifts based on data collected through monitoring devices. The algorithm uses random forests and predicts unknown cases with more than 99% accuracy. The algorithm can be used to help people quantify how well they do a specific activity.
