install.packages("tidyverse")
install.packages("caret")
install.packages("lubridate")
install.packages("readr")
install.packages("icesTAF")
install.packages("knitr")
install.packages("VIM")
install.packages("naniar")
install.packages("Hmisc")
install.packages("corrplot")
install.packages('caTools')
install.packages("ROCR")
install.packages("PROC")


library("tidyverse")
library("caret")
library("lubridate")
library("readr")
library("icesTAF")
library("knitr")
library("ggplot2")
library("VIM")
library("naniar")
library("Hmisc")
library("corrplot")
library("caTools")
library(ROCR)
library(pROC)

.
# Import the datasets.
# 'wine' is the red and white wine dataset

wine <- read.csv("C:/Users/computer/Desktop/Teesside University/CIS4047-N-BF1-2021 Data Science Foundations/Accessment/20%/winequalityN.csv", sep = ',', header = TRUE)
head(wine, n = 10)
tail(wine, n = 3)

colnames(wine)        # View the headers
summary(wine)         # View Summary

str(wine)             #Check Datatypes
sum(is.na(wine))      #check for NULL values
is.na(wine)

# Missing Values Visualization
gg_miss_var(wine)
res<-summary(aggr(wine, sortVar=TRUE))$combinations

# Replace missing values with column mean
# put in a new dataframe
wineq <- wine
wineq$fixed.acidity[is.na(wineq$fixed.acidity)]<-mean(wineq$fixed.acidity,na.rm=TRUE)
wineq$pH[is.na(wineq$pH)]<-mean(wineq$pH,na.rm=TRUE)
wineq$volatile.acidity[is.na(wineq$volatile.acidity)]<-mean(wineq$volatile.acidity,na.rm=TRUE)
wineq$sulphates[is.na(wineq$sulphates)]<-mean(wineq$sulphates,na.rm=TRUE)
wineq$citric.acid[is.na(wineq$citric.acid)]<-mean(wineq$citric.acid,na.rm=TRUE)
wineq$residual.sugar[is.na(wineq$residual.sugar)]<-mean(wineq$residual.sugar,na.rm=TRUE)
wineq$chlorides[is.na(wineq$chlorides)]<-mean(wineq$chlorides,na.rm=TRUE)

#check for missing values again
sum(is.na(wineq))
summary(wineq)


hist.data.frame(wineq)

#Scatterplot Matrix of Variables
plot(wineq)

# Use only numerical values
winequ <- wineq[, -1]

head(wineq)

#Correlation Heatmap of Variables
corrplot(cor(winequ))

# Correlations
corrplot(cor(winequ), method = "number",
         title = "Correlation Plot",
         tl.pos = "n", mar = c(2, 1, 3, 1))

#Distribution of wine quality ratings
ggplot(wineq,aes(x=quality))+geom_bar(stat = "count",position = "dodge")+
  scale_x_continuous(breaks = seq(3,8,1))+
  ggtitle("Distribution of Wine Quality Ratings")+
  theme_classic()

#Distribution of white/red wines
ggplot(wineq,aes(x=type,fill=factor(type)))+geom_bar(stat = "count",position = "dodge")+ #scale_y_continuous(labels=scales::percent, limits=(10, 100))+
  #scale_x_continuous(breaks = seq(0,1,1),limits=(10, 1000))+
  ggtitle("Distribution of White/Red Wines")+
  theme_classic()

#Distribution of quality for white/red wines
ggplot(wineq, aes(quality,fill=type, type=c("white"))) +
  geom_histogram(binwidth = .5,col="black") +  
  facet_grid(type ~ .)+
  labs(title="Histogram Showing Qulity of Wine", 
       subtitle="Wine Quality across Red and White colors of Wine")

str(wineq)

#Will give us column number which might insignificant variance
nzv <- nearZeroVar(wineq)
print(paste("---Column number with----", nzv))

# Encoding the type column
wineq$type = factor(wineq$type, 
                         levels = c('white','red'), 
                         labels = c(1.0, 2.0))
str(wineq)

# Splitting the dataset 70/30
set.seed(123)
split = sample.split(wineq$quality, SplitRatio = 0.7)

#Creating the training set and test set separately
#training_set = subset(wineq, split == TRUE)
#test_set = subset(wineq, split == FALSE)
#training_set
#test_set

# Create Normalize function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Normalize
wineq.norm<- as.data.frame(lapply(wineq[,2:12], normalize))





#Rename
winee<-wineq.norm
head(winee)

plotViewport(winee$quality)

#If quality is greater than 6 then wine is good else wine is Bad
winee$quality <- ifelse (as.integer(wineq$quality) > 6, 1, 0)

#Distribution of quality revamped

ggplot(winee,aes(x=quality))+geom_bar(stat = "count",position = "dodge")+
  scale_x_continuous(breaks = seq(3,8,1))+
  ggtitle("Distribution of Wine Quality Revamped")+
  theme_classic()

#Downsampling
set.seed(47)
winef <- winee
winef$quality<-as.factor(winef$quality)
winef<- downSample(x = winef[,-12], y = winef$quality, list = F, yname = "quality")

#Distribution of quality revamped after downsampling

ggplot(winef,aes(x=quality))+geom_bar(stat = "count",position = "dodge")+
  #scale_x_continuous(breaks = seq(3,8,1))+
  ggtitle("Distribution of Wine Quality Downsampling")+
  theme_classic()

#winee$type<-wineq$type
#winee$quality<-wineq$quality



summary(winef)
head(winee)

#Applying the SVM Machine Learning Technique

intrain <- createDataPartition(y = winef$quality, p= 0.7, list = FALSE)
training <- winef[intrain,]
testing <- winef[-intrain,]
 dim(testing)
 
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
install.packages("e1071")
library("e1071")

svm_linear = svm(formula = quality ~ .,
                 data = training,
                 type = 'C-classification',
                 kernel = 'linear')
#Predicting SVM

pred_svm_linear=predict(svm_linear,testing)

#Confusion Matric for SVM
confusionMatrix(pred_svm_linear,testing$quality)
cm_SVM <- table(pred_svm_linear,testing$quality)

#Calculating other Metrices
n_svm = sum(cm_SVM) # number of instances
nc_svm = nrow(cm_SVM) # number of classes
diag_svm = diag(cm_SVM) # number of correctly classified instances per class 
rowsums_svm = apply(cm_SVM, 1, sum) # number of instances per class
colsums_svm = apply(cm_SVM, 2, sum) # number of predictions per class
p_svm = rowsums_svm / n_svm # distribution of instances over the actual classes
q_svm = colsums_svm / n_svm # distribution of instances over the predicted classes


diag_svm
rowsums_svm
colsums_svm


accuracy_svm = sum(diag_svm) / n_svm 

accuracy_svm

precision_svm = diag_svm / colsums_svm 
recall_svm = diag_svm / rowsums_svm 
f1_svm = 2 * precision_svm * recall_svm / (precision_svm + recall_svm) 
data.frame(precision_svm, recall_svm, f1_svm)


# Plotting ROC Curve

ROCRpredsvm = prediction(as.numeric(pred_svm_linear), testing$quality)
ROCRperfsvm = performance(ROCRpredsvm, "tpr", "fpr")
aucsvm <- slot(performance(ROCRpredsvm, "auc"), "y.values")[[1]] # Area Under Curve
plot(ROCRperfsvm, colorize=TRUE)
abline(h=seq(0,1,0.05), v=seq(0,1,0.05), col = "lightgray", lty = "dotted")
lines(c(0,1),c(0,1), col = "gray", lwd =2)
text(0.6,0.2,paste("AUC=", round(aucsvm,4), sep=""), cex=1.4)
title("ROC SVM")

#Parameter Tuning
svm_pt <- svm(quality ~., data = training, kernel  = "radial",
              cost = 1, gamma = 0.04545455, epsilon = 0.1)
pred_svm_pt=predict(svm_pt, testing)

confusionMatrix(testing$quality, pred_svm_pt)
cm_SVM_pt <- table(testing$quality, pred_svm_pt)

#Calculating other Metrices for SVM Parameter Tuning
n_SVM_pt = sum(cm_SVM_pt) # number of instances
nc_SVM_pt = nrow(cm_SVM_pt) # number of classes
diag_SVM_pt = diag(cm_SVM_pt) # number of correctly classified instances per class 
rowsums_SVM_pt = apply(cm_SVM_pt, 1, sum) # number of instances per class
colsums_SVM_pt = apply(cm_SVM_pt, 2, sum) # number of predictions per class
p_SVM_pt = rowsums_SVM_pt / n_SVM_pt # distribution of instances over the actual classes
q_SVM_pt = colsums_SVM_pt / n_SVM_pt # distribution of instances over the predicted classes


diag_SVM_pt
rowsums_SVM_pt
colsums_SVM_pt


accuracy_SVM_pt = sum(diag_SVM_pt) / n_SVM_pt 

accuracy_SVM_pt

precision_SVM_pt = diag_SVM_pt / colsums_SVM_pt 
recall_SVM_pt = diag_SVM_pt / rowsums_SVM_pt 
f1_SVM_pt = 2 * precision_SVM_pt * recall_SVM_pt / (precision_SVM_pt + recall_SVM_pt) 
data.frame(precision_SVM_pt, recall_SVM_pt, f1_SVM_pt)

#Comparing ROC Curve Before and adter Parameter tuning

test_auc <- function(prob) {
  roc(testing$quality, prob)
}

auc_svm <- test_auc(as.numeric(pred_svm_linear))

auc_svm_pt <- test_auc(as.numeric(pred_svm_pt))


df_auc_svm <- bind_rows(data_frame(TPR = auc_svm$sensitivities, 
                                FPR = 1 - auc_svm$specificities, 
                                Model = "SVM"),
                        data_frame(TPR = auc_svm_pt$sensitivities, 
                                FPR = 1 - auc_svm_pt$specificities, 
                                Model = "SVM_PT"))

df_auc_svm %>% 
  ggplot(aes(FPR, TPR, color = Model)) +
  geom_line(size = 1) +
  theme_bw() +
  coord_equal() +
  geom_abline(intercept = 0, slope = 1, color = "gray37", size = 1, linetype = "dashed") + 
  labs(x = "FPR (1 - Specificity)", 
       y = "TPR (Sensitivity)", 
       title = "ROC Curve and AUC: SVM VS SVM Parameter Tuned")





#Random Forest

#Building the decision tree
install.packages("rpart")
install.packages("rpart.plot")
install.packages("randomForest")
install.packages("rattle")
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(rattle)

training$quality<-as.factor(training$quality)
testing$quality<-as.factor(testing$quality)

summary(training)

#Applying Random forest Model
rf <-rpart(training$quality~., data=training)

#Making predictions 
pred_rf <- predict(rf, testing, type="class")

#Random Forest Confusion Matrix
confusionMatrix(as.factor(pred_rf),testing$quality)
cm_rf <- table(as.factor(pred_rf),testing$quality)

#ROC for Random Forest
par(mfrow=c(1,1))
ROCRpredrf = prediction(as.numeric(pred_rf), testing$quality)
ROCRperfrf = performance(ROCRpredrf, "tpr", "fpr")
aucrf <- slot(performance(ROCRpredrf, "auc"), "y.values")[[1]] # Area Under Curve
plot(ROCRperfrf, colorize=TRUE)
abline(h=seq(0,1,0.05), v=seq(0,1,0.05), col = "lightgray", lty = "dotted")
lines(c(0,1),c(0,1), col = "gray", lwd =2)
text(0.6,0.2,paste("AUC=", round(aucrf,4), sep=""), cex=1.4)
title("ROC Random Forest")

#Calculating other Metrices

n_rf = sum(cm_rf) # number of instances
nc_rf = nrow(cm_rf) # number of classes
diag_rf = diag(cm_rf) # number of correctly classified instances per class 
rowsums_rf = apply(cm_rf, 1, sum) # number of instances per class
colsums_rf = apply(cm_rf, 2, sum) # number of predictions per class
p_rf = rowsums_rf / n_rf # distribution of instances over the actual classes
q_rf = colsums_rf / n_rf # distribution of instances over the predicted classes


diag_rf
rowsums_rf
colsums_rf


accuracy_rf = sum(diag_rf) / n_rf 

accuracy_rf

precision_rf= diag_rf / colsums_rf 
recall_rf = diag_rf / rowsums_rf 
f1_rf = 2 * precision_rf * recall_rf / (precision_rf + recall_rf) 
data.frame(precision_rf, recall_rf, f1_rf)



# Random Forest parameter tuning
rf_pt <- randomForest(quality ~., data = training, ntree  = 10000,
                 mtry= 10, importance = TRUE, PROXIMITY=TRUE)


# Predict RF PT
predrf_pt=predict(rf_pt,testing)

#Confusion Matrix for RF PT
confusionMatrix(predrf_pt,testing$quality)

cm_rf_pt <- table(predrf_pt,testing$quality)

#Calculating other Metrices for Random Forest Parameter Tuning
n_rf_pt = sum(cm_rf_pt) # number of instances
nc_rf_pt = nrow(cm_rf_pt) # number of classes
diag_rf_pt = diag(cm_rf_pt) # number of correctly classified instances per class 
rowsums_rf_pt = apply(cm_rf_pt, 1, sum) # number of instances per class
colsums_rf_pt = apply(cm_rf_pt, 2, sum) # number of predictions per class
p_rf_pt = rowsums_rf_pt / n_rf_pt # distribution of instances over the actual classes
q_rf_pt = colsums_rf_pt / n_rf_pt # distribution of instances over the predicted classes


diag_rf_pt
rowsums_rf_pt
colsums_rf_pt


accuracy_rf_pt = sum(diag_rf_pt) / n_rf_pt 

accuracy_rf_pt

precision_rf_pt = diag_rf_pt / colsums_rf_pt 
recall_rf_pt = diag_rf_pt / rowsums_rf_pt 
f1_rf_pt = 2 * precision_rf_pt * recall_rf_pt / (precision_rf_pt + recall_rf_pt) 
data.frame(precision_rf_pt, recall_rf_pt, f1_rf_pt)

#Comparing ROC Curve Before and after Parameter tuning for Random Forest

test_auc <- function(prob) {
  roc(testing$quality, prob)
}

auc_rf <- test_auc(as.numeric(pred_rf))

auc_rf_pt <- test_auc(as.numeric(predrf_pt))


df_auc_rf <- bind_rows(data_frame(TPR = auc_rf$sensitivities, 
                                   FPR = 1 - auc_rf$specificities, 
                                   Model = "RANDOM FOREST"),
                        data_frame(TPR = auc_rf_pt$sensitivities, 
                                   FPR = 1 - auc_rf_pt$specificities, 
                                   Model = "RANDOM FOREST PT"))

df_auc_rf%>% 
  ggplot(aes(FPR, TPR, color = Model)) +
  geom_line(size = 1) +
  theme_bw() +
  coord_equal() +
  geom_abline(intercept = 0, slope = 1, color = "gray37", size = 1, linetype = "dashed") + 
  labs(x = "FPR (1 - Specificity)", 
       y = "TPR (Sensitivity)", 
       title = "ROC Curve and AUC: Random Forest VS Random Forest Parameter Tuned")






#Logistic Regression model

log <- glm(formula = quality ~., data=training, family = "binomial")

summary(log)

#Predict Model for Logistic Regression

pred_log<- predict(log, testing, type = "response")

predicted_log<-ifelse(pred_log> 0.5,1,0)

#COnfusion Matrix for Logistic Regression

confusionMatrix(as.factor(predicted_log), testing$quality)

cm_log <- table(data = as.factor(predicted_log), reference = testing$quality)

cm_log

#Calculating other Metrices for Logistic Regression
n_log = sum(cm_log) # number of instances
nc_log = nrow(cm_log) # number of classes
diag_log = diag(cm_log) # number of correctly classified instances per class 
rowsums_log = apply(cm_log, 1, sum) # number of instances per class
colsums_log = apply(cm_log, 2, sum) # number of predictions per class
p_log = rowsums_log / n_log # distribution of instances over the actual classes
q_log = colsums_log / n_log # distribution of instances over the predicted classes


diag_log
rowsums_log
colsums_log


accuracy_log = sum(diag_log) / n_log 

accuracy_log

precision_log = diag_log / colsums_log 
recall_log = diag_log / rowsums_log 
f1_log = 2 * precision_log * recall_log / (precision_log + recall_log) 
data.frame(precision_log, recall_log, f1_log)



#ROC Curve for Logistic Regression

par(mfrow=c(1,1))
ROCRpred = prediction(predicted_log, testing$quality)
ROCRperf = performance(ROCRpred, "tpr", "fpr")
auc <- slot(performance(ROCRpred, "auc"), "y.values")[[1]] # Area Under Curve
plot(ROCRperf, colorize=TRUE)
abline(h=seq(0,1,0.05), v=seq(0,1,0.05), col = "lightgray", lty = "dotted")
lines(c(0,1),c(0,1), col = "gray", lwd =2)
text(0.6,0.2,paste("AUC=", round(auc,4), sep=""), cex=1.4)
title("ROC Logistic Regression")

# Parameter Tuning for Logistic Regression 
#1 Try out the train function to see if 'parameter' gets tuned
set.seed(1)
log_pt <- train(quality ~., data=training, method='glm')
log_pt
predlog_pt<- predict(log_pt, testing, type = "raw")
confusionMatrix(data = as.factor(predlog_pt), reference = testing$quality)


#2 try another parameter tuning logistic regression

set.seed(123)
log_pt2 <- train(
  quality ~., 
  data = training, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)
predlog_pt2<- predict(log_pt2, testing)
confusionMatrix(data = as.factor(predlog_pt2), reference = testing$quality)

#3 try another parameter tuning logistic regression

log_pt3 <- glm(formula = quality ~., data=training, family = "binomial")

summary(log_pt3)

#Predict Model for Logistic Regression Parameter Tuning 3

pred_log_pt3<- predict(log_pt3, testing, type = "response")

predicted_log_pt3<-ifelse(pred_log_pt3> 0.45,1,0)

#COnfusion Matrix for Logistic Regression parameter Tuning 3

confusionMatrix(as.factor(predicted_log_pt3), testing$quality)

cm_log_pt3 <- table(data = as.factor(predicted_log_pt3), reference = testing$quality)

cm_log_pt3

#Calculating other Metrices for Logistic Regression Parameter Tuning 3
n_log_pt3 = sum(cm_log_pt3) # number of instances
nc_log_pt3 = nrow(cm_log_pt3) # number of classes
diag_log_pt3 = diag(cm_log_pt3) # number of correctly classified instances per class 
rowsums_log_pt3 = apply(cm_log_pt3, 1, sum) # number of instances per class
colsums_log_pt3 = apply(cm_log_pt3, 2, sum) # number of predictions per class
p_log_pt3 = rowsums_log_pt3 / n_log_pt3 # distribution of instances over the actual classes
q_log_pt3 = colsums_log_pt3 / n_log_pt3 # distribution of instances over the predicted classes


diag_log_pt3
rowsums_log_pt3
colsums_log_pt3


accuracy_log_pt3 = sum(diag_log_pt3) / n_log_pt3 

accuracy_log_pt3

precision_log_pt3 = diag_log_pt3 / colsums_log_pt3 
recall_log_pt3 = diag_log_pt3 / rowsums_log_pt3 
f1_log_pt3 = 2 * precision_log_pt3 * recall_log_pt3 / (precision_log_pt3 + recall_log_pt3) 
data.frame(precision_log_pt3, recall_log_pt3, f1_log_pt3)

#Comparing ROC Curve Before and after Parameter tuning for Logistic Regression

test_auc <- function(prob) {
  roc(testing$quality, prob)
}

auc_log <- test_auc(as.numeric(pred_rf))

auc_log_pt <- test_auc(as.numeric(predrf_pt))


df_auc_log <- bind_rows(data_frame(TPR = auc_log$sensitivities, 
                                  FPR = 1 - auc_log$specificities, 
                                  Model = "LOGISTIC REGRESSION"),
                       data_frame(TPR = auc_log_pt$sensitivities, 
                                  FPR = 1 - auc_log_pt$specificities, 
                                  Model = "LOGISTIC REGRESSION PT"))

df_auc_log%>% 
  ggplot(aes(FPR, TPR, color = Model)) +
  geom_line(size = 1) +
  theme_bw() +
  coord_equal() +
  geom_abline(intercept = 0, slope = 1, color = "gray37", size = 1, linetype = "dashed") + 
  labs(x = "FPR (1 - Specificity)", 
       y = "TPR (Sensitivity)", 
       title = "ROC Curve and AUC: Random Forest VS Random Forest Parameter Tuned")


#Comparing SVM, Random Forest and Logistic regression after Parameter Tuning.

test_auc <- function(prob) {
  roc(testing$quality, prob)
}

auc_svm <- test_auc(as.numeric(pred_svm_linear))

auc_rf_pt <- test_auc(as.numeric(pred_rf))

auc_log <- test_auc(as.numeric(pred_log))


df_auc <- bind_rows(data_frame(TPR = auc_svm$sensitivities, 
                                   FPR = 1 - auc_svm$specificities, 
                                   Model = "SVM"),
                        data_frame(TPR = auc_log_pt$sensitivities, 
                                   FPR = 1 - auc_log_pt$specificities, 
                                   Model = "RANDOM FOREST"),
                        data_frame(TPR = auc_log$sensitivities, 
                                   FPR = 1 - auc_log$specificities, 
                                   Model = "LOGISTIC REGRESSION"))

df_auc%>% 
  ggplot(aes(FPR, TPR, color = Model)) +
  geom_line(size = 1) +
  theme_bw() +
  coord_equal() +
  geom_abline(intercept = 0, slope = 1, color = "gray37", size = 1, linetype = "dashed") + 
  labs(x = "FPR (1 - Specificity)", 
       y = "TPR (Sensitivity)", 
       title = "ROC Curve and AUC: SVM Vs Random Forest VS LOGISTIC REGRESSION")



#Comparing SVM, Random Forest and Logistic regression before Parameter Tuning.

test_auc <- function(prob) {
  roc(testing$quality, prob)
}

auc_svm_pt <- test_auc(as.numeric(pred_svm_pt))

auc_rf_pt<- test_auc(as.numeric(predrf_pt))

auc_log_pt <- test_auc(as.numeric(pred_log_pt3))


df_auc_pt <- bind_rows(data_frame(TPR = auc_svm_pt$sensitivities, 
                               FPR = 1 - auc_svm_pt$specificities, 
                               Model = "SVM_PT"),
                    data_frame(TPR = auc_rf_pt$sensitivities, 
                               FPR = 1 - auc_rf_pt$specificities, 
                               Model = "RANDOM FOREST_PT"),
                    data_frame(TPR = auc_log_pt$sensitivities, 
                               FPR = 1 - auc_log_pt$specificities, 
                               Model = "LOGISTIC REGRESSION_PT3"))


df_auc_pt%>% 
  ggplot(aes(FPR, TPR, color = Model)) +
  geom_line(size = 1) +
  theme_bw() +
  coord_equal() +
  geom_abline(intercept = 0, slope = 1, color = "gray37", size = 1, linetype = "dashed") + 
  labs(x = "FPR (1 - Specificity)", 
       y = "TPR (Sensitivity)", 
       title = "ROC Curve and AUC: SVM_PT Vs RANDOM FOREST_PT VS LOGISTIC REGRESSION_PT")
