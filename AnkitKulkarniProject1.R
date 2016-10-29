#Iris Dataset
#Naive Bayes on Iris Dataset
#load iris data and install the packages caret, klar and combinat with dependencies = TRUE
data("iris")
library(caret)
#contains the method nb for naive bayes
library(klaR)
library(combinat)
set.seed(0375)
#seed = first two and last two digits of UFID
sample_size <- floor(0.80*nrow(iris))
#sample size taken as 80% of the total number of records randomly
TrainInData <- sample(seq_len(nrow(iris)), size = sample_size)
train_data <- iris[TrainInData,]
test_data <- iris[-TrainInData,]
#The above statements give training data as 80% and test data as the remaining data of 20%
naivebayes <- train(Species~., data = train_data, method = "nb")
#nb is the naive bayes method in caret package
prediction <- predict(naivebayes, test_data)
summary(test_data)
#summary of test data to compare with the predicted output
cm <- confusionMatrix(prediction, test_data$Species,  positive = NULL)
cm
plot(naivebayes)
#the above code gives confusion matrix along with accuracy




#JRipper on Iris Dataset
#load iris data and install packages RWeka and caret with dependencies = TRUE
library(RWeka)
#contains the method JRip
library(caret)
data(iris)
#the iris dataset gets loaded in memory
set.seed(0375)
#seed = first 2 and last 2 digits of UFID
sample_size <- floor(0.80*nrow(iris))
#sampling of data into 80% training data and 20% test data
TrainInData <- sample(seq_len(nrow(iris)), size = sample_size)
train_data <- iris[TrainInData,]
test_data <- iris[-TrainInData,]
#divide the training and test data into attributes and classes
TrainData <- train_data[,1:4]
TrainClass <- train_data[,5]
TestData <- test_data[,1:4]
TestClass <- test_data[,5]
#JRipper method (JRip) in RWeka
jripper <- train(TrainData, TrainClass, method = "JRip")
prediction <- table(predict(jripper, TestData), TestClass)
cm2 <- confusionMatrix(prediction, positive = NULL, prevalence = NULL)
cm2
#the above code gives confusion matrix along with accuracy
summary(test_data)
plot(jripper)
#used to plot values




#K-Nearest Neighbour on Iris Dataset
#load iris data and install packages class and caret with dependencies = TRUE
data(iris)
library(class)
library(caret)
#seed set as first two and last two digits of UFID
set.seed(0375)
sample_size <- floor(0.80*nrow(iris))
#sample size taken as 80% of total dataset
TrainInData <- sample(seq_len(nrow(iris)), size = sample_size)
train_data <- iris[TrainInData,]
test_data <- iris[-TrainInData,]
#Test data set as 20% of total dataset
TrainData <- train_data[,1:4]
TrainClass <- train_data[,5]
TestData <- test_data[,1:4]
TestClass <- test_data[,5]
set.seed(0375)
#seed set as first two and last two digits of UFID
knearestn <- knn(train = TrainData, test = TestData, cl = TrainClass, k = 5)
#using knn method in class package with k=5 (square root of number of records)
prediction <- table(knearestn, TestClass)
cm3 <- confusionMatrix(prediction, positive = NULL, prevalence = NULL)
cm3
#gives confusion matrix and accuracy
summary(test_data)
plot.default(knearestn)
#used to plot the values




#C4.5 on Iris Dataset
#load iris data and install RWeka, caret and partykit packages with dependencies = TRUE
library(RWeka)
library(caret)
library(partykit)
data(iris)
#set seed = first two and last two digits of UFID
set.seed(0375)
sample_size <- floor(0.80*nrow(iris))
#sample size = 80% of data set
TrainInData <- sample(seq_len(nrow(iris)), size = sample_size)
train_data <- iris[TrainInData,]
test_data <- iris[-TrainInData,]
TestData <- test_data[,1:4]
TestClass <- test_data[,5]
#uses the J48 method in RWeka package
cfit <- J48(Species~., data = train_data)
prediction <- table(predict(cfit, TestData), TestClass)
cm4 <- confusionMatrix(prediction, positive = NULL, prevalence = NULL)
cm4
#gives confusion matrix and accuracy
summary(test_data)
plot(cfit)
#used to plot the values using partykit package




#Oblique Tree on Iris Dataset
#load the iris data and install oblique.tree and caret packages with dependencies = TRUE
library(oblique.tree)
library(caret)
data(iris)
#set seed = first two and last two digits of UFID
set.seed(0375)
#sample size = 80% of the data
sample_size <- floor(0.80*nrow(iris))
TrainInData <- sample(seq_len(nrow(iris)), size = sample_size)
train_data <- iris[TrainInData,]
test_data <- iris[-TrainInData,]
#divide the attributes and class to be predicted
TestData <- test_data[,1:4]
TestClass <- test_data[,5]
#oblique.tree creation
obtree <- oblique.tree(formula = Species~., data = train_data, oblique.splits = "on")
#plot oblique tree with heading as "Oblique Tree"
plot(obtree);text(obtree);title(main = "Oblique Tree")
prediction <- table(predict(obtree, newdata = test_data, type = c("class"), update.oblique.tree = FALSE), TestClass)
cm5 <- confusionMatrix(prediction, positive = NULL, prevalence = NULL)
cm5
#gives the confusion matrix along with accuracy
summary(test_data)
#the summary of test data can be used to compare with the output of confusion matrix




#Life Expectancy Dataset
#Naive Bayes on Life Expectancy Dataset
#load the life_expectancy.csv from the working directory
library(e1071)
library(caret)
#install e1071 and caret packages with dependencies = TRUE
ledata = read.csv("life_expectancy.csv")
set.seed(0375)
#set seed = first two and last two digits of UFID
sample_size <- floor(0.80*nrow(ledata))
#sample size = 80% of dataset
TrainInData <- sample(seq_len(nrow(ledata)), size = sample_size)
train <- ledata[TrainInData, ]
test <- ledata[-TrainInData, ]
#take training data attributes only from 3:7 ignoring first 2 as they don't help in classification
TrainData <- train[,3:7]
TrainClass <- train[,8]
TestData <- test[,3:7]
TestClass <- test[,8]
#use the naivebayes method in e1071 package
bayes <- naiveBayes(TrainData, TrainClass)
prediction = table(predict(bayes, TestData), TestClass)
summary(test)
cm6 <- confusionMatrix(prediction, positive = NULL, prevalence = NULL)
cm6
#gives confusion matrix and accuracy of the result




#JRipper on Life Expectancy Dataset
#load life_expectancy.csv from the working directory
library(RWeka)
library(caret)
#install RWeka and caret packages with dependencies = TRUE
ledata = read.csv("life_expectancy.csv")
set.seed(0375)
#set seed = 80% of dataset
smp_size <- floor(0.80*nrow(ledata))
TrainInData <- sample(seq_len(nrow(ledata)), size = smp_size)
train <- ledata[TrainInData, ]
test <- ledata[-TrainInData, ]
#consider only 3:7 attributes as 1 and 2 don't help in classification
TrainData <- train[,3:7]
TrainClass <- train[,8]
TestData <- test[,3:7]
TestClass <- test[,8]
#use JRip method present in RWeka
jripperle <- train(TrainData, TrainClass, method = "JRip")
prediction = table(predict(jripperle, TestData), TestClass)
summary(test)
cm7 <- confusionMatrix(prediction, positive = NULL, prevalence = NULL)
cm7
#gives confusion matrix and accuracy
plot(jripperle)
#used to plot the values of result




#K-Nearest Neighbour on Life Expectancy Dataset
#load life_expectancy.csv from the working directory
library(class)
library(caret)
#install class and caret packages with dependencies = TRUE
ledata = read.csv("life_expectancy.csv")
#set seed = 80% of data
set.seed(0375)
sample_size <- floor(0.80*nrow(ledata))
TrainInData <- sample(seq_len(nrow(ledata)), size = sample_size)
train <- ledata[TrainInData, ]
test <- ledata[-TrainInData, ]
TrainData <- train[,3:7]
TrainClass <- train[,8]
TestData <- test[,3:7]
TestClass <- test[,8]
set.seed(0375)
#use knn method from class package and take k = 5 (square root of number of test records)
knearestn <- knn(train = TrainData, test = TestData, cl = TrainClass, k = 5)
prediction <- table(knearestn, TestClass)
cm7 <- confusionMatrix(prediction, positive = NULL, prevalence = NULL)
cm7
#gives confusion matrix and accuracy
summary(test)
plot.default(knearestn)
#used to plot the values




#C4.5 on Life Expectancy Dataset
#load life_expectancy.csv from the working directory
library(RWeka)
library(caret)
library(partykit)
#install the RWeka, caret and partykit packages with dependencies = TRUE
ledata = read.csv("life_expectancy.csv")
set.seed(0375)
#set seed = first two and last two digits of UFID
sample_size <- floor(0.80*nrow(ledata))
#set sample size = 80% of data
TrainInData <- sample(seq_len(nrow(ledata)), size = sample_size)
train <- ledata[TrainInData, ]
test <- ledata[-TrainInData, ]
TrainData <- train[,3:8]
TestData <- test[,3:7]
TestClass <- test[,8]
#use J48 method in RWeka package
cfourfive <- J48(Continent ~., data = TrainData)
prediction = table(predict(cfourfive, TestData), TestClass)
cm5 <- confusionMatrix(prediction, positive = NULL, prevalence = NULL)
cm5
#gives confusion matrix and accuracy
summary(test)
plot(cfourfive)
#gives plot of values




#Oblique Tree on Life Expectancy Dataset
#load life_expectancy.csv from the working directory
library(oblique.tree)
library(caret)
#install oblique.tree and caret packages with dependencies = TRUE
ledata = read.csv("life_expectancy.csv")
set.seed(0375)
#set seed = first two and last two digits of UFID
sample_size <- floor(0.80*nrow(ledata))
#sample size = 80% of dataset
TrainInData <- sample(seq_len(nrow(ledata)), size = sample_size)
train <- ledata[TrainInData, ]
test <- ledata[-TrainInData, ]
TrainData <- train[,3:8]
TestData <- test[,3:7]
TestClass <- test[,8]
#use oblique.tree method to create the oblique tree for classification
ob.tree <- oblique.tree(formula = Continent~., data = TrainData, oblique.splits = "only")
plot(ob.tree);text(ob.tree);title(main = "Oblique Tree")
#plot the values
prediction <- table(predict(ob.tree, newdata = test, type = c("class"), update.oblique.tree = FALSE), TestClass)
cm10 <- confusionMatrix(prediction, positive = NULL, prevalence = NULL)
cm10
#gives confusion matrix and accuracy
summary(test)
#used to compare the prediction with summary of test data