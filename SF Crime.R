#San Francisco Crime Classification project for Kaggle
#Author - Catie Petersen
#Credit to Tianqi Chen for his excellent blog post on using xgboost which got me started

library(lubridate)
library(xgboost)
require(methods)
library(data.table)
library(magrittr)
library(chron)
library(ggplot2)
library(Ckmeans.1d.dp)
library(dplyr)
library(reshape2)
library(readr)
set.seed(123)

#Import training data

train <- read.csv("C:/Users/IBM_ADMIN/Downloads/train.csv")

#Import testing data

test <- read.csv("C:/Users/IBM_ADMIN/Downloads/test.csv")

#Awesome visualization by PeterCooman of count of crimes by Category and PdDistrict (not required but fun)

crimes_by_district_plot <- table(train$Category,train$PdDistrict)
crimes_by_district_plot <- melt(crimes_by_district_plot)
names(crimes_by_district_plot) <- c("Category","PdDistrict","Count")

g <- ggplot(crimes_by_district_plot,aes(x=Category, y=Count,fill = Category)) + 
  geom_bar(stat = "Identity") + 
  coord_flip() +
  facet_grid(.~PdDistrict) +
  theme(legend.position = "none")
ggsave(g, file="Crimes_by_district.png", width=20, height=8)


#Feature creation for building the model
#Parse time of day and month out of the Dates time stamp
train$Time <- hours(train$Dates)
train$Month <- month(train$Dates)         

test$Time <- hours(test$Dates)
test$Month <- month(test$Dates)

#XG Boost Model Creation
#Everything to integers because xgboost only takes integers

playground <- train
playground$Category <- as.integer(playground$Category)
playground$DayOfWeek <- as.integer(playground$DayOfWeek)
playground$PdDistrict <- as.integer(playground$PdDistrict)
playground$Time <- hours(playground$Dates)
playground$Time <- as.integer(playground$Time)
playground$Month <- as.integer(playground$Month)

playgroundtest <- test
playgroundtest$DayOfWeek <- as.integer(playgroundtest$DayOfWeek)
playgroundtest$PdDistrict <- as.integer(playgroundtest$PdDistrict)
playgroundtest$Time <- hours(playgroundtest$Dates)
playgroundtest$Time <- as.integer(playgroundtest$Time)
playgroundtest$Month <- as.integer(playgroundtest$Month)

#Create the label (the thing the model is trying to predict, the dependent variable)
y <- playground$Category
#xgboost is expecting the label to start at 0, so need to reduce the Category integers by 1
z <- matrix(data = 1, nrow = 878049, ncol = 1)
y <- y - z

#Delete unneeded columns, including the dependent variable because otherwise xgboost will use that in the model
playground$Category <- NULL
playground$Dates <- NULL
playground$Descript <- NULL
playground$Resolution <- NULL
playground$Address <- NULL

playgroundtest$Category <- NULL
playgroundtest$Dates <- NULL
playgroundtest$Descript <- NULL
playgroundtest$Resolution <- NULL
playgroundtest$Address <- NULL
playgroundtest$Id <- NULL

#Create matricies for test and train data, since xgboost only takes numeric matricies
trainMatrix <- data.matrix(playground)
testMatrix <- data.matrix(playgroundtest)

#Create model
#Set the parameters, link for parameters explanations:
#https://github.com/dmlc/xgboost/blob/master/doc/parameter.md#parameters-for-tree-booster
#eta of 1.0 and max delta step of 1 makes the model more conservative, nround of 50 is kind of over the top for number of trees, probably risking overfit
numberOfClasses <- max(y) + 1
param <- list("objective" = "multi:softprob","eval_metric" = "mlogloss","num_class" = numberOfClasses, eta = 1.0, max_delta_step = 1)
nround <- 50

#train the model
bst <- xgboost(param=param, data = trainMatrix, label = y, nrounds=nround)

#Feature importance (not necessary for predictive model, but fun to do, courtesy of Tianqi Chen)
# Get the feature real names
names <- dimnames(trainMatrix)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
xgb.plot.importance(importance_matrix[1:6,])

#Create prediction
#Apply model to test data
resultsxg <- predict(bst,testMatrix)

#Create submission
#Column names to make them a perfect match
colnameslist <- c("Id","ARSON", "ASSAULT", "BAD CHECKS", "BRIBERY", "BURGLARY","DISORDERLY CONDUCT", "DRIVING UNDER THE INFLUENCE","DRUG/NARCOTIC","DRUNKENNESS","EMBEZZLEMENT","EXTORTION","FAMILY OFFENSES","FORGERY/COUNTERFEITING","FRAUD", "GAMBLING","KIDNAPPING","LARCENY/THEFT","LIQUOR LAWS","LOITERING","MISSING PERSON","NON-CRIMINAL","OTHER OFFENSES","PORNOGRAPHY/OBSCENE MAT","PROSTITUTION","RECOVERED VEHICLE","ROBBERY","RUNAWAY","SECONDARY CODES","SEX OFFENSES FORCIBLE","SEX OFFENSES NON FORCIBLE","STOLEN PROPERTY","SUICIDE","SUSPICIOUS OCC","TREA","TRESPASS","VANDALISM","VEHICLE THEFT","WARRANTS","WEAPON LAWS")

#Prepare the boosted tree results
#get the results of the boosted tree model into a matrix shape (One row per category, one column per observation, the matrix is filled by columns)
resultsxg <- matrix(resultsxg,39,length(resultsxg)/39)
#transpose the matrix (flip it so it's going the right way)
resultsxg <- t(resultsxg)
#shrink the size of the file down by getting rid of some digits and scientific notation
resultsxg <- format(resultsxg, digits=2,scientific=F)
#turn into a data frame
resultsxg <- as.data.frame(resultsxg)
#Add the IDs back in
resultsxg <- cbind(test$Id, resultsxg)
#Update with column names
colnames(resultsxg) <- colnameslist


#Export submission file
write.table(resultsxg, file = "C:/xgsubmission.csv", row.names = FALSE, quote = FALSE, sep = ",")
