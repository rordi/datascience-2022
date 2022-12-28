# ---------------------------------------------------------------------
# Assignment 2 - CLASSIFICATION - PREPROCESS
# ---------------------------------------------------------------------
#
# Group A2
# Dietrich Rordorf, Marco Lecci, Sarah Castratori
#
# This script is used to preprocess the raw data from CSV to useable
# state.
#
# =====================================================================
#HINTS FROM ASSIGNMENT PAPER / HOLGER
    #• Data preprocessing is easier here; no feature engineering is needed.
    #• You may be able to reuse parts of the exercises we used in our examples during lectures.
    #• All in- and output values need to be floating numbers (or integers in exceptions) in the range of [0,1].
    #• Please note that a neural network expects a R matrix or vector, not data frames. Transform your
    #data (e.g. a data frame) into a matrix with data.matrix if needed.
    #• There are some models which show an accuracy higher than 90% (!) for training (and test) data –
    #after learning more than 1000 epochs.

#MAIN TASK FROM ASSIGNMENT PAPER/ HOLGER
  #• Design your network. Why did you use a feed-forward network, or a convolutional or recursive network – and why not?
  #• Use k-fold validation (with k = 10) to find the best hyperparameters for your network.
  #• Use the average of the accuracy to evaluate the performance of your trained network.
  #• Find a “reasonable” good model. Argue why that model is reasonable. If you are not able to find a reasonable good model, explain what you all did to find a good model and argue why you think that’s not a good model.
  #• Save your trained neural network with save_model_hdf5. Also save your data sets you used for training, testing and validation.


#ATTRIBUTE OVERVIEW 
  #Attribute Name Explanation Remarks

  #ID                   Client number
  #CODE_GENDER          Gender
  #FLAG_OWN_CAR         Is there a car
  #FLAG_OWN_REALTY      Is there a property 
  #CNT_CHILDREN         Number of children
  #AMT_INCOME_TOTAL     Annual income 
  #NAME_INCOME_TYPE     Income category 
  #NAME_EDUCATION_TYPE  Education level 
  #NAME_FAMILY_STATUS   Marital status
  #NAME_HOUSING_TYPE    Way of living 
  #DAYS_BIRTH           Birthday, Count backwards from current day (0), -1 means yesterday
  #DAYS_EMPLOYED        Start date of employment, Count backwards from current day (0), . If positive, it means the person unemployed.
  #FLAG_MOBIL           Is there a mobile phone 
  #FLAG_WORK_PHONE      Is there a work phone 
  #FLAG_PHONE           Is there a phone
  #FLAG_EMAIL           Is there an email 
  #OCCUPATION_TYPE      Occupation
  #CNT_FAM_MEMBERS      Family size

#OBSERVATIONS 
  #OCCUPATION_TYPE contains 20699 NA samples. All 365243 DAYS_EMPLOYED samples correspond to the NA 
  #DAYS_EMPLOYMENT -1000 years unemployement (value 365243) records, must be handled in pre-processing
  #FLAG_MOBIL useless, every record is flagged yes
  #FLAG_WORK_PHONE, FLAG_EMAIL, FLAG_PHONE might not be relevant, but kept

# ATTRIBUTES / FEATURES TO CONSIDER DROPPING
#proposed attributes to drop: ID, Flag Mobil,  CNT_Children

#TARGET STATUS
#TARGET = status 
#The last attribute status contains the “pay-back behavior”, i.e. when did that customer 
#pay back their depts: Please note: We are learning only the pay-back behavior. 
#The decision, i.e. if we accept a customer or not, is done in another process step – not here!

# 0: 1-29 days past due
# 1: 30-59 days past due
# 2: 60-89 days overdue
# 3: 90-119 days overdue
# 4: 120-149 days overdue
# 5: Overdue or bad debts, write-offs for more than 150 days
# C: paid off that month
# X: No loan for the month

#Observation: 77% of the samples are having status 0

#convert non-numerical attributes to numeric. Hint from Holger: All in- and output values need to be floating numbers 
#(or integers in exceptions) in the range of [0,1] 
# one hot encoding/binary for : "CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY",
#"NAME_INCOME_TYPE",#"NAME_EDUCATION_TYPE",#"NAME_FAMILY_STATUS","NAME_HOUSING_TYPE",
#"FLAG_WORK_PHONE","FLAG_PHONE","FLAG_EMAIL","OCCUPATION_TYPE"
# how to treat DAYS_BIRTH? see Marco's solution
# how to treat DAYS_EMPLOYED - see Marco's solution
# Occupation has 1/3 NA

#FEATURE NORMALIZATION
#do feature-wise normalization: for each feature in the input data (a column in the input data matrix), 
#you subtract the mean of the feature and divide by the standard deviation, so that the feature is centered 
#around 0 and has a unit standard deviation. using the `scale()` function

#Note that the quantities that we use for normalizing the test data have been computed using the training data. 
#We should never use in our workflow any quantity computed on the test data, even for something as simple as data normalization.
#from training example as well. Needs to be adapted! 

#VALIDATION APPROACH
#use K-fold cross-validation as per instructions from Holger in class 1st of December and take the full data set 
#published for the training with kfold. K-Fold consists of splitting the available data into K partitions (typically K=4 or 5), 
#then instantiating K identical models, and training each one on K-1 partitions while evaluating on the remaining partition. 
#The validation score for the model used would then be the average of the K validation scores obtained.
#from trianing example. obviously must be updated

install.packages("tensorflow")
library(tensorflow)
install_tensorflow()
library(tensorflow)
tf$constant("Hellow Tensorflow")
# allow for reproducible results

install.packages("keras")
install.packages("caret")

install.packages("readr")
install.packages("dplyr")
install.packages("Sequential")
install.packages("fastDummies")
install.packages("corrplot")
install.packages("data.table")
install.packages("tidyr")
install.packages("reticulate")
library(keras)
library(ggplot2)
library(lattice)
library(caret)

library(readr)
library(Sequential)
library(dplyr)
library(fastDummies)
library(corrplot)
library(data.table)
library(tidyr)
library(reticulate)

setwd("~/Documents/FHNW/Data Science/Assignment/datascience-2022/A2_classification")
rm(list=ls())
df<- fread("Dataset-part-2.csv", sep = ",", header = TRUE)
df1<-df #don't really understand this step. but seems to have no influence on the dummies. 
df$OCCUPATION_TYPE<-replace_na(df$OCCUPATION_TYPE, "NA")

#head(df)
#glimpse(df)
View(df1)
View(df)
#add in code the names of the columns 
#create dummy variables and one-hot encode them
df <- dummy_cols(df, select_columns = c("CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY",
                                        "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
                                        "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE",
                                        "FLAG_WORK_PHONE","FLAG_PHONE","FLAG_EMAIL","OCCUPATION_TYPE","status"), 
                 remove_selected_columns = TRUE)

View(df)

#retired / recoded. create target dummies with column status, one-hot encode them into eight columns
#status <-df$status
#df_target <- data.frame(status)
#View(df_target)
#df_target_dummies<-df_target
#numcol<-ncol(df_target_dummies)
#df_target_dummies<-dummy_cols(df_target_dummies, select_columns = c("status"))
#View(df_target_dummies)

#remove status from df data frame, since it is stored now as dummies in d_target_dummies
#df_subset <- subset(df, select = -status)
#View(df_subset)
#head(df_subset)
#glimpse(df_subset)
#change name to df1
#df <- df_subset
#View(df)

#copy / paste from Marco: 
#Function range that is applicable to columns with either all positive or negative values

range01 <- function(x){(x-min(x))/(max(x)-min(x))}

# sclae/normalize Income falling in 0-1 range

df$AMT_INCOME_TOTAL<-range01(df$AMT_INCOME_TOTAL)

#More intuitive to consider days of birth as a positive value

df$DAYS_BIRTH<--(df$DAYS_BIRTH)
df$DAYS_BIRTH<-range01(df$DAYS_BIRTH)

#It is probable that when the data was missing they used a positive number, therefore we use this parameter as a penalization and assign to it the value 0 in order to then be able to standardize it with the in range function
df$DAYS_EMPLOYED<-ifelse(df$DAYS_EMPLOYED>0,0,df$DAYS_EMPLOYED)
df$DAYS_EMPLOYED<--df$DAYS_EMPLOYED
df$DAYS_EMPLOYED<-range01(df$DAYS_EMPLOYED)

#scaling

df$CNT_CHILDREN<- range01(df$CNT_CHILDREN)

#Scaling

df$CNT_FAM_MEMBERS<- range01(df$CNT_FAM_MEMBERS)

#Unique value, FLAG_Mobile has always value "1", hence it is considered not relevant
df<-df[,-"FLAG_MOBIL"]

#drop CNT_CHILDREN cause is redundant with CNT_FAM_MEMBERS, correlation: 0.87

df<-df[,-"CNT_CHILDREN"]
View(df)
colnames(df) #"status" 


# transform your data frame into a matrix or array
data_matrix <- as.matrix(df)
dim(data_matrix)
colnames(data_matrix)

x <- data_matrix[, c(2:57)] # features
y <- data_matrix[, c(58:65)] # target values

# architecture
model <- keras_model_sequential()

model %>%
  layer_dense(units = 8, activation = "relu", input_shape = ncol(x)) %>%
  layer_dense(units = 8, activation = "softmax")

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

# train with k-fold
k <- 10

set.seed(123)

folds <- createFolds(y, k = k)
i <- 1
for (i in 1:k) {
val_idx <- folds[[i]]
train_idx <- unlist(folds[-i])
  
  # Use train_idx to select the rows of the data matrix for training
x_fold_train <- x
y_fold_train <- y
  
  # Use val_idx to select the rows of the data matrix for validation
x_fold_val <- x
y_fold_val <- y
  
# Train the model on the current fold
model %>% fit(
  x = x_fold_train, y = y_fold_train,
  epochs = 10,
  batch_size = 32,
  validation_data = list(x_fold_val, y_fold_val),
  verbose = 0
)

# Evaluate the model on the current fold
fold_history <- model %>% evaluate(
  x_fold_val, y_fold_val,
  verbose = 0
)

# Save the training history for the current fold
history[[i]] <- fold_history
}
    
# You can access the results of the cross-validation using the `results` object
print(results)







#....

# Some memory clean-up
k_clear_session()


# Define the model architecture
model <- keras_model_sequential() %>%
  layer_dense(units = 58, input_shape = c(N), activation = "relu") %>%
  layer_dense(units = 8, activation = "softmax")

# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

# Define the k-fold cross-validation scheme
k <- 10
kfold <- createFold(y, k = k, list = TRUE, returnTrain = TRUE)

# Initialize a list to store the evaluation scores
scores <- list()

# Loop through each fold
for (i in 1:k) {
  # Split the data into train and test sets
  train_idx <- kfold[[i]]
  test_idx <- setdiff(1:length(y), train_idx)
  x_train <- x[train_idx, ]
  y_train <- y[train_idx]
  x_test <- x[test_idx, ]
  y_test <- y[test_idx]
  
# Train the model on the train set
model %>% fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 32
  )
  
# Evaluate the model on the test set
score <- model %>% evaluate(x_test, y_test, verbose = 0)
  
# Store the evaluation score
scores[[i]] <- score
}

# Compute the mean and standard deviation of the evaluation scores
mean_score <- mean(sapply(scores, "[[", "accuracy"))
sd_score <- sd(sapply(scores, "[[", "accuracy"))


#from holger: compute the average of the per-epoch MAE scores for all folds
average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

#plot this
library(ggplot2)
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()

#use `geom_smooth()` to try to get a clearer picture
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth()

#According to this plot, it seems that validation MAE stops improving significantly after XY? epochs. Past that point, 
#we start overfitting.

#Once we are done tuning other parameters of our model, besides the number of epochs, one could also adjust the size of the 
#hidden layers, we can train a final production model on all of the training data, with the best parameters, then look at 
#its performance on the test data:
# Get a fresh, compiled model.
model <- build_model()

# Train it on the entirety of the data.
model %>% fit(train_data, train_targets,
              epochs = 80, batch_size = 16, verbose = 0)

result <- model %>% evaluate(test_data, test_targets)

#compute result
result
