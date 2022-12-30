# ---------------------------------------------------------------------
# Assignment 2 - CLASSIFICATION - PREPROCESS
# ---------------------------------------------------------------------
#
# Group A2
# Dietrich Rordorf, Marco Lecci, Sarah Castratori
#
# This script is used to preprocess the raw data from CSV to usable
# state.
#
# =====================================================================
# HINTS FROM ASSIGNMENT PAPER / HOLGER
    # • Data preprocessing is easier here; no feature engineering is needed.
    # • You may be able to reuse parts of the exercises we used in our examples during lectures.
    # • All in- and output values need to be floating numbers (or integers in exceptions) in the range of [0,1].
    # • Please note that a neural network expects a R matrix or vector, not data frames. Transform your
    #   data (e.g. a data frame) into a matrix with data.matrix if needed.
    # • There are some models which show an accuracy higher than 90% (!) for training (and test) data –
    #   after learning more than 1000 epochs.

# MAIN TASK FROM ASSIGNMENT PAPER/ HOLGER
  # • Design your network. Why did you use a feed-forward network, or a convolutional or recursive network – and why not?
  # • Use k-fold validation (with k = 10) to find the best hyperparameters for your network.
  # • Use the average of the accuracy to evaluate the performance of your trained network.
  # • Find a “reasonable” good model. Argue why that model is reasonable. If you are not able to find a reasonable good model, explain what you all did to find a good model and argue why you think that’s not a good model.
  # • Save your trained neural network with save_model_hdf5. Also save your data sets you used for training, testing and validation.


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



# =====================================================================
# PRELIMINARIES
# =====================================================================

# clear envionrment
rm(list=ls())

# allow for reproducible results
set.seed(1)

# set memory limit: this works only on Windows (you can copy-paste the below commented-out command into console and run it there)
# memory.limit(700000)


# Apple M1 & M2 users --> please install according to https://gist.github.com/rordi/9970c8840614644e01a53d68e51f37fd
#
# Other users (Windows, older Intel-based Macs), please copy-paste the following commands to the RStudio console manually:
# 
# install.packages("tensorflow")
# library(reticulate)
# library(tensorflow)
# install_tensorflow()
# install.packages("keras")
# library(keras)
# install_keras()


# --> This r-reticulate Keras call installs tensorflow-metal again, which I have to remove manually again!! It is not compatible with Apple's GPU chips.
# --> open terminal (iterm)
# --> check the python / conda envs availalbe:
#       conda info --envs
# --> activate the conda env for r-reticulate, for me this was:
#       source /Users/didi/Library/r-miniconda-arm64/bin/activate
# --> then still in iterm uninstall the tensorflow-metal again (be sure to use the same coda env python distro):
#       /Users/didi/Library/r-miniconda-arm64/envs/r-reticulate/bin/python -m pip uninstall tensorflow-metal

# confirm installation 
library(tensorflow)
tf$constant("Hello Tensorflow!")


# install other package dependencies
install.packages("Sequential")
install.packages("caret")
install.packages("readr")
install.packages("dplyr")
install.packages("fastDummies")
install.packages("corrplot")
install.packages("data.table")
install.packages("tidyr")
install.packages("ROSE")


library(Sequential)
library(ggplot2)
library(lattice)
library(caret)
library(readr)
library(dplyr)
library(fastDummies)
library(corrplot)
library(data.table)
library(tidyr)
library(ROSE)


df<- fread("./A2_classification/Dataset-part-2.csv", sep = ",", header = TRUE)


# =====================================================================
# INITIAL OBSERVATIONS
# =====================================================================

# print basic description of data frame
dim(df)    # we have 19 variables and ~67.6K observations (really few observations for training a NN! -> we should use cross-validation with 10% test splits)
str(df)
glimpse(df)
View(df)

# Initial Observations:
# 
# --> OCCUPATION_TYPE: has "NA" coded as a string
# --> NAME_EDUCATION_TYPE: categorical, seems to have some sort of hierarchy encoded such as in "Secondary / secondary special"
# --> status: variable name has different spelling and the status are numbers encoded as strings; this is our targt feature
# --> several variables are categorical and need one-hot encoding:
#    - CODE_GENDER
#    - FLAG_OWN_CAR
#    - FLAG_OWN_REALTY
#    - NAME_INCOME_TYPE
#    - NAME_EDUCATION_TYPE
#    - NAME_FAMILY_STATUS
#    - NAME_HOUSING_TYPE
# --> several feature seem to be already one-hot encoded (to be checked if they really only contain 0 and 1s) but seem to indicate if the person has a type of contact yes/no (are they all really useful features?):
#    - FLAG_MOBIL
#    - FLAG_WORK_PHONE
#    - FLAG_PHONE
#    - FLAG_EMAIL
# --> DAYS_EMPLOYED: can be positive (employed) or negative (likely days since last employment, this could thus be unemployed or retired persons)


# =====================================================================
# HELPER FUNCTIONS TO DESCRIBE AND PREPROCESS FEATURES
# =====================================================================

#**
#* function that describes a (numeric) feature 
#* usage example: describe_feature(df$int_rate, "Interest Rate")
#* 
describe_feature <- function(feature, feature_name = "Feature") {
  # show number of NAs
  message(paste("Number of NAs: ", sum(is.na(feature))))
  
  # replace NA with 0
  feature_handled<-replace_na(feature,0)
  
  # show number of unique values
  message(paste("Number of unique values: ", length(unique(feature_handled))))
  
  # show number of zero values (relative)
  message(paste("Sparsity relative: ", length(which(feature_handled == 0))/length(feature_handled)))
  
  # outliers detection (exclude null values as we have some very sparse attributes)
  outliers<-boxplot.stats(feature_handled)$out
  outliers_count<-length(outliers)
  message(paste("Potential outliers (1.5 * IQR): ", outliers_count))
  if (outliers_count>0) {
    message(paste("Potential outlier range of values: ", toString(range(outliers))))
  }
  
  # a deep-dive into the distribution
  message(paste("98th, 99th and 99.9th percentiles: ", toString(quantile(feature_handled, c(.98, .99, .999)))))
  message("Summary of distribution (also check the box plot and histogram): ")
  boxplot(feature_handled, main=feature_name, xlab=feature_name)
  hist(feature_handled, main=feature_name, xlab=feature_name)
  summary(feature_handled) # keep it as last command in function
}

#**
#* function to replace NAs with 0 or another value that we can pass as argument (e.g. a mean or median)
#*
handle_na <- function(feature, replace_with = 0) {
  feature_handled<-replace_na(as.numeric(feature), replace_with)
  return(feature_handled)
}

#**
#* function to apply a (maximum) threshold value to outliers
#* e.g., we can cut-off the annual income distribution at threshold = 150'000 USD - any value above will be replaced with 150'000 USD
#* 
apply_threshold <- function(feature, threshold = 1) {
  feature_handled<-ifelse(feature>threshold, threshold, feature)
}

#**
#* Function for min max scaling (standardization) of a feature
#* e.g. when applying this function, the values in the collumn will be scaled to 0-1 range
#* 
min_max_scale <- function(x){(x-min(x))/(max(x)-min(x))}


# =====================================================================
# DATA PREPARATION
# =====================================================================

# Occupation Type: fill empty values with "NA"
df$OCCUPATION_TYPE<-replace_na(df$OCCUPATION_TYPE, "NA")

# one-hot encode categorical features with exactly 2 classes as 1 column
df <- dummy_cols(df, select_columns = c(
  "CODE_GENDER"
), remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# one-hot encode categorical features with more than 2 classes as 1 column per class
df <- dummy_cols(df, select_columns = c(
  "FLAG_OWN_CAR",
  "FLAG_OWN_REALTY",
  "NAME_INCOME_TYPE",
  "NAME_EDUCATION_TYPE",
  "NAME_FAMILY_STATUS",
  "NAME_HOUSING_TYPE",
  "FLAG_WORK_PHONE",
  "FLAG_PHONE",
  "FLAG_EMAIL",
  "OCCUPATION_TYPE",
  "status"
), remove_first_dummy = FALSE, remove_selected_columns = TRUE)

# take a loot at the modified dataframe
View(df)

# Total income: huge outliers - we apply a treshold, then normalize the feature (scale to 0-1)
describe_feature(df$AMT_INCOME_TOTAL, "Total Income Amount") # huge outliers, we apply a threshold of 1.5 * IQR: Q3 + 1.5 * (Q3 - Q1) = 225K + 1.5 * (225K - 112.5K) = 393.75K
df$AMT_INCOME_TOTAL<-apply_threshold(df$AMT_INCOME_TOTAL, threshold = 393750)
df$AMT_INCOME_TOTAL=min_max_scale(df$AMT_INCOME_TOTAL)

# Days since birth: it is more intuitive to consider days since birth as a positive value
describe_feature(df$DAYS_BIRTH, "Days since birth")
df$DAYS_BIRTH<--(df$DAYS_BIRTH)
df$DAYS_BIRTH<-min_max_scale(df$DAYS_BIRTH)

# Days employed: the only positive value is improbable (365243 days, would be 1000 years). We assume that positive value
# indicates missing data. We will thus apply a threshold of 0 before normalizing the feature (scale to 0-1).
describe_feature(df$DAYS_EMPLOYED, "Days employmed")
df$DAYS_EMPLOYED<-apply_threshold(df$DAYS_EMPLOYED, threshold = 0)
describe_feature(df$DAYS_EMPLOYED, "Days employmed") # looks more like a powerlaw distribution now
df$DAYS_EMPLOYED<-min_max_scale(df$DAYS_EMPLOYED)

# Children count: normalize the feature
describe_feature(df$CNT_CHILDREN, "Children count")
df$CNT_CHILDREN<- min_max_scale(df$CNT_CHILDREN)

# Family members count: normalize the feature
describe_feature(df$CNT_FAM_MEMBERS, "Family members count")
df$CNT_FAM_MEMBERS<- min_max_scale(df$CNT_FAM_MEMBERS)

# check correlations between family and children count
cor(df$CNT_CHILDREN, df$CNT_FAM_MEMBERS) # 0.8784203 -> strongly correlated, we keep only the family members count

# drop the CNT_CHILDREN because is heavily correlated / redundant with CNT_FAM_MEMBERS
df<-df[,-"CNT_CHILDREN"]

# Flag mobiles: all values are equal to 1, this feature is meaning less and we thus drop it
describe_feature(df$FLAG_MOBIL, "Flag Mobile")
df<-df[,-"FLAG_MOBIL"]


# Status (our class attribute for prediction!):
# We tried to one-hot encode the status before. Here we try additionally to
# encode them from 1 to 8 to see as it seemed to make the results better
df$status_0<-ifelse(df$status_0=="1",1,0)
df$status_1<-ifelse(df$status_1=="1",2,0)
df$status_2<-ifelse(df$status_2=="1",3,0)
df$status_3<-ifelse(df$status_3=="1",4,0)
df$status_4<-ifelse(df$status_4=="1",5,0)
df$status_5<-ifelse(df$status_5=="1",6,0)
df$status_C<-ifelse(df$status_C=="1",7,0)
df$status_X<-ifelse(df$status_X=="1",8,0)


# remove the ID column
df<-df[,-"ID"]


# we are done with the data preparation - show a summary of the prepared df
summary(df)
View(df)


# not sure what Marco's intention was here - asked him on WeChat, for now commenting out
#df_characters<-dummy_cols(df[,c(2,3,4,7,8,9,10,17)],)
#df_characters<-df_characters[,-c(1,2,3,4,5,6,7,8)]
#df_characters$ID<-df$ID
#df_num<-select_if(df, is.numeric)
#df <- merge(df_characters, df_num, by="ID", all.x = FALSE, all.y = FALSE)



# =====================================================================
# TRAIN / TEST SET SPLITS
# =====================================================================


# TODO: balance the dataset with oversampling of the underrepresented classes
# TODO: replace with k-fold test/train split

df_train<-df[1:50000,]
#data_balanced_over <- ovun.sample(status ~., data =df_train , method = "over",N =50000*3)

# dataset prepared with simple holdout
df_train<-as.matrix(df[1:50000,-c(54,55,56,57,58,59,60,61)])
df_test<-as.matrix(df[50001:67614,-c(54,55,56,57,58,59,60,61)])
df_label<-as.matrix(df[1:50000,c(54,55,56,57,58,59,60,61)])
df_label_test<-as.matrix(df[50001:67614,c(54,55,56,57,58,59,60,61)])



# =====================================================================
# NEURAL NETWORK DESIGN AND FIT
# =====================================================================

#**
#* function that builds our model (so that we can call it several times)
#* 
build_model <- function() {
  
  # Prepare the gradient descent optimizer (Marco may have an older version of
  # Tensorflow < 2.3 becase some params in Keras optimizer_sgd changed name)
  SGD <- optimizer_sgd(
    learning_rate = 1e-6, # use "lr" in older releases of tensorflow
    momentum = 0.9,
    weight_decay = 1e-6, # use "decay" in older releases of tensorflow
    nesterov = FALSE,
    clipnorm = NULL,
    clipvalue = NULL)
  
  # Build the Keras network model
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = 200, activation = "relu", input_shape = c(55)) %>% 
    layer_dense(units = 8, activation = "softmax")

  summary(model)
    
  model %>% compile(
    optimizer = SGD, # "rmsprop" or SDG
    loss = "categorical_crossentropy", 
    metrics = c("accuracy")
  )
}


# Train the model
model<-build_model()
model %>%
  fit(
    df_train, df_label,
    epochs = 1000,
    batch_size = 128
  )

# Evaluate the model
metric< model %>% evaluate(df_test,df_label_test)
metric






