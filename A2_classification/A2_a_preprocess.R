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
#  • Data preprocessing is easier here; no feature engineering is needed.
#  • You may be able to reuse parts of the exercises we used in our examples during lectures.
#  • All in- and output values need to be floating numbers (or integers in exceptions) in the range of [0,1].
#  • Please note that a neural network expects a R matrix or vector, not data frames. Transform your
#    data (e.g. a data frame) into a matrix with data.matrix if needed.
#  • There are some models which show an accuracy higher than 90% (!) for training (and test) data –
#    after learning more than 1000 epochs.

# MAIN TASK FROM ASSIGNMENT PAPER/ HOLGER
#  • Design your network. Why did you use a feed-forward network, or a convolutional or recursive network – and why not?
#  • Use k-fold validation (with k = 10) to find the best hyperparameters for your network.
#  • Use the average of the accuracy to evaluate the performance of your trained network.
#  • Find a “reasonable” good model. Argue why that model is reasonable. If you are not able to find a reasonable good model, 
#    explain what you all did to find a good model and argue why you think that’s not a good model.
#  • Save your trained neural network with save_model_hdf5. Also save your data sets you used for training, testing and validation.


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

# Disable GPU                                                                   
Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)

# set memory limit: this works only on Windows (you can copy-paste the below commented-out command into console and run it there)
# memory.limit(700000)

# For Apple M1 & M2 users with latest Apple GPUs
install_arm64<-function () {
  install.packages("tensorflow")
  library(reticulate)
  
  # replace the file path to your Anaconda python installation accordingly
  virtualenv_create("r-reticulate", python = "/Users/my_user/opt/anaconda3/bin/python")
  install_tensorflow(envname = "r-reticulate")
  install.packages("keras")
  library(keras)
  install_keras(envname = "r-reticulate")
 
  print("NOW PLEASE FOLLOW STEPS IN https://gist.github.com/rordi/9970c8840614644e01a53d68e51f37fd to pip uninstall tensorflow-metal !!")
}

# For all other users
install_amd64<-function () {
  install.packages("tensorflow")
  library(reticulate)
  library(tensorflow)
  install_tensorflow()
  install.packages("keras")
  library(keras)
  install_keras()
}

# manually run install_arm64() or install_amd64() in the R console depending on your system architecture

# install other package dependencies
install_packages_classification<-function() {
  install.packages("Sequential")
  install.packages("caret")
  install.packages("readr")
  install.packages("dplyr")
  install.packages("fastDummies")
  install.packages("corrplot")
  install.packages("data.table")
  install.packages("tidyr")
}
#install_packages_classification() # uncomment when running first time!

# confirm installation 
library(tensorflow)
library(keras)                                                                  
use_backend("tensorflow")
tf$constant("Hello Tensorflow!")

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


# =====================================================================
# LOAD DATA INTO DATAFRAME
# =====================================================================

# read the data from the CSV
df<-fread("./A2_classification/Dataset-part-2.csv", sep = ",", header = TRUE)

# no we randomize the order in the dataframe before doing anything else (we do not know how random the order is in the CSV)
df[sample(1:nrow(df)), ]



# =====================================================================
# INITIAL OBSERVATIONS
# =====================================================================

# print basic description of data frame
dim(df)    # we have 19 variables and ~67.6K observations (really few observations for training a NN! -> we should use cross-validation with 10% test splits)
str(df)
glimpse(df)
#View(df)

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
#* e.g. when applying this function, the values in the column will be centered and scaled to 0-1 range
#* 
min_max_normalize <- function(x, scale = TRUE) {
  # first scale using 2 standard deviations around mean
  if (scale == TRUE) {
    m = mean(x)
    s = 2 * sd(x)
    x<-scale(x, center = m, scale = s)
  }

  # then we apply min-max normalization so that values are between 0 and 1
  return ((x-min(x))/(max(x)-min(x)))
}


# =====================================================================
# DATA PREPARATION
# =====================================================================


# remove the ID column
df<-df[,-"ID"]

# look at the unique values in what seems to be categorical input variables
unique(df$CODE_GENDER) # M and F -> encode as 1 col 1/0
unique(df$FLAG_OWN_CAR) # Y and N -> encode as 1 col 1/0
unique(df$FLAG_OWN_REALTY) # Y and N -> encode as 1 col 1/0
unique(df$FLAG_WORK_PHONE) # 0 and 1, no further encoding needed
unique(df$FLAG_PHONE) # 0 and 1, no further encoding needed
unique(df$FLAG_EMAIL) # 0 and 1, no further encoding needed
unique(df$FLAG_MOBIL) # all values are null -> drop this column
unique(df$NAME_INCOME_TYPE) # "Pensioner", "Commercial associate", "Working", "State servant", "Student" --> TODO possibly "Commercial associate", "Working", "State servant" can be combined
unique(df$NAME_EDUCATION_TYPE) # "Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree" --> TODO possibly "Higher education" and "Academic degree" can be combined
unique(df$NAME_FAMILY_STATUS) # "Married", "Separated", "Widow, "Civil marriage", "Single / not married" --> TODO combine "Married" and "Civil marriage"
unique(df$NAME_HOUSING_TYPE) # "House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment", "Co-op apartment" --> TODO combine all apartments
unique(df$OCCUPATION_TYPE) # 18 categories plus missing (NA) values --> encode missing values as string "None" so that those will also be one-ht encoded

# Code gender: encode M --> 1 and F --> 0 (no gender bias intended!) as new col with suffix _M and remove original column
df$CODE_GENDER_M<-ifelse(df$CODE_GENDER=="M", 1, 0)
df<-df[,-"CODE_GENDER"]

# Flag Own Car: encode Y --> 1 and N --> 0 as new col with suffix _Y and remove original column
df$FLAG_OWN_CAR_Y<-ifelse(df$FLAG_OWN_CAR=="Y", 1, 0)
df<-df[,-"FLAG_OWN_CAR"]

# Flag Own Realty: encode Y --> 1 and N --> 0 as new col with suffix _Y and remove original column
df$FLAG_OWN_REALTY_Y<-ifelse(df$FLAG_OWN_REALTY=="Y", 1, 0)
df<-df[,-"FLAG_OWN_REALTY"]

# Flag mobiles: drop column --> all values are equal 1
df<-df[,-"FLAG_MOBIL"]

# Occupation Type: fill empty values with a string "None" so that it will be one-hot encoded as well subsequently
df$OCCUPATION_TYPE<-replace_na(df$OCCUPATION_TYPE, "None")

# Name Education Type: try to generalize the model better by reducing sparsity of some of the categorical values
df$NAME_EDUCATION_TYPE<-replace(df$NAME_EDUCATION_TYPE, df$NAME_EDUCATION_TYPE == "Academic degree", "Higher education") # only 38 values, we combine Academic degree with Higher education
df$NAME_EDUCATION_TYPE<-replace(df$NAME_EDUCATION_TYPE, df$NAME_EDUCATION_TYPE == "Lower secondary", "Secondary") # only 716, combine with other secondary type
df$NAME_EDUCATION_TYPE<-replace(df$NAME_EDUCATION_TYPE, df$NAME_EDUCATION_TYPE == "Secondary / secondary special", "Secondary")
table(df$NAME_EDUCATION_TYPE)

# Name Family Status: try to generalize the model better by reducing sparsity of some of the categorical values
df$NAME_FAMILY_STATUS<-replace(df$NAME_FAMILY_STATUS, df$NAME_FAMILY_STATUS == "Civil marriage", "Married") # combine "Civil marriage" into "Married", what's the difference?
table(df$NAME_FAMILY_STATUS)

# All other catgorical features are muti-class (>2 classes) and need special one-hot encoding into one col per unique class value
df <- dummy_cols(df, select_columns = c(
  "NAME_INCOME_TYPE",
  "NAME_EDUCATION_TYPE",
  "NAME_FAMILY_STATUS",
  "NAME_HOUSING_TYPE",
  "OCCUPATION_TYPE"
), remove_first_dummy = FALSE, remove_selected_columns = TRUE)

# Total income: huge outliers - we apply a treshold, then normalize the feature (scale to 0-1)
describe_feature(df$AMT_INCOME_TOTAL, "Total Income Amount") # huge outliers, we apply a threshold of 1.5 * IQR: Q3 + 1.5 * (Q3 - Q1) = 225K + 1.5 * (225K - 112.5K) = 393.75K
df$AMT_INCOME_TOTAL<-apply_threshold(df$AMT_INCOME_TOTAL, threshold = 350000)
df$AMT_INCOME_TOTAL<-min_max_normalize(df$AMT_INCOME_TOTAL)
describe_feature(df$AMT_INCOME_TOTAL, "Total Income Amount (normalized)")

# Days since birth: it is more intuitive to consider days since birth as a positive value
describe_feature(df$DAYS_BIRTH, "Days since birth")
df$DAYS_BIRTH<-abs(df$DAYS_BIRTH)
df$DAYS_BIRTH<-min_max_normalize(df$DAYS_BIRTH)
describe_feature(df$DAYS_BIRTH, "Days since birth (normalized)")

# Days employed: the only positive value is improbable (365243 days, would be 1000 years). We assume that positive value
# indicates missing data. We will thus apply a threshold of 0 before log-normalizing the feature (scale to 0-1).
describe_feature(df$DAYS_EMPLOYED, "Days employed")
df$DAYS_EMPLOYED<-apply_threshold(df$DAYS_EMPLOYED, threshold = 0)
df$DAYS_EMPLOYED<-abs(df$DAYS_EMPLOYED)
df$DAYS_EMPLOYED<-apply_threshold(df$DAYS_EMPLOYED, threshold = 10000000000) # log(10000000000) = 10
df$DAYS_EMPLOYED<-log(df$DAYS_EMPLOYED)
df$DAYS_EMPLOYED<-ifelse(is.finite(df$DAYS_EMPLOYED), df$DAYS_EMPLOYED, 0)
df$DAYS_EMPLOYED<-min_max_normalize(df$DAYS_EMPLOYED, scale = FALSE) # scale to 0-1
describe_feature(df$DAYS_EMPLOYED, "Days employed (log-normalized)")

# Children count: log-normalize the feature
describe_feature(df$CNT_CHILDREN, "Children count")
df$CNT_CHILDREN<-apply_threshold(df$CNT_CHILDREN, threshold = 7) # only 2 values have more
df$CNT_CHILDREN<-min_max_normalize(df$CNT_CHILDREN, scale = FALSE) # scale to 0-1
describe_feature(df$CNT_CHILDREN, "Children count (normalized)")

# Family members count: normalize the feature
describe_feature(df$CNT_FAM_MEMBERS, "Family members count")
df$CNT_FAM_MEMBERS<-apply_threshold(df$CNT_FAM_MEMBERS, threshold = 10) # only 2 values have more
df$CNT_FAM_MEMBERS<-min_max_normalize(df$CNT_FAM_MEMBERS, scale = FALSE) # scale to 0-1
describe_feature(df$CNT_FAM_MEMBERS, "Family members count (normalized)")


# check correlations between family and children count
#cor(df$CNT_CHILDREN, df$CNT_FAM_MEMBERS) # 0.8784203 -> strongly correlated, we keep only the family members count

# drop the CNT_CHILDREN because is heavily correlated / redundant with CNT_FAM_MEMBERS
#df<-df[,-"CNT_CHILDREN"]

# we are done with the data preparation for the input variables - show a summary of the prepared df
summary(df)
View(df)


# =====================================================================
# HANDLE CLASS ATTRIBUTE
# =====================================================================

df$status_factor<-df$status %>% as.factor()
df$status_numeric<-df$status_factor %>% as.numeric() # convert to numeric  (will encode classes from 1 to 8)
df$status_numeric<-df$status_numeric-1 # substract 1 from all status codes so that we have a range 0 to 7
table(df$status_numeric)

df<-df[,-"status"] # remove status label from the original df
df<-df[,-"status_factor"] # remove status_factor label from the original df
#View (df)


# =====================================================================
# HANDLE CLASS IMBALANCE AND CREATE TRAIN / TEST SET SPLITS
# =====================================================================

#**
#* A simple function to oversample a minority class by simply duplicating it's rows n times
#*  
copy_class_data <- function(df_train, n, class) {
  df_class<-filter(df_train, status_numeric == class)
  for (j in 1:n) {
    df_train<-rbind(df_train, df_class)
  }
  return (df_train)
}


#**
#* A simple function to balance classes by oversampling (copying) the minority classes recursively
#* using copy_class_data() function defined above
#*  
oversample_classes<-function (df_train) {
  num_majority<-sum(df_train$status_numeric == 0) # number of values of the majority class
  for (i in 1:7) {
    num_minority<-sum(df_train$status_numeric == i) # number of values of the minority class
    duplication_factor<-ceiling(5000 / num_minority) - 1  # oversample so that we have at least 5K for each class
    if (duplication_factor > 1) {
      df_train<-copy_class_data(df_train, n = duplication_factor, class=i)  
    }
  }
  
  return (df_train)
}

#**
#* A simple function to encode the status labels in columns from 0 to 7 instead of simple one-hot encoding
#* 
encode_status<-function (df) {
  encoded<-to_categorical(df$status_numeric, num_classes=8)
  #for (n in 1:8) {
  #  encoded[,n]<-n*(encoded[,n]) 
  #}
  return (encoded)
}


# TODO: replace with k-fold test/train split

test_split<-0.2
train_row<-(nrow(df)-round(nrow(df)*test_split, digits = 0))
test_row<-(train_row + 1)

# prepare train data, oversample minority classes, encode labels
df_train<-df[0:train_row,]
table(df_train$status_numeric)
df_train<-oversample_classes(df_train)
table(df_train$status_numeric) # class values should be oversampled now!

# encode the status labels as a matrix
data_train_label<-encode_status(df_train)

# remove status columns from df_train and convert to matrix
df_train<-df_train[,-"status_numeric"]
data_train<-as.matrix(df_train)

View(data_train)
View(data_train_label)


# prepare the test data (on test data we do NOT handle the class imbalance!)
df_test<-df[test_row:nrow(df),]

# one-hot encode the status labels as a matrix
data_test_label<-encode_status(df_test)

# remove status columns from df_test and convert to matrix
df_test<-df_test[,-"status_numeric"]
data_test<-as.matrix(df_test)

View(data_test)
View(data_test_label)


# =====================================================================
# NEURAL NETWORK DESIGN AND FIT
# =====================================================================

#**
#* function that builds our SGD optimizer
#* 
build_optimizer_sgd<-function () {
  # Tensorflow < 2.3 becase some params in Keras optimizer_sgd changed name)
  sgd<-optimizer_sgd(
    learning_rate = 1e-3, # use "lr" in older releases of tensorflow !
    #lr = 1e-3,
    momentum = 0.9,
    weight_decay = 1e-6, # use "decay" in older releases of tensorflow !
    #decay = 1e-6,
    nesterov = FALSE,
    clipnorm = NULL,
    clipvalue = NULL)
  return (sgd)
}


#**
#* function that builds our ADAM optimizer
#* 
build_optimizer_adam<-function () {
  adam<-optimizer_adam( 
    lr = 1e-3, # use "lr" in older releases of tensorflow !
    #lr = 1e-4,
    beta_1 = 0.9,
    beta_2 = 0.999,
    decay = 0, # use "decay" in older releases of tensorflow !
    #decay = 1e-6
  )
  return (adam)
}


#**
#* function that builds our model (so that we can call it several times)
#* 
build_model <- function() {
  shape_input<-c(ncol(data_train))
  shape_output<-8
  
  neurons<-600
  dropout<-0.06
  
  model<-keras_model_sequential() 
  model %>% 
    layer_conv_1d(filters=64, kernel_size=2, input_shape=shape_input, activation="relu") %>%
    layer_max_pooling_1d() %>%
    layer_flatten() %>%
    layer_dense(units = shape_output, activation = "softmax") # Output layer

  summary(model)
  
  return (model)
}


#**
#* function that builds a 1D CNN model (so that we can call it several times)
#* 
build_model_1d <- function() {
  shape_input<-c(ncol(data_train_1d), 1)
  shape_output<-8
  
  model = keras_model_sequential() %>%
    layer_conv_1d(filters=64, kernel_size=8, input_shape=shape_input, activation="relu") %>%
    layer_conv_1d(filters=64, kernel_size=2, activation="relu") %>%
    layer_max_pooling_1d() %>%
    layer_flatten() %>%
    layer_dense(units = 640, activation = "relu") %>%
    layer_dense(units = 640, activation = "relu") %>%
    layer_dense(units = 640, activation = "relu") %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = shape_output, activation = "softmax") # Output layer
  
  summary(model)
  
  return (model)
}

# force delete any leftovers from previous runs!
rm(model)
rm(optimizer)
rm(metrics)
gc()


# prepare data for 1d CNN
data_train_1d = array(data_train, dim = c(nrow(data_train), ncol(data_train), 1))
data_test_1d = array(data_test, dim = c(nrow(data_test), ncol(data_test), 1))

# Build the model and optimizer
#model<-build_model()
model<-build_model_1d()
optimizer<-build_optimizer_adam()


# Compile the model
model<-model %>%
  compile(
    optimizer=optimizer,
    loss="categorical_crossentropy", 
    metrics=c("accuracy")
  )

# Train the model
history<-model %>%
  fit(
    data_train_1d, data_train_label,
    epochs = 100,
    batch_size = 64,
    validation_data=list(data_test_1d, data_test_label)
  )

# Evaluate the trained model
plot(history)
#lastlayer = length(model$layers) - 1
#weight<-as.matrix(model$layers[[lastlayer]]$weights[[1]])
#bias<-as.matrix(model$layers[[lastlayer]]$weights[[2]])

# Evaluate the model on test data
metrics<-model %>% evaluate(data_test_1d, data_test_label)
metrics

# Observations from metrics:
#
# - activation function: relu on input and hidden layers gave better results than sigmoid (last layer always softmax as we have a multi-valued class prediction)
# - smaller batch sizes gave better results, we tried 128, 64, 32 and 16. With 16 the training was really becoming slow and difficult to run the model with
#   different params as part of hyperparams tuning. Also difference of accuracy in test data was not big for 16 versus 32, so we decided that 32 is a good number.
# - SGD optimizer:
#    - learning rate of 0.01 was too big, reduced to initial value of 1e-4
#    - weight decay: added some decay to avoid overfitting to training data (generalize the model better); we tried with 1e-5 and 1e-4, the smaller one was slightly better
# - Adam optimizer:
#    - learning rate of 1e-4 was too large, 5e-5 showed less fast convergence
#



