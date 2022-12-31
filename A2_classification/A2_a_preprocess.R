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




# =====================================================================
# LOAD DATA INTO DATAFRAME
# =====================================================================

# read the data from the CSV
df<- fread("./A2_classification/Dataset-part-2.csv", sep = ",", header = TRUE)

# first we randomize the order in the dataframe before doing anything else (we do not know how random the order is in the CSV!)
df[sample(1:nrow(df)), ]



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
min_max_normalize <- function(x) {
  # then we apply min-max normalization so that values are between 0 and 1
  return ((x-min(x))/(max(x)-min(x)))
}

#**
#* Function to copy dataset over itself (n = 1 --> 2x data, n = 2 --> 4x data, n = 3 --> 8x data)
#* e.g. when using large epochs like 1000 we need enough data, with this function we can just duplicate the dataset n times
#*  
copy_data <- function(data_array, n) {
  for (i in 1:n) {
    data_array<-rbind(data_array, data_array)
  }
  return (data_array)
}


# =====================================================================
# DATA PREPARATION
# =====================================================================


# we move the status (our target class attribute) into a separate dataframe. It is important not to randomize the
# order of the data in the dataframe after this step as otherwise the position of a row in df no longer corresponds
# with the order of the corresponding label in the df_label
status<-df$status
df_label<-data.frame(status, stringsAsFactors = TRUE) # create new dataframe just with the status col
df_label$status<-df_label$status %>% as.numeric() # convert to numeric  (will encode X and C into 7 and 8)
df_label$status<-df_label$status-1 # substract 1 from all status codes so that we have a range 0 to 7
unique(df_label$status)
df_label<-to_categorical(df_label$status, num_classes=8) # one-hot encode the status labels into 8 cols V1 to V8
View(df_label)
df<-df[,-"status"] # remove status label from the original df


# remove the ID column
df<-df[,-"ID"]

# look at the unique values in what seems to be categorical input variables
unique(df$CODE_GENDER) # M and F -> encode as 1 col 1/0
unique(df$FLAG_OWN_CAR) # Y and N -> encode as 1 col 1/0
unique(df$FLAG_OWN_REALTY) # Y and N -> encode as 1 col 1/0
unique(df$FLAG_WORK_PHONE) # 0 and 1, no further encoding needed
unique(df$FLAG_PHONE) # 0 and 1, no further encoding needed
unique(df$FLAG_EMAIL) # 0 and 1, no further encoding needed
unique(df$FLAG_MOBILE) # all values are null -> drop this column
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

# Flag mobiles: drop column, all values are equal
df<-df[,-"FLAG_MOBIL"]

# Occupation Type: fill empty values with a string "None" so that it will be one-hot encoded as well subsequently
df$OCCUPATION_TYPE<-replace_na(df$OCCUPATION_TYPE, "None")

# All other catgorical features are muti-class (>2 classes) and need special one-hot encoding into one col per unique class value
df <- dummy_cols(df, select_columns = c(
  "NAME_INCOME_TYPE",
  "NAME_EDUCATION_TYPE",
  "NAME_FAMILY_STATUS",
  "NAME_HOUSING_TYPE",
  "OCCUPATION_TYPE"
), remove_first_dummy = FALSE, remove_selected_columns = TRUE)

# take a loot at the modified dataframe
View(df)

# Total income: huge outliers - we apply a treshold, then normalize the feature (scale to 0-1)
describe_feature(df$AMT_INCOME_TOTAL, "Total Income Amount") # huge outliers, we apply a threshold of 1.5 * IQR: Q3 + 1.5 * (Q3 - Q1) = 225K + 1.5 * (225K - 112.5K) = 393.75K
df$AMT_INCOME_TOTAL<-apply_threshold(df$AMT_INCOME_TOTAL, threshold = 393750)
df$AMT_INCOME_TOTAL<-min_max_normalize(df$AMT_INCOME_TOTAL)

# Days since birth: it is more intuitive to consider days since birth as a positive value
describe_feature(df$DAYS_BIRTH, "Days since birth")
df$DAYS_BIRTH<-abs(df$DAYS_BIRTH)
df$DAYS_BIRTH<-min_max_normalize(df$DAYS_BIRTH)

# Days employed: the only positive value is improbable (365243 days, would be 1000 years). We assume that positive value
# indicates missing data. We will thus apply a threshold of 0 before normalizing the feature (scale to 0-1).
describe_feature(df$DAYS_EMPLOYED, "Days employmed")
df$DAYS_EMPLOYED<-apply_threshold(df$DAYS_EMPLOYED, threshold = 0)
describe_feature(df$DAYS_EMPLOYED, "Days employmed") # looks more like a powerlaw distribution now
df$DAYS_EMPLOYED<-min_max_normalize(df$DAYS_EMPLOYED)

# Children count: normalize the feature
describe_feature(df$CNT_CHILDREN, "Children count")
df$CNT_CHILDREN<-min_max_normalize(df$CNT_CHILDREN)

# Family members count: normalize the feature
describe_feature(df$CNT_FAM_MEMBERS, "Family members count")
df$CNT_FAM_MEMBERS<-min_max_normalize(df$CNT_FAM_MEMBERS)

# check correlations between family and children count
cor(df$CNT_CHILDREN, df$CNT_FAM_MEMBERS) # 0.8784203 -> strongly correlated, we keep only the family members count

# drop the CNT_CHILDREN because is heavily correlated / redundant with CNT_FAM_MEMBERS
df<-df[,-"CNT_CHILDREN"]


# we are done with the data preparation for the input variables - show a summary of the prepared df
summary(df)
View(df)



# =====================================================================
# TRAIN / TEST SET SPLITS
# =====================================================================


# TODO: balance the dataset with oversampling of the underrepresented classes
# TODO: replace with k-fold test/train split

test_split = 0.1
train_row =  (nrow(df)-round(nrow(df)*test_split, digits = 0))
test_row = (train_row + 1)

# prepare train data as a matrix (keras does not like dataframes)
data_train<-as.matrix(df[0:train_row,])
data_train_label<-as.matrix(df_label[0:train_row,])

# prepare test data as a matrix
data_test<-as.matrix(df[test_row:nrow(df),])
data_test_label<-as.matrix(df_label[test_row:nrow(df),])

#data_balanced_over <- ovun.sample(status ~., data =df_train , method = "over",N =50000*3)



# =====================================================================
# NEURAL NETWORK DESIGN AND FIT
# =====================================================================

#**
#* function that builds our model (so that we can call it several times)
#* 
build_model <- function(shape_input, shape_output) {
  
  # Prepare the gradient descent optimizer (Marco may have an older version of
  # Tensorflow < 2.3 becase some params in Keras optimizer_sgd changed name)
  SGD <- optimizer_sgd(
    learning_rate = 1e-6, # use "lr" in older releases of tensorflow !
    momentum = 0.9,
    weight_decay = 1e-6, # use "decay" in older releases of tensorflow !
    nesterov = FALSE,
    clipnorm = NULL,
    clipvalue = NULL)

  # amount of neurons in hidden layer: rule of thumb: mean of input and ouput shapes
  hidden_layer = round((shape_input+shape_output)/2, digits = 0)

  # Build the Keras network model
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = shape_input, activation = "relu", input_shape = c(shape_input)) %>%   # first layer, n = input shape
    layer_dense(units = hidden_layer, activation = "relu") %>%                                  # hidden laser, n ca. mean of input and outpu shapre (rule of thumb)
    layer_dense(units = shape_output, activation = "softmax")                                    # last hidden layer, n = number of classes, with softmax activation

  summary(model)
    
  model %>% compile(
    optimizer = SGD, # "rmsprop" or SDG
    loss = "categorical_crossentropy", 
    metrics = c("accuracy")
  )
}


# force deelte any lefotvers from previous runs!
rm(model)
rm(metrics)

# Train the model
shape_input=ncol(df)
shape_output=ncol(df_label)
model<-build_model(shape_input, shape_output)
model %>%
  fit(
    data_train, data_train_label,
    epochs = 1000,
    batch_size = 128
  )

# Evaluate the model
metrics<-model %>% evaluate(data_test, data_test_label)
metrics


