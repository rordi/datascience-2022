# ---------------------------------------------------------------------
# Assignment 1 - REGRESSION - PREPROCESS
# ---------------------------------------------------------------------
#
# Group A2
# Dietrich Rordorf, Marco Lecci, Sarah Castratori
#
# This script was used to preprocess the raw data from CSV to a usable
# state, do some feature selection and feature engineering, and experiment
# with different regression models.
#
# =====================================================================



# =====================================================================
# PRELIMINARIES
# =====================================================================

# clear envionrment
rm(list=ls())

# allow for reproducible results by setting seed = 1. We can also change the seed and run
# the script a few times to see if the regression model is robust.
set.seed(1)

# set memory limit: this works only on Windows (you can copy-paste the below commented-out command into console and run it there)
# memory.limit(700000)

# install other package dependencies
install_packages_regression<-function() {
  install.packages("corrplot")
  install.packages("dplyr")
  install.packages("scales")
  install.packages("data.table")
  install.packages("skimr")
  install.packages("fastDummies")
  install.packages("car")
  install.packages("tm")
  install.packages("SnowballC")
  install.packages("randomForest")
  install.packages("reshape")
  install.packages("Metrics")
  install.packages("caret")
  install.packages("gbm")
}
#install_packages_regression() # uncomment when running first time!

# load packages
library("corrplot")
library("dplyr")
library("scales")
library("stringr")
library("data.table")
library("tidyverse")
library("skimr")
library("fastDummies")
library("car")
library("tm")
library("SnowballC")
library("randomForest")
library("reshape")
library("Metrics")
library("rpart")
library("caret")
library('gbm')



# =====================================================================
# LOAD RAW DATA
# =====================================================================

# NOTE: path are relative to the root of the project, thus paths need to start with "./A1_regression/" prefix !

# unzip raw data (uncomment line below if you do not have the CSV yet locally)
#unzip("./A1_regression/LCdata.csv.zip", exdir = "./A1_regression")

# load raw data from csv into data frame
df_raw <- fread("./A1_regression/LCdata.csv", sep=";")

# first we randomize the order in the dataframe before doing anything else (we do not know how random the order is in the CSV)!
df_raw[sample(1:nrow(df_raw)), ]

# copy the raw data in df, which will be our working copy
df<-df_raw



# =====================================================================
# PURGE IRRELEVANT ATTRIBUTES
# =====================================================================

# drop attributes that are not present for new applicants / in unseen data -
# this is a list of attributes that was provided by Gwen in class
df = subset(df, select = -c(
  collection_recovery_fee,
  installment,
  issue_d,
  last_pymnt_amnt,
  last_pymnt_d,
  loan_status,
  next_pymnt_d,
  out_prncp,
  out_prncp_inv,
  pymnt_plan,
  recoveries,
  term,
  total_pymnt,
  total_pymnt_inv,
  total_rec_int,
  total_rec_late_fee,
  total_rec_prncp
))

# drop more attributes that we identified as irrelevant
df = subset(df, select = -c(
  id,             # only providing an order in which the applications were saved into the database otherwise meaningless
  member_id,      # too many unique values to be reasonably used
  url             # the url contains the id, only providing an order in which the applications were saved into the database otherwise meaningless
))


# drop attributes that have no description in the data dictionary (it is uncertain what exactly it is). However, based 
# on experiments Marco found that some are still useful, so commented this out again and keep these attributes
#df = subset(df, select = -c(
#  verification_status,        # could be is_inc_v but not certain
#  verification_status_joint   # could be verified_status_joint but not certain
#))

# dump csv with attributes dropped into CSSV for e.g., further analysis in Tableau Prep
# write.csv(df, "./A1_regression/LCdata_0_dropped.csv")


# =====================================================================
# HANDLE EMPTY RECORDS
# =====================================================================

# it seems some attributes have excatly 25 NAs - a closer look reveals that it is the same record IDs in different attributes
# (meaning that these rows seem filled with missing values)
which(is.na(df$delinq_2yrs))
which(is.na(df$inq_last_6mths))
which(is.na(df$open_acc))
which(is.na(df$pub_rec))
which(is.na(df$total_acc))
which(is.na(df$acc_now_delinq))

# drop those 25 empty rows from the df
df<-df[-which(is.na(df$delinq_2yrs))]
df_raw<-df_raw[-which(is.na(df_raw$delinq_2yrs))] # make df_raw same length as df so that we can easily re-copy an attribute to df later on if we make mistakes


# =====================================================================
# CREATE FRAME FOR OUR TARGET VARIABLE
# =====================================================================
#
# this is a step that WILL NOT BE DONE in the final script - the final script is not supposed to contain the int_rate!

# we take a closer look at the int_rate, which is our target variable!
df_int_rate <- df[, c("int_rate")]
df_int_rate$int_rate<-(df_int_rate$int_rate/100) # scale percentage to numeric [TODO - we will need to scale back MSE according to Gwen's email]
df = subset(df, select = -c(
  int_rate # remove or target from the working copy data frame
))


# =====================================================================
# CREATE NLP FRAME FOR FREE-TEXT STRING ATTRIBUTES
# =====================================================================

# create the NLP dataframe
df_nlp<-df[,c("title", "emp_title", "desc")]

# remove string attributes from the main dataframe (we can copy back later interesting attributes from NLP frame after processing - our main dataframe should be purely numeric!)
df = subset(df, select = -c(
  title,         # the title of the loan application -- @TODO -- code as dummy variables / NLP? (ca. 56K unique values)
  emp_title,     # the job title of the loan applicant -- @TODO -- code as dummy variables / NLP? (ca. 265K unique values)
  desc           # some free text provided by loan applicant on why the want / need to borrow -- @TODO -- use NLP? (86% null, ca. 112K unique values)
))



# =====================================================================
# INITIAL OBSERVATIONS
# =====================================================================

# print basic description of data frame
dim(df)    # we have 49 attributes (incl. target) left after initial selection and ~798K observations
skim(df)
#View(df) # uncomment to open the table view of the df

# we need to filter for numeric variables to use corrplot and also handle NAs first...
# corrplot(df)

# Initial observations:
#
# --> after initial drop of variables, we have 49 variables left and ~798K observations
# --> existing versus new applicants: new applicants have many data attributes missing (0 or NA) --> we may drop some of these attributes
# --> some variables have many missing values (NA) or many 0 values (sparse variables) --> some customers are existing customers of LC versus new customers that may not have many of these attributes
# --> numeric variables have several orders of magnitude difference (we need scale numeric variables)
# --> some applications are "joint" applications (co-borrowers, see field "verified_status_joint"), can be interpreted in order "not verified" < "income source verified" < "verified by LC"
# --> int_rate is our label variable (the one to predict)
# --> several variables show a NA count of exactly 25 (and seems that it is ~25 rows that ar mostly NAs)
# --> policy_code has always the same value (1) - we can drop it
# --> "home_ownership" is coded as string, may possible be interpreted in an order NONE < RENT < MORTGAGE
# --> (out?) "issue_d" seems to indicate when the loan was issued - this variable is not future-proof cannot be used like this - may somehow need to convert into age of the loan in months
# --> "loan_status" is unstructured but includes a status that we may need to extract
# --> "desc" and "title" may need some NLP treatment
# --> "last_pymnt_d" and "next_pymnt_d" are coded as dates, we may need to compute the delta in months instead
# -->  "NAs in mths_since_last_delinq refers to people who never commit a crime"
# --> annual_inc: some missing values, probably an important predictor; we need to find a strategy to sample for the missing values (mean, median, nearest neighbour)
# --> verification_status: coded as string, may possible be interpreted in an order NOT VERIF < VERIF < VERIF BY LC or code as dummy vars
# --> "term" is coded as string such as " 36 months" 
#
# scaling: we may keep the minimum as 0 and use a percentage such as 99% to cut-off outliers. Outliers are replaced with the treshold value.
# For instance for income if threshold is 150'000 USD we will replace all outliers >150'000 USD with 150'000 USD. Then we scale from 0-1 scale.
# Gwen advised against it in coaching, pointint that the treshold value would be arbitrary. We can stick to the 1.5 * IRQ definition used in box plots
# to stick to a common definition of outliers, or use something like 2 stdev around median. Just droping the outliers from the training set it not
# a wise choice: in the unseen data we will also likely have some outliers - in unseen data we can not drop them but have to provide a prediction!
#
# data is from 2007 to 2015, we can assume that interest rates follow some temporal economic-cyclical pattern. However, Gwen mentioned in class that
# we should full ignore this. In reality we would probably want to predict the interest spread from some inter-bank load interest rate (e.g. LIBOR).




# =====================================================================
# HELPER FUNCTIONS TO DESCRIBE AND PREPROCESS FEATURES
# =====================================================================

# select the functions and run them once to define them in the env. Then you can also use them on the command line !

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
  
  # what is the correlation of the feature with our target variable?
  message(paste("Correlation with target variable: ", cor(feature_handled,df_int_rate$int_rate)))
  
  # outliers detection (exclude null values as we have some very sparse attributes)
  outliers<-boxplot.stats(feature_handled)$out
  outliers_count<-length(outliers)
  message(paste("Number of outliers (1.5*IQR method): ", outliers_count))
  if (outliers_count>0) {
    # compute the threshold for outliers according to 1.5*IRQ meth
    q1=quantile(feature_handled, 0.25, na.rm=TRUE)
    q3=quantile(feature_handled, 0.75, na.rm=TRUE)
    iqr=q3-q1
    iqr_threshold=q1+1.5*iqr
    message(paste("Threshold for outliers (1.5*IQR method): ", iqr_threshold))
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
#* function to convert zip codes to numeric
#* 
handle_zip <- function(feature, divide_by = 1) {
  feature_handled<-sub("xx", "", feature) # replace "xx" in zip codes (e.g. "123xx")
  feature_handled<-as.numeric(feature_handled)
  feature_handled<-round(feature_handled/divide_by, digits = 0)   # we found that reducing to 1 or 2 digits does not improve the attribute
  return(feature_handled)
}

#**
#* function to handle states data; this is based on findings from dummy encoding
#* using the states with negative corr. coeff. ("good states") improved the result
#* 
handle_states<-function(df) {
  return(
    ifelse(
      df$addr_state=="MA" | 
        df$addr_state=="CA" |
        df$addr_state=="MA" | 
        df$addr_state=="CA" | 
        df$addr_state=="IL" | 
        df$addr_state=="NH" | 
        df$addr_state=="WI" | 
        df$addr_state=="CO" | 
        df$addr_state=="DC" | 
        df$addr_state=="TX" | 
        df$addr_state=="NJ" | 
        df$addr_state=="CT" | 
        df$addr_state=="AZ" | 
        df$addr_state=="ME" | 
        df$addr_state=="MN" | 
        df$addr_state=="MT" | 
        df$addr_state=="VT" | 
        df$addr_state=="OR" | 
        df$addr_state=="RI" | 
        df$addr_state=="ID" | 
        df$addr_state=="ND" | 
        df$addr_state=="NE" | 
        df$addr_state=="IA" | 
        df$addr_state=="MO"
      , 1, 0)
  ) 
}

#**
#* function to employment length data; this is based on findings from dummy encoding
#* using the states with negative corr. coeff. ("good states") improved the result
#* It is not a beauty of programming but it works ;)
#* 
handle_empt_length<-function(feature) {
  return(ifelse(feature=="< 1 year", 0.5,
                ifelse(feature=="1 year", 1,
                       ifelse(feature=="2 years", 2,
                              ifelse(feature=="3 years", 3,
                                     ifelse(feature=="4 years", 4,
                                            ifelse(feature=="5 years", 5,
                                                   ifelse(feature=="6 years", 6,
                                                          ifelse(feature=="7 years", 7,
                                                                 ifelse(feature=="8 years", 8,
                                                                        ifelse(feature=="9 years", 9,
                                                                               ifelse(feature=="10+ years", 10, NA))))))))))))
}

# =====================================================================
# FEATURE SELECTION OF EXISTING FEATURES
# =====================================================================

# In a next step we define our own function "describe_feature" and run it on every numerical attribute. The results are 
# recorded in a Google Sheet for better overview: https://docs.google.com/spreadsheets/d/1d9JnSfMhEuIjDAsVg6EK4g-UefP_-IXGu-S9b9Y1gbY


# Categorical features
# --------------------------------------------------------------------

# zip codes shows a high number of unique values and low correlation to target
# we also tried to use only 1 or 2 digits, but did not find any significant change in correlation
# thus we drop the ZIP. Also, we will use "good states" as a separate feature later on.
describe_feature(handle_zip(df$zip_code), "ZIP Codes 3 digits")
describe_feature(handle_zip(df$zip_code, divide_by = 10), "ZIP Codes 2 digits")
describe_feature(handle_zip(df$zip_code, divide_by = 100), "ZIP Codes 1 digits")
df = subset(df, select = -c(
  zip_code
))



# Dummy variables for categorical features
# --------------------------------------------------------------------

# create separate df with dummy variables for some of the categorical string attributes
df$verification_status_combined<-ifelse(df$verification_status_joint != '', df$verification_status_joint, df$verification_status) # combine verification status
df = subset(df, select = -c(
  verification_status,
  verification_status_joint
))


df_dummies<-df
numcol<-ncol(df_dummies)
df_dummies<-dummy_cols(df_dummies, select_columns = c("verification_status_combined", "purpose", "home_ownership", "addr_state"))
df_dummies<-df_dummies[,(numcol+1):ncol(df_dummies)]
#View(df_dummies) # uncomment to open the table view of the df

# here we compute the correlations with target variable for each of the dummy variables and sort them descending by absolute value
# (either strongly negative or strongly positive correlated on top). 
z<-cor(df_int_rate$int_rate, df_dummies)
z[z == 1] <- NA #drop perfect
z<-na.omit(melt(z)) # melt! 
z[order(-abs(z$value)),] # sort

# based on theses correlations, it seems the following dummy vars seem interesting tp pursue further!
#
#1   1    verification_status_combined_Not Verified -2.178569e-01
#3   1        verification_status_combined_Verified  2.106720e-01
#5   1                          purpose_credit_card -1.846414e-01
#6   1                   purpose_debt_consolidation  9.647653e-02
#13  1                                purpose_other  9.198582e-02
#15  1                       purpose_small_business  7.394402e-02
#23  1                          home_ownership_RENT  6.176908e-02
#19  1                      home_ownership_MORTGAGE -6.152897e-02
#12  1                               purpose_moving  4.402282e-02
#9   1                                purpose_house  4.061277e-02
#11  1                              purpose_medical  3.030260e-02
#4   1                                  purpose_car -2.892246e-02
#43  1                                addr_state_MA -2.033319e-02
#10  1                       purpose_major_purchase -1.813078e-02
#16  1                             purpose_vacation  1.807930e-02
#25  1                                addr_state_AL  1.289503e-02
#8   1                     purpose_home_improvement -1.209862e-02
#14  1                     purpose_renewable_energy  1.188388e-02
#35  1                                addr_state_HI  1.105064e-02
#17  1                              purpose_wedding  1.082496e-02
#66  1                                addr_state_TN  9.213030e-03

# copy home ownership RENT and MORTGAGE dummies to main df
df$home_ownership_RENT<-df_dummies$home_ownership_RENT
describe_feature(df$home_ownership_RENT, "Home (Rent)") # 0.062 correlation
df$home_ownership_MORTGAGE<-df_dummies$home_ownership_MORTGAGE
describe_feature(df$home_ownership_MORTGAGE, "Home (Mortgage)") # -0.062 correlation

# emp_length --> is coded as string such as "3 years" or "< 1 year" and needs conversion to numeric space
df$emp_length<-handle_empt_length(df$emp_length)
df$emp_length_numeric<-as.numeric(df$emp_length)
describe_feature(df$emp_length_numeric, "Employment Length")

# apply the median of the distribution to the NAs in employment length (probably more robust than mean as the distribution is skewed because of the
# last bin)
df$emp_length_median<-df$emp_length_numeric
df$emp_length_median[is.na(df$emp_length_median)]<-median(df$emp_length_median,na.rm=TRUE) # replace NAs with median
describe_feature(df$emp_length_median, "Employment Length (median applied)") # improves the correlation

df = subset(df, select = -c(
  emp_length,
  emp_length_numeric
))

# create a separate feature "good states" for the states that have neg. correlation coeff with the target
# variable (this worked better as group rather than using dummy encoded states individually)
df$good_states<-handle_states(df_raw)
describe_feature(df$good_states, "Good States") #-0.029 correlation

# initial_list_status (w/f) -> encode as 0/a
df$initial_list_status<-replace(df$initial_list_status, df$initial_list_status == 'w', 0)
df$initial_list_status<-replace(df$initial_list_status, df$initial_list_status == 'f', 1)
df$initial_list_status<-as.numeric(df$initial_list_status)
describe_feature(df$initial_list_status, "Initial list status") # 0.115 correlation


# initial_list_status (w/f) -> encode as 0/a
df$application_type<-replace(df$application_type, df$application_type == 'INDIVIDUAL', 0)
df$application_type<-replace(df$application_type, df$application_type == 'JOINT', 1)
df$application_type<-as.numeric(df$application_type)
describe_feature(df$application_type, "Applcation type") # 0.010 correlation


# drop the categorical features from the original df
df = subset(df, select = -c(
  verification_status_combined,
  purpose,
  home_ownership,
  addr_state
))


# Numeric features
# --------------------------------------------------------------------

# loan amount is the amount requested by the borrowers, while funded amounts are what investors committed and what was finally borrowed
# so we believe funded amounts data will actually only be available after the interest rate was computed. Thus we keep the loan_amount but
# drop the other two amount attributes!
describe_feature(df$loan_amnt, "Loan Amount")
describe_feature(df$funded_amnt, "Funded Amount")
describe_feature(df$funded_amnt_inv, "Funded Invested Amount")

cor(df$loan_amnt, df$funded_amnt)        # correlation: 0.9992714 between loan and funded amount
cor(df$loan_amnt, df$funded_amnt_inv)    # correlation: 0.9971339 between loan and funded by investors amount
df = subset(df, select = -c(
  funded_amnt,
  funded_amnt_inv
))


# opened installment accounts (current, past 12 months, past 24 months)
# because if any was opened in past 6 months, it will also be included in 12 and 24 months. 6 month shows strongest correlation with target,
# thus we finally drop the 12m and 24m attributes
describe_feature(df$open_il_6m, "Current Installment Accounts")
describe_feature(df$open_il_12m, "Installment Accounts 12 months")
describe_feature(df$open_il_24m, "Installment Accounts 24 months")
df$open_il_6m<-handle_na(df$open_il_6m)
df$open_il_12m<-handle_na(df$open_il_12m)
df$open_il_24m<-handle_na(df$open_il_24m)
cor(df$open_il_6m, df$open_il_12m)   # correlation: 0.57 between 6m and 12m
cor(df$open_il_6m, df$open_il_24m)   # correlation: 0.67 between 6m and 24m
cor(df$open_il_12m, df$open_il_24m)  # correlation: 0.85 between 12m and 24m
df = subset(df, select = -c(
  open_il_12m,
  open_il_24m
))


# annual_inc
# (!) missing values should refer to 0 income people: according to dataset documentation these are usually students
# or people that do not have an employment descrition. However doesn't impact the type of replacement because there are 4 in total.
df$annual_inc<-handle_na(df$annual_inc)
df$annual_inc_joint<-handle_na(df$annual_inc_joint)

describe_feature(df$annual_inc, "Annual Income")
describe_feature(df$annual_inc_joint, "Annual Income (joint)")

df$annual_inc_combined<-ifelse(df$application_type==1, df$annual_inc_joint, df$annual_inc) # merge joint income onto income column
describe_feature(df$annual_inc_combined, "Annual Income (combined)") # -0.073 correlation
df$annual_inc_combined<-apply_threshold(df$annual_inc_combined, threshold = 112500) # applying 1.5*IRQ 
describe_feature(df$annual_inc_combined, "Annual Income (combined, thresholded)") # -0.11 correlation

df = subset(df, select = -c(
  annual_inc,
  annual_inc_joint
))

# dti
describe_feature(df$dti, "DTI") # has outliers, 0.077 correlation
df$dti<-handle_na(df$dti)

# dti_joint
describe_feature(df$dti_joint, "DTI Joint") # has outliers, 0.012 correlation
df$dti_joint<-handle_na(df$dti_joint)

# new feature: declared_dti (combining dti and dti_joint) --> declared_dti brings benefit, correlation increase from 7.7% to 16.4%.
df$declared_dti<-ifelse(df$dti_joint==0, df$dti, df$dti_joint)
describe_feature(df$declared_dti, "Declared DTI") # after joining, only 1 outlier left, correlation  0.164

# wee keep the new declared_dti feature and drop the original ones
df = subset(df, select = -c(
  dti,
  dti_joint
))


# earliest_cr_line, format is Apr-1955, simply convert to year by taking substring of last 4 chars
df$earliest_cr_line<-as.numeric(substr(df$earliest_cr_line, 5, 8))
describe_feature(df$delinq_2yrs, "Earliest CR Line") # 0.055 correlation

# last_credit_pull_d, format is Apr-1955, simply convert to year by taking substring of last 4 chars
df$last_credit_pull_d<-as.numeric(substr(df$last_credit_pull_d, 5, 8)) 
df$last_credit_pull_d<-handle_na(df$last_credit_pull_d)
describe_feature(df$last_credit_pull_d, "Last credit pull date") # 0.001 correlation

# delinq_2yrs --> <0.01% missing values
describe_feature(df$delinq_2yrs, "Delinquencies 2 years") # 0.055 correlation
df$delinq_2yrs<-handle_na(df$delinq_2yrs)


# mths_since_last_delinq
describe_feature(df$mths_since_last_delinq, "Months since last delinquency") # 0.046 correlation
df$mths_since_last_delinq<-replace_na(df$mths_since_last_delinq,0)


# inq_last_6mths --> almost 50% of records have an entry, and seems a good indicator
describe_feature(df$inq_last_6mths, "Inquries past 6 months") # 0.228 correlation
df$inq_last_6mths<-replace_na(df$inq_last_6mths,0) # replace NA with 0
df$inq_last_6mths<-abs(df$inq_last_6mths) # fix typing errors: negative values are not meaningful


# inq_last_12mths --> sparse attribute, weak correlation with int_rate
describe_feature(df$inq_last_12m, "Inquries past 12 months") # 0.012 correlation
df$inq_last_12m<-replace_na(df$inq_last_12m,0) # replace NA with 0
df$inq_last_12m<-abs(df$inq_last_12m) # fix typing errors: negative values are not meaningful
cor(df$inq_last_12m,df_int_rate$int_rate) # -0.001%
cor(df$inq_last_12m,df$inq_last_6mths) #0.03% (although they seem independent, each inq in past 6 months is also an inq in past 12 months thus the two have dependency)

# we drop inq_last_12mths and inq_last_6mths is ma much better predictor
df = subset(df, select = -c(
  inq_last_12m
))


# inq_fi --> spare attribute, weak correlation
describe_feature(df$inq_fi, "Inquries 'FI'") # 0.002 correlation
df$inq_fi<-replace_na(df$inq_fi,0)


# mths_since_last_delinq --> 51% missing values
describe_feature(df$mths_since_last_delinq, "Months since last deliquency") # 0.046 correlation
df$mths_since_last_delinq[is.na(df$mths_since_last_delinq)]<-ifelse(df$delinq_2yrs==0,0,df$delinq_2yrs*12)
cor(df_int_rate$int_rate,df$mths_since_last_delinq)


# mths_since_last_record --> 84% missing values
describe_feature(df$mths_since_last_record, "Months since last record") # 0.061 correlation
df$mths_since_last_record<-replace_na(df$mths_since_last_record,0)


# open_acc
describe_feature(df$open_acc, "Open Account") # many outliers, -0.011 correlation
df$open_acc<-handle_na(df$open_acc)
df$open_acc<-apply_threshold(df$open_acc, threshold = 17) # threshold 1.5*IRQ
describe_feature(df$open_acc, "Open Account (tresholded)") # -0.013 correlation (still weak)

# open_acc_6m
describe_feature(df$open_acc_6m, "Open Account 6 months") # many outliers, -0.002 correlation
df$open_acc_6m<-handle_na(df$open_acc_6m)

# tot_coll_amt
describe_feature(df$tot_coll_amt, "Total collection amout") # many outliers, -0.001 correlation
df$tot_coll_amt<-handle_na(df$tot_coll_amt)

# tot_cur_bal
describe_feature(df$tot_cur_bal, "Total current balance") # many outliers, -0.075 correlation
df$tot_cur_bal<-handle_na(df$tot_cur_bal)

# pub_rec
describe_feature(df$pub_rec, "Public Record") # many outliers, -0.011 correlation
df$pub_rec<-handle_na(df$pub_rec)
cor(df_int_rate$int_rate,df$pub_rec)


# revol_bal
describe_feature(df$revol_bal, "Revolving Balance") # -0.036 correlation
df$revol_bal<-handle_na(df$revol_bal)


# revol_util (I investigated the NAs of the attribute but didn't find any explanation
# for the missing values. So I tried replacement with 0, mean and median, the latter
# had a slighly better correlation so I kept it.)
describe_feature(df$revol_util, "Revolving Util") # 0.268 correlation
df$revol_util<-apply_threshold(df$revol_util, threshold = 91.45) # threshold 1.5*IRQ
df$revol_util<-handle_na(df$revol_util, median(df$revol_util, na.rm = TRUE))
describe_feature(df$revol_util, "Revolving Util (thresholded, median)") # 0.269 correlation


# total_acc
describe_feature(df$total_acc, "Total Account") # -0.039 correlation
df$total_acc<-apply_threshold(df$total_acc, threshold = 39.5) # threshold 1.5*IRQ
df$total_acc<-handle_na(df$total_acc, median(df$total_acc, na.rm = TRUE))
describe_feature(df$total_acc, "Total Account (thresholded, median)") # -0.048 correlation


# collections_12_mths_ex_med (Power Law distribution, make sense to replace it with 0)
describe_feature(df$collections_12_mths_ex_med, "Collections past 12 months") # 0.013 correlation
df$collections_12_mths_ex_med<-ifelse(df$collections_12_mths_ex_med>=1, 1, 0) # try convert to a binary flag
df$collections_12_mths_ex_med<-handle_na(df$collections_12_mths_ex_med)
cor(df_int_rate$int_rate,df$collections_12_mths_ex_med) # 0.014 correlation, clightly better but still weak


# mths_since_last_major_derog
describe_feature(df$mths_since_last_major_derog, "Monsth since last mjr derogation") # 0.063 correlation
df$mths_since_last_major_derog<-handle_na(df$mths_since_last_major_derog)
cor(df_int_rate$int_rate,df$mths_since_last_major_derog)


#total_bal_il
describe_feature(df$total_bal_il, "Total Balance 'IL'") # -0.016 correlation
df$total_bal_il<-handle_na(df$total_bal_il)


#all_util
describe_feature(df$total_bal_il, "All util") # -0.016 correlation
df$all_util<-handle_na(df$all_util)


# open_rv_12m, open_rv_24m --> drop, both are weak and sparse and interdependent
describe_feature(df$open_rv_12m, "Open RY 12 months") # -0.01 correlation
describe_feature(df$open_rv_24m, "Open RY 24 months") # -0.01 correlation 
cor(handle_na(df$open_rv_12m), handle_na(df$open_rv_24m)) # correlation (dependency): 0.87
df = subset(df, select = -c(
  open_rv_12m,
  open_rv_24m
))


# acc_now_delinq
describe_feature(df$acc_now_delinq, "Account now delinquent") # 0.026 correlation
df$acc_now_delinq<-handle_na(df$acc_now_delinq)


# policy_code --> SD is 0, all values are identical
describe_feature(df$policy_code, "Policy code")
df = subset(df, select = -c(
  policy_code
))

# mths_since_rcnt_il
describe_feature(df$mths_since_rcnt_il, "Months since recent 'IL'") # -0.036 correlation
df$mths_since_rcnt_il<-handle_na(df$mths_since_rcnt_il)

# il_util
describe_feature(df$il_util, "'IL' Util") # -0.02 correlation
df$il_util<-handle_na(df$il_util)

# max_bal_bc
describe_feature(df$max_bal_bc, "Max Balance BC") # -0.036 correlation
df$max_bal_bc<-handle_na(df$max_bal_bc)

# total_rev_hi_lim
describe_feature(df$total_rev_hi_lim, "Total Revolving High Limit") # -0.14 correlation
df$total_rev_hi_lim<-handle_na(df$total_rev_hi_lim)

# total_cu_tl
describe_feature(df$total_cu_tl, "Total Cu Tl") # -0.014 correlation
df$total_cu_tl<-handle_na(df$total_cu_tl)

# double-check that we handled all features: we should not have NAs left in the df!
which(colSums(is.na(df))>0)

# show summary of the df after feature handling
skim(df)

# compute correlations between all our features and our target variable --> we can use it to
# drop some features that contribute only very little to explain the interest rate
z<-cor(df_int_rate$int_rate, df)
z[z == 1] <- NA #drop perfect
z<-na.omit(melt(z)) # melt! 
z[order(-abs(z$value)),] # sort









# =====================================================================
# FEATURE ENGINEERING ON TEXT FEATURES (NLP)
# =====================================================================


#NLP empl_title variable
NLP<-VCorpus(VectorSource(df_nlp$emp_title))
NLP<-tm_map(NLP,content_transformer(tolower))
NLP<-tm_map(NLP,removeNumbers)
NLP<-tm_map(NLP,removePunctuation)
NLP<-tm_map(NLP,removeWords,stopwords())
NLP<-tm_map(NLP,stemDocument, language = c("english")) 
NLP<-tm_map(NLP,stripWhitespace) 
NLP_m<-DocumentTermMatrix(NLP)
NLP_m1<-removeSparseTerms(NLP_m, 0.999)
NLP_dataset<-as.data.frame(as.matrix(NLP_m1))
NLP_dataset$int_rate<-df_int_rate$int_rate
cor(NLP_dataset$int_rate,NLP_dataset)

#Makes sense to me merge all the variables that represent a good job position together, below the formula for the deployment phase (in the candidate variables)
NLP_dataset$Good_employement<-NLP_dataset$engin+NLP_dataset$director+NLP_dataset$senior+NLP_dataset$manag+NLP_dataset$presid+NLP_dataset$analyst+NLP_dataset$project+NLP_dataset$system
NLP_dataset$Good_employement2<-ifelse(NLP_dataset$engin+NLP_dataset$director+NLP_dataset$senior+NLP_dataset$manag+NLP_dataset$presid+NLP_dataset$analyst+NLP_dataset$project>=1,1,0)

#NLP desc

NLP_desc<-VCorpus(VectorSource(df_nlp$desc))
NLP_desc<-tm_map(NLP_desc,content_transformer(tolower))
NLP_desc<-tm_map(NLP_desc,removeNumbers)
NLP_desc<-tm_map(NLP_desc,removePunctuation)
NLP_desc<-tm_map(NLP_desc,removeWords,stopwords())
NLP_desc<-tm_map(NLP_desc,stemDocument, language = c("english")) 
NLP_desc<-tm_map(NLP_desc,stripWhitespace) 
NLP_m_desc<-DocumentTermMatrix(NLP_desc)
NLP_desc1<-removeSparseTerms(NLP_m_desc, 0.999)
NLP_desc_dataset<-as.data.frame(as.matrix(NLP_desc1))
NLP_desc_dataset$int_rate<-df_int_rate$int_rate
cor(NLP_desc_dataset$int_rate,NLP_desc_dataset)
Base_model<- lm(int_rate~.,data=NLP_desc_dataset)
summary(Base_model)
# in my opinion it is a bit redundant if we consider that we have also purpose, but may be some pattern also here




# =====================================================================
# FEATURE ENGINEERING: CANDIDATE NEW VARIABLES
# =====================================================================

#Class_Income
df$Class_Income<-ifelse(df$annual_inc<15000,0,ifelse(df$annual_inc>=15000 & df$annual_inc<35000,1,ifelse(df$annual_inc>=35000 & df$annual_inc<60000,2,ifelse(df$annual_inc>=60000 & df$annual_inc<100000,3,4))))
cor(df_int_rate$int_rate,df$Class_Income)


#Has_something_wrong_done
df$Has_something_wrong_done<-ifelse(df$pub_rec==0 & df$delinq_2yrs==0 & df$mths_since_last_major_derog==0,0,1)
cor(df_int_rate$int_rate,df$Has_something_wrong_done)

#months_since_bad_situation
df$months_since_bad_situation<-(df$mths_since_last_major_derog+df$mths_since_last_delinq+df$mths_since_last_record)
cor(df$months_since_bad_situation,df_int_rate$int_rate)

#Loan_to_Wealth_index
df$Loan_to_Wealth_index<-df$loan_amnt/(1+df$tot_cur_bal+7*df$annual_inc)
cor(df_int_rate$int_rate,df$Loan_to_Wealth_index)

#Good_employment
df_nlp$Good_employment<-ifelse(grepl("physician|chief|professor|attorney|scientist|advisory|executive|financial|project|consultant|teacher|software|president|manager|owner|director|analyst|engineer|senior|President|Manager|Owner|Director|Analyst|Engineer|Senior|Software|Teacher|Consultant|Project|Financial|Executive|Advisory|Scientist|Attorney|Professor|Chief|Physician", df_nlp$emp_title)==TRUE,1,0)
cor(df_int_rate$int_rate,df_nlp$Good_employment)
df$Good_employment<-df_nlp$Good_employment

#(!) still possible to investigate how other variables can be combined to produce better predictors





# train split (choose the number of raw by changing the percentage in the row_train)
variables_for_prediction<-df[, c(
                               "Class_Income",
                               "emp_length2",
                               "total_acc",
                               "revol_bal",
                               "tot_cur_bal",
                               "purpose_car",
                               "Good_employment",
                               "inq_last_6mths",
                               "revol_util",
                               "Loan_to_Wealth_index",
                               "verification_status_Not Verified",
                               "declared_dti",
                               "Has_something_wrong_done",
                               "purpose_credit_card",
                               "purpose_debt_consolidation",
                               "purpose_house",
                               "purpose_medical",
                               "purpose_moving",
                               "purpose_other",
                               "purpose_small_business",
                               "int_rate"
                             )]
nrow_train<-round(nrow(variables_for_prediction)*0.75,0)
Train_data<-variables_for_prediction[0:nrow_train,]
Test_data<-variables_for_prediction[(nrow_train+1):nrow(variables_for_prediction),]




# =====================================================================
# COMPARE DIFFERENT REGRESSION MODELS
# =====================================================================

# Multiple Linear regression
Linear_regression<- lm(int_rate~.,data=Train_data)
Linear_regression_prediction<-predict(Linear_regression,Test_data)
print(paste0('MAE: ' , mae(Test_data$int_rate,Linear_regression_prediction)))
summary(Linear_regression)
vif(Base_model)


# Regression tree
Regression_tree <- rpart(int_rate ~., data=Train_data, control=rpart.control(cp=.0001))
Regression_tree_prediction<-predict(Regression_tree,Test_data)
print(paste0('MAE: ' , mae(Test_data$int_rate,Regression_tree_prediction)))


# Random Forest
random_forest <- randomForest(int_rate~., data=Train_data, maxnodes = 100, mtry=10, ntree = 200 )
random_forest_prediction<-predict(random_forest,Test_data)
print(paste0('MAE: ' , mae(Test_data$int_rate,random_forest_prediction)))

# AdaBoost
model_adaboost <- gbm(int_rate ~.,data = Train_data,
                      distribution = "gaussian",
                      cv.folds = 10,
                      shrinkage = .01,
                      n.minobsinnode = 10,
                      n.trees = 500)
model_adaboost_prediction<-predict(model_adaboost ,Test_data)
print(paste0('MAE: ' , mae(Test_data$int_rate,model_adaboost_prediction)))


# Multiple linear regression tree

# XgBoost
