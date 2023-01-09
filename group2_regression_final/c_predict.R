# ---------------------------------------------------------------------
# Assignment 1 - REGRESSION - PREDICT
# ---------------------------------------------------------------------
#
# Group A2
# Dietrich Rordorf, Marco Lecci, Sarah Castratori
#
# This script is used to load unseen data, preprocess it, load the final
# winning model and make the predictions for unseen data.
#
# =====================================================================


# =====================================================================
# PRELIMINARIES
# =====================================================================

# clear environment
rm(list=ls())

# allow for reproducible results by setting seed = 1. We can also change the seed and run
# the script a few times to see if the regression model is robust.
set.seed(1)


# =====================================================================
# CSV PATH
# =====================================================================

# TODO: Gwen & Holger to add the path to their unseen data file
csv_filepath<-'./LCdata_unseen.csv'


# =====================================================================
# LOAD PACKAGE DEPENDENCIES PATH
# =====================================================================

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
  install.packages("xgboost")
}
install_packages_regression()

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
library('xgboost')



# =====================================================================
# LOAD DATA FROM CSV
# =====================================================================

df<-fread(csv_filepath)



# =====================================================================
# DATA PREPROCESSING
# =====================================================================

# Helper functions for preprocessing
# --------------------------------------------------------------------

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
#* function to employment length data
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

#**
#* function to regroup good verification status into one flag, this is based on findings from
#* dummy encoding
#*
handle_verification_status<-function(df) {
  return(
    ifelse(
      df$verification_status_combined == 'Verified' | 
        df$verification_status_combined == 'Source Verified'
      , 1, 0)
  )
}


# Drop irrelevant attributes
# --------------------------------------------------------------------
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
  total_rec_prncp,
  id,
  member_id,
  url,
  title,
  desc,
  funded_amnt,
  funded_amnt_inv,
  open_il_12m,
  open_il_24m,
  inq_last_12m,
  open_rv_12m,
  open_rv_24m,
  policy_code
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

# copy verification status into the main df
df$good_verification_status<-handle_verification_status(df)

# home ownership RENT vs MORTGAGE dummies to main df
df$home_ownership_RENT<-df_dummies$home_ownership_RENT
df$home_ownership_MORTGAGE<-df_dummies$home_ownership_MORTGAGE

# emp_length
df$emp_length<-handle_empt_length(df$emp_length)
df$emp_length_numeric<-as.numeric(df$emp_length)
df$emp_length_median<-df$emp_length_numeric
df$emp_length_median[is.na(df$emp_length_median)]<-6 # replace NAs with median (was 6 in our data)
df = subset(df, select = -c(
  emp_length,
  emp_length_numeric
))

# good_states
df$good_states<-handle_states(df)

# copy all purposes dummies to the main df - they show all good positive or negative corrleation with the target
df$purpose_credit_card<-df_dummies$purpose_credit_card
df$purpose_debt_consolidation<-df_dummies$purpose_debt_consolidation
df$purpose_other<-df_dummies$purpose_other
df$purpose_small_business<-df_dummies$purpose_small_business
df$purpose_moving<-df_dummies$purpose_moving
df$purpose_house<-df_dummies$purpose_house
df$purpose_medical<-df_dummies$purpose_medical
df$purpose_car<-df_dummies$purpose_car
df$purpose_major_purchase<-df_dummies$purpose_major_purchase
df$purpose_vacation<-df_dummies$purpose_vacation
df$purpose_home_improvement<-df_dummies$purpose_home_improvement
df$purpose_renewable_energy<-df_dummies$purpose_renewable_energy
df$purpose_wedding<-df_dummies$purpose_wedding

# initial_list_status (w/f) -> encode as 0/a
df$initial_list_status<-replace(df$initial_list_status, df$initial_list_status == 'w', 0)
df$initial_list_status<-replace(df$initial_list_status, df$initial_list_status == 'f', 1)
df$initial_list_status<-as.numeric(df$initial_list_status)

# application_type
df$application_type<-replace(df$application_type, df$application_type == 'INDIVIDUAL', 0)
df$application_type<-replace(df$application_type, df$application_type == 'JOINT', 1)
df$application_type<-as.numeric(df$application_type)

# drop the categorical features from the original df
df = subset(df, select = -c(
  verification_status_combined,
  purpose,
  home_ownership,
  addr_state
))


# Numerical features
# --------------------------------------------------------------------

# open_il_6m
df$open_il_6m<-handle_na(df$open_il_6m)

# annual_inc
df$annual_inc<-handle_na(df$annual_inc)
df$annual_inc_joint<-handle_na(df$annual_inc_joint)
df$annual_inc_combined<-ifelse(df$application_type==1, df$annual_inc_joint, df$annual_inc) # merge joint income onto income column
df$annual_inc_combined<-apply_threshold(df$annual_inc_combined, threshold = 150000) # applying 1.5*IRQ 
df = subset(df, select = -c(
  annual_inc,
  annual_inc_joint
))

# declared_dti
df$dti<-handle_na(df$dti)
df$dti_joint<-handle_na(df$dti_joint)
df$declared_dti<-ifelse(df$dti_joint==0, df$dti, df$dti_joint)
df = subset(df, select = -c(
  dti,
  dti_joint
))

# earliest_cr_line, format is Apr-1955
df$earliest_cr_line<-as.numeric(substr(df$earliest_cr_line, 5, 8))

# last_credit_pull_d, format is Apr-1955
df$last_credit_pull_d<-as.numeric(substr(df$last_credit_pull_d, 5, 8)) 
df$last_credit_pull_d<-handle_na(df$last_credit_pull_d)

# delinq_2yrs
df$delinq_2yrs<-handle_na(df$delinq_2yrs)

# mths_since_last_delinq
df$mths_since_last_delinq<-replace_na(df$mths_since_last_delinq,0)

# inq_last_6mths
df$inq_last_6mths<-replace_na(df$inq_last_6mths,0) # replace NA with 0
df$inq_last_6mths<-abs(df$inq_last_6mths) # fix typing errors: negative values are not meaningful

# inq_fi
df$inq_fi<-replace_na(df$inq_fi,0)

# mths_since_last_delinq
df$mths_since_last_delinq[is.na(df$mths_since_last_delinq)]<-ifelse(df$delinq_2yrs==0,0,df$delinq_2yrs*12)

# mths_since_last_record
df$mths_since_last_record<-replace_na(df$mths_since_last_record,0)

# open_acc
df$open_acc<-handle_na(df$open_acc)
df$open_acc<-apply_threshold(df$open_acc, threshold = 17) # threshold 1.5*IRQ

# open_acc_6m
df$open_acc_6m<-handle_na(df$open_acc_6m)

# tot_coll_amt
df$tot_coll_amt<-handle_na(df$tot_coll_amt)

# tot_cur_bal
df$tot_cur_bal<-handle_na(df$tot_cur_bal)
df$tot_cur_bal<-apply_threshold(df$tot_cur_bal, threshold = 280000) # threshold 1.5*IRQ

# pub_rec
df$pub_rec<-handle_na(df$pub_rec)

# revol_bal
df$revol_bal<-handle_na(df$revol_bal)
df$revol_bal<-apply_threshold(df$revol_bal, threshold = 35000) # threshold 1.5*IRQ

# revol_util
df$revol_util<-apply_threshold(df$revol_util, threshold = 91.45) # threshold 1.5*IRQ
df$revol_util<-handle_na(df$revol_util, 56) # 56 was the median in our data

# total_acc
df$total_acc<-apply_threshold(df$total_acc, threshold = 39.5) # threshold 1.5*IRQ
df$total_acc<-handle_na(df$total_acc, 24) # 24 was the median in our data

# collections_12_mths_ex_med
df$collections_12_mths_ex_med<-ifelse(df$collections_12_mths_ex_med>=1, 1, 0) # try convert to a binary flag
df$collections_12_mths_ex_med<-handle_na(df$collections_12_mths_ex_med)

# mths_since_last_major_derog
df$mths_since_last_major_derog<-handle_na(df$mths_since_last_major_derog)

#total_bal_il
df$total_bal_il<-handle_na(df$total_bal_il)

#all_util
df$all_util<-handle_na(df$all_util)

# acc_now_delinq
df$acc_now_delinq<-handle_na(df$acc_now_delinq)

# mths_since_rcnt_il
df$mths_since_rcnt_il<-handle_na(df$mths_since_rcnt_il)

# il_util
df$il_util<-handle_na(df$il_util)

# max_bal_bc
df$max_bal_bc<-handle_na(df$max_bal_bc)

# total_rev_hi_lim
df$total_rev_hi_lim<-handle_na(df$total_rev_hi_lim)

# total_cu_tl
df$total_cu_tl<-handle_na(df$total_cu_tl)


# Engineered features
# --------------------------------------------------------------------

# has_wrong_doing
df$has_wrong_doing<-ifelse(df$pub_rec==0 & df$delinq_2yrs==0 & df$mths_since_last_major_derog==0,0,1)

# months_since_bad_situation
df$months_since_bad_situation<-(df$mths_since_last_major_derog+df$mths_since_last_delinq+df$mths_since_last_record)

# loan_to_wealth_index
df$loan_to_wealth_index<-(df$loan_amnt/(1+df$tot_cur_bal+7*df$annual_inc))
df$loan_to_wealth_index<-apply_threshold(df$loan_to_wealth_index, 0.045) # treshold 1.5*IRQ

# good_employment
df$good_employment<-ifelse(grepl("physician|chief|professor|attorney|scientist|advisory|executive|financial|project|consultant|teacher|software|president|manager|owner|director|analyst|engineer|senior|President|Manager|Owner|Director|Analyst|Engineer|Senior|Software|Teacher|Consultant|Project|Financial|Executive|Advisory|Scientist|Attorney|Professor|Chief|Physician", df$emp_title)==TRUE,1,0)
df = subset(df, select = -c(
  emp_title
))


# =====================================================================
# FEATURE SELECTION
# =====================================================================

df_selection<-df[, c(
  "revol_util",
  "loan_to_wealth_index",
  "inq_last_6mths",
  "good_verification_status",
  "purpose_credit_card",
  "declared_dti",
  "loan_amnt",
  "total_rev_hi_lim",
  "initial_list_status",
  "annual_inc_combined",
  "earliest_cr_line",
  "purpose_debt_consolidation",
  "purpose_other",
  "months_since_bad_situation",
  "good_employment", 
  "tot_cur_bal",
  "purpose_small_business",
  "home_ownership_RENT",
  "delinq_2yrs",
  "pub_rec",
  "total_acc",
  "mths_since_last_delinq",
  "purpose_moving",
  "purpose_house",
  "mths_since_rcnt_il",
  "max_bal_bc",
  "purpose_medical",
  "good_states",
  "revol_bal",
  "acc_now_delinq"
)] 


# =====================================================================
# FEATURE SCALING
# =====================================================================

min_max_scale<-function(x, known_min, known_max) {
  range = (known_max - known_min)
  x<-ifelse(x < known_min, known_min, x)
  x<-ifelse(x > known_max, known_max, x)
  return ((x-known_min)/range)
}

# we use fix min and max values based on the dataset so that we can copy this 1:1 to the prediction file
df_selection$revol_util<-min_max_scale(df_selection$revol_util, 0, 92)
df_selection$loan_to_wealth_index<-min_max_scale(df_selection$loan_to_wealth_index, 0, 0.045)
df_selection$inq_last_6mths<-min_max_scale(df_selection$inq_last_6mths, 0, 33)
df_selection$declared_dti<-min_max_scale(df_selection$declared_dti, 0, 43.5)
df_selection$loan_amnt<-min_max_scale(df_selection$loan_amnt, 500, 35000)
df_selection$total_rev_hi_lim<-min_max_scale(df_selection$total_rev_hi_lim, 0, 9999999)
df_selection$annual_inc_combined<-min_max_scale(df_selection$annual_inc_combined, 0, 150000)
df_selection$earliest_cr_line<-min_max_scale(df_selection$earliest_cr_line, 1944, 2012)
df_selection$months_since_bad_situation<-min_max_scale(df_selection$months_since_bad_situation, 0, 420)
df_selection$tot_cur_bal<-min_max_scale(df_selection$tot_cur_bal, 0, 280000)
df_selection$delinq_2yrs<-min_max_scale(df_selection$delinq_2yrs, 0, 39)
df_selection$pub_rec<-min_max_scale(df_selection$pub_rec, 0, 63)
df_selection$total_acc<-min_max_scale(df_selection$total_acc, 0, 39)
df_selection$mths_since_last_delinq<-min_max_scale(df_selection$mths_since_last_delinq, 0, 188)
df_selection$mths_since_rcnt_il<-min_max_scale(df_selection$mths_since_rcnt_il, 0, 363)
df_selection$revol_bal<-min_max_scale(df_selection$revol_bal, 0, 35000)
df_selection$acc_now_delinq<-min_max_scale(df_selection$acc_now_delinq, 0, 14)
df_selection$max_bal_bc<-min_max_scale(df_selection$max_bal_bc, 0, 83047)


# =====================================================================
# LOAD MODEL AND PREDICT
# =====================================================================

model_xgboost = readRDS('./winning_model.rda')
yhat<-predict(model_xgboost, data.matrix(df_selection))

yhat

print("Script completed. Please find the predictions in variable yhat!")