# ---------------------------------------------------------------------
# Assignment 1 - REGRESSION - PREPROCESS
# ---------------------------------------------------------------------
#
# Group A2
# Dietrich Rordorf, Marco Lecci, Rizoanun Nasa, Sarah Castratori
#
# This script is used to preprocess the raw data from CSV to a useable
# state.
#
# =====================================================================

# allow for reproducible results
set.seed(1)

# install package dependencies
install.packages("scales")

# load packages
library("scales")

# unzip raw data
unzip("./A1_regression/LCdata.csv.zip", exdir = "./A1_regression")

# load raw data from csv into data frame
df <- read.csv("./A1_regression/LCdata.csv", sep=";")

# drop id attributes (bear no meaning)
df = subset(df, select = -c(id, member_id, url))

# drop attributes that are not present for new applicants / in unseen data - list provided by Gwen
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

# dump csv with attributes dropped into CSSV for e.g., further analysis in Tableau Prep
write.csv(df, "./A1_regression/LCdata_0_dropped.csv")

# print basic description of data frame
str(df)

# Initial observations:
# --> after inital drop of variables, we have 52 variables left and ~798K observations
# --> existing versus new applicants: new applicants have many data attributes missing (0 or NA) --> we may drop some of these attributes
# --> some variables have many missing values (NA) or many 0 values (sparse variables) --> some customers are existing customers of LC versus new customers that may not have many of these attributes
# --> numeric variables have several orders of magnitude difference (we need scale numeric features)
# --> some applications are "joint" applications (co-borrowers, see field "verified_status_joint"), can be interpreted in order "not verified" < "income source verified" < "verified by LC"
# --> int_rate is outr label variable (the one to predict)
#
# Possible preprocessing operations:
#  - loan_amount: convert to num and scale / normalize
#  - funded_amnt: convert to num and scale / normalize
#  - funded_amnt_inv: look-up the meaning in terminology
#  - emp_title: code as dummy variables (check first the amount of values)
#  - emp_length: convert to months (numeric) and scale / normalize
#  - home_ownership: coded as string, may possible be interpreted in an order NONE < RENT < MORTGAGE or code as dummy vars
#  - annual_inc: some missing values, probably an important predictor; we need to find a strategy to sample for the missing values (mean, median, nearest neighbour)
#  - verification_status: coded as string, may possible be interpreted in an order NOT VERIF < VERIF < VERIF BY LC or code as dummy vars

# --> "term" is coded as string such as "36 months"
# --> "emp_length" is coded as string such as "3 years" or "< 1 year"
# --> "home_ownership" is coded as string, may possible be interpreted in an order NONE < RENT < MORTGAGE

# --> "issue_d" seems to indicate when the loan was issued - this variable is not future-proof cannot be used like this - may somehow need to convert into age of the loan in months
# --> "loan_status" is unstructured but includes a status that we may need to extract
# --> "url" includes and id and may be removed as it does not seem to bear any meaning
# --> "desc" and "title" may need some NLP treatment
# --> "zip_code" may need to be translated into latitude / longitude to be used as more appropriate geographical indicator
# --> "last_pymnt_d" and "next_pymnt_d" are coded as dates, we may need to compute the delta in months instead

# 
# scaling: we may keep the minimum as 0 and use a percentage such as 99% to cut-off outliers. Outliers are replaced with the treshold value. For instance for income if
# threshold is 100'000 USD we will replace all outliers >100'000 USD with 100'000 USD. Then we scale from 0-1 scale.
#
# data is from 2007 to 2015

# check for sparsity of attributes
percent(colMeans(is.na(df)))

# missing values in:
# annual_inc --> <0.01%
# delinq_2yrs --> <0.01%
# inq_last_6mths --> <0.01%
# mths_since_last_delinq --> 51%
# mths_since_last_record --> 84%
# open_acc
# pub_rec
# revol_bal
# revol_util
# total_acc
# collections_12_mths_ex_med
# mths_since_last_major_derog
# annual_inc_joint
# dti_joint

# a closer look at the int_rate, which is the label attribute
structure(df$int_rate)
hist(df$int_rate)
summary(df$int_rate)

# interest rate is in the range 5.32 - 28.99 --> we may divide by 100 - but if we do, the last step in our prediction of interest rate will be to multiply again with 100 to have same order of magnitude
