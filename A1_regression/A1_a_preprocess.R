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
#install.packages("...")

# load packages
#library("...")

# unzip raw data
unzip("./A1_regression/LCdata.csv.zip", exdir = "./A1_regression")

# load raw data from csv into data frame
df <- read.csv("./A1_regression/LCdata.csv", sep=";")

# print basic description of data frame
str(df)

# initial observations:
# --> we have 72 variables and ~798K observations
# --> some variables have many missing values (NA) or many 0 values (sparse variables)
# --> numeric variables have several orders of magnitute difference (we may need to scale some numeric features to 0-1 scale after cutting off outliers)
# --> "id" and "member_id" may be removed as they do not seem to bear any meaning
# --> "term" is coded as string such as " 36 months"
# --> "emp_length" is coded as string such as "3 years" or "< 1 year"
# --> "home_ownership" is coded as string, may possible be interpreted in an order NONE < RENT < MORTGAGE
# --> "verification_status" is coded as string but might be a binary 0/1
# --> "issue_d" seems to indicate when the loan was issued - this variable is not future-proof cannot be used like this - may somehow need to convert into age of the loan in months
# --> "loan_status" is unstructured but includes a status that we may need to extract
# --> "url" includes and id and may be removed as it does not seem to bear any meaning
# --> "desc" and "title" may need some NLP treatment
# --> "zip_code" may need to be translated into latitude / longitude to be used as more appropriate geographical indicator
# --> "last_pymnt_d" and "next_pymnt_d" are coded as dates, we may need to compute the delta in months instead





