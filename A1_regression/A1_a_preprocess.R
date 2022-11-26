# ---------------------------------------------------------------------
# Assignment 1 - REGRESSION - PREPROCESS
# ---------------------------------------------------------------------
#
# Group A2
# Dietrich Rordorf, Marco Lecci, Sarah Castratori
#
# This script is used to preprocess the raw data from CSV to a useable
# state.
#
# =====================================================================


# =====================================================================
# PRELIMINARIES
# =====================================================================

# clear envionrment
rm(list=ls())

# set memory limit: this works only on Windows
# memory.limit(700000)

# allow for reproducible results
set.seed(1)

# install package dependencies
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
install.packages("Metrics")
install.packages("caret")
install.packages('gbm')

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
library("Metrics")
library("rpart")
library("caret")
library('gbm')


# =====================================================================
# LOAD RAW DATA
# =====================================================================

# unzip raw data
unzip("./A1_regression/LCdata.csv.zip", exdir = "./A1_regression")

# load raw data from csv into data frame
df <- fread("./A1_regression/LCdata.csv", sep=";")


# =====================================================================
# PURGE IRRELEVANT ATTRIBUTES
# =====================================================================

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

# dump more attributes that we identified as irrelevant
df = subset(df, select = -c(
  id,             # only providing an order in which the applications were saved into the database otherwise meaningless
  member_id,      # too many unique values to be reasonably used
  url             # the url contains the id, only providing an order in which the applications were saved into the database otherwise meaningless
))

# dump csv with attributes dropped into CSSV for e.g., further analysis in Tableau Prep
write.csv(df, "./A1_regression/LCdata_0_dropped.csv")


# =====================================================================
# CREATE NLP FRAME FOR STRING ATTRIBUTES
# =====================================================================

# create the NLP dataframe
df_nlp<-df[,c("title","emp_title","desc")]

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
dim(df)    # we have 49 variables left and ~798K observations
str(df)

# Initial observations:
# --> after initial drop of variables, we have 49 variables left and ~798K observations
# --> existing versus new applicants: new applicants have many data attributes missing (0 or NA) --> we may drop some of these attributes
# --> some variables have many missing values (NA) or many 0 values (sparse variables) --> some customers are existing customers of LC versus new customers that may not have many of these attributes
# --> numeric variables have several orders of magnitude difference (we need scale numeric features)
# --> some applications are "joint" applications (co-borrowers, see field "verified_status_joint"), can be interpreted in order "not verified" < "income source verified" < "verified by LC"
# --> int_rate is our label variable (the one to predict)
#
# Possible preprocessing operations:
#  - loan_amount: convert to num and scale / normalize
#  - emp_title: code as dummy variables (check first the amount of values)
#  - emp_length: convert to months (numeric) and scale / normalize
#  - home_ownership: coded as string, may possible be interpreted in an order NONE < RENT < MORTGAGE or code as dummy vars


# =====================================================================
# FEATURE SELECTION OF EXISTING FEATURES
# =====================================================================

# define a small helper function that describes a feature 
describe_feature <- function(feature) {
  # show number of NAs
  message(paste("Number of NAs: ", sum(is.na(feature))))
  
  # replace NA with 0 and show how many % are zero-values
  feature_handled<-replace_na(feature,0)
  message(paste("Zero-values in %: ", length(which(feature_handled == 0))/length(feature_handled)))
  
  # what is the correlation of the feature with our target variable?
  message(paste("Correlation with target variable: ", cor(feature_handled,df$int_rate)))
}


# loan amount is the amount requested by the borrowers, while funded amounts are what investors committed and what was finally borrowed
# so we believe funded amounts data will actually only be available after the interest rate was computed. Thus we keep the loan_amount but
# drop the other two amount attributes!
describe_feature(df$loan_amnt)
describe_feature(df$funded_amnt)
describe_feature(df$funded_amnt_inv)

cor(df$loan_amnt, df$funded_amnt)       # correlation: 0.9992714 between loan and funded amount
cor(df$loan_amnt, df$funded_amnt_inv)    # correlation: 0.9971339 between loan and funded by investors amount
df = subset(df, select = -c(
  funded_amnt,
  funded_amnt_inv
))

# verification status verified seems to be a good predictor, states not really and also purpose seems to have poor correlations
df<-dummy_cols(df,
               select_columns = c(
                 "verification_status",
                 "purpose",
                 "home_ownership",
                 "addr_state"
                 ),
               remove_first_dummy = FALSE
               )
df_dummies<-df[,53:121]
cor(df$int_rate,df_dummies)
cor_dummies<-sort(as.vector(cor(df$int_rate,df_dummies)))

#  - annual_inc: some missing values, probably an important predictor; we need to find a strategy to sample for the missing values (mean, median, nearest neighbour)
#  - verification_status: coded as string, may possible be interpreted in an order NOT VERIF < VERIF < VERIF BY LC or code as dummy vars
# --> "term" is coded as string such as " 36 months" 

# emp_length --> is coded as string such as "3 years" or "< 1 year" and needs conversion to numeric space
df$emp_length<-ifelse(df$emp_length=="< 1 year",0.5,ifelse(df$emp_length=="1 year",1,ifelse(df$emp_length=="2 years",2,ifelse(df$emp_length=="3 years",3,ifelse(df$emp_length=="4 years",4,ifelse(df$emp_length=="5 years",5,ifelse(df$emp_length=="6 years",6,ifelse(df$emp_length=="7 years",7,ifelse(df$emp_length=="8 years",8,ifelse(df$emp_length=="9 years",9,ifelse(df$emp_length=="10+ years",15,df$emp_length)))))))))))
df$emp_length<-as.numeric(df$emp_length)
describe_feature(df$emp_length)
df$emp_length[is.na(df$emp_length)]<-mean(df$emp_length,na.rm=TRUE) # replace NAs with mean
describe_feature(df$emp_length) # improves the correlation

df$emp_length2<- ifelse(df$emp_length<=2,0,ifelse(df$emp_length>2 & df$emp_length<=5,1,2))
#!!!!irrelevant we can leave it out: 0.6% correlation

# --> "home_ownership" is coded as string, may possible be interpreted in an order NONE < RENT < MORTGAGE
# --> (out?) "issue_d" seems to indicate when the loan was issued - this variable is not future-proof cannot be used like this - may somehow need to convert into age of the loan in months
# --> "loan_status" is unstructured but includes a status that we may need to extract
# --> "url" includes and id and may be removed as it does not seem to bear any meaning
# looks meaningless also to me
# --> "desc" and "title" may need some NLP treatment
# --> "zip_code" may need to be translated into latitude / longitude to be used as more appropriate geographical indicator
# --> "last_pymnt_d" and "next_pymnt_d" are coded as dates, we may need to compute the delta in months instead
#-->  "NAs in mths_since_last_delinq refers to people who never commit a crime"


# scaling: we may keep the minimum as 0 and use a percentage such as 99% to cut-off outliers. Outliers are replaced with the treshold value. For instance for income if
# threshold is 100'000 USD we will replace all outliers >100'000 USD with 100'000 USD. Then we scale from 0-1 scale.
#
# data is from 2007 to 2015

# check for sparsity of attributes
percent(colMeans(is.na(df)))
percent(colMeans(is.na(numbers)))


# missing values in:
# annual_inc --> <0.01%
# (!) missing values should refer to 0 income people, I saw the description and are usually students or people that doesn't have an employment desc. However doesn't impact the type of replacement because there are 4.
df$annual_inc<-replace_na(df$annual_inc,0)
cor(df$int_rate,df$annual_inc)

# dti 
df$dti_joint<-replace_na(df$dti_joint,0)
cor(df$int_rate,df$dti) # 7.7%

# new feature: declared_dti (combining dti and dti_joint) --> declared_dti brings benefit, correlation increase from 7.7% to 16.4%.
df$declared_dti<-ifelse(df$dti_joint==0,df$dti,df$dti_joint)
cor(df$int_rate,df$declared_dti) # 16.4%

# delinq_2yrs --> <0.01% missing values
df$delinq_2yrs<-replace_na(df$delinq_2yrs,0)

#mths_since_last_delinq
df$mths_since_last_delinq<-ifelse(df$delinq_2yrs==0,0,df$mths_since_last_delinq)
df$mths_since_last_delinq<-replace_na(df$mths_since_last_delinq,0)
cor(df$int_rate,df$mths_since_last_delinq)

# inq_last_6mths --> almost 50% of records have an entry, and seems a good indicator
sum(is.na(df$inq_last_6mths))
df$inq_last_6mths<-replace_na(df$inq_last_6mths,0) # replace NA with 0
df$inq_last_6mths<-abs(df$inq_last_6mths) # fix typing errors: negative values are not meaningful
length(which(df$inq_last_6mths == 0))/length(df$inq_last_6mths) # 56% zero values (sparse attribute)
cor(df$inq_last_6mths,df$int_rate) #22.8%

# inq_last_12mths --> sparse attribute, weak correlation with int_rate
sum(is.na(df$inq_last_12m))
df$inq_last_12m<-replace_na(df$inq_last_12m,0) # replace NA with 0
df$inq_last_12m<-abs(df$inq_last_12m) # fix typing errors: negative values are not meaningful
length(which(df$inq_last_12m == 0))/length(df$inq_last_12m) # 98% zero values (sparse attribute)
cor(df$inq_last_12m,df$int_rate) # -0.001%
cor(df$inq_last_12m,df$inq_last_6mths) #0.03% (although they seem independent, each inq in past 6 months is also an inq in past 12 months thus the two have dependency)

# we drop inq_last_12mths
df = subset(df, select = -c(
  inq_last_12m
))

# inq_fi --> spare attribute, weak correlation
sum(is.na(df$inq_last_12m))
df$inq_fi<-replace_na(df$inq_fi,0)
length(which(df$inq_fi == 0))/length(df$inq_fi) # 99% zero values (sparse attribute)
cor(df$inq_fi, df$int_rate) # 0.002

# we drop inq_fi
df = subset(df, select = -c(
  inq_fi
))

# mths_since_last_delinq --> 51% missing values
df$mths_since_last_delinq[is.na(df$mths_since_last_delinq)]<-ifelse(df$delinq_2yrs==0,0,df$delinq_2yrs*12)
cor(df$int_rate,df$mths_since_last_delinq)

# mths_since_last_record --> 84% missing values
df$mths_since_last_record<-replace_na(df$mths_since_last_record,0)
cor(df$int_rate,df$mths_since_last_record)

# open_acc

# pub_rec
df$pub_rec<-replace_na(df$pub_rec,0)
cor(df$int_rate,df$pub_rec)

# revol_bal
df$revol_bal<-replace_na(df$revol_bal,0)
cor(df$int_rate,df$revol_bal)

# revol_util (I investigated the NAs of the attribute but didn't find any explanation for the missing values. So I tried replacement with 0, mean and median, the latter had a slighly better correlation so I kept it.)
df$revol_util[is.na(df$revol_util)] <- median(df$revol_util, na.rm = TRUE)
cor(df$int_rate,df$revol_util)

# total_acc
df$total_acc[is.na(df$total_acc)] <- median(df$total_acc, na.rm = TRUE)
cor(df$int_rate,df$total_acc)

# collections_12_mths_ex_med (Power Law distribution, make sense to replace it with 0, btw it is irrelevant)
df$collections_12_mths_ex_med<-replace_na(df$collections_12_mths_ex_med,0)
cor(df$int_rate,df$total_acc)

# mths_since_last_major_derog
df$mths_since_last_major_derog<-replace_na(df$mths_since_last_major_derog,0)
cor(df$int_rate,df$mths_since_last_major_derog)

#All the open_il_"" (too many missing values and no replacement is useful to increase the correlation  the open_il because they are irrelevant)
df$open_il_24m<-replace_na(df$open_il_24m,0)
cor(df$int_rate,df$open_il_24m)
#total_bal_il (irrelevant)
df$total_bal_il<-replace_na(df$total_bal_il,0)
cor(df$int_rate,df$total_bal_il)
#all_util (irrelaevant)
df$all_util<-replace_na(df$all_util,0)
cor(df$int_rate,df$all_util)
#emp_lenght (irrelevant)
df$emp_length<-ifelse(df$emp_length=="< 1 year",0.5,ifelse(df$emp_length=="1 year",1,ifelse(df$emp_length=="2 years",2,ifelse(df$emp_length=="3 years",3,ifelse(df$emp_length=="4 years",4,ifelse(df$emp_length=="5 years",5,ifelse(df$emp_length=="6 years",6,ifelse(df$emp_length=="7 years",7,ifelse(df$emp_length=="8 years",8,ifelse(df$emp_length=="9 years",9,ifelse(df$emp_length=="10+ years",15,df$emp_length)))))))))))
df$emp_length<-as.numeric(df$emp_length)
df$emp_length[is.na(df$emp_length)]<-mean(df$emp_length,na.rm=TRUE)
df$emp_length2<-ifelse(df$emp_length<5,0,1)
cor(df$emp_length2,df$int_rate)

#acc_now_delinq
df$acc_now_delinq<-replace_na(df$acc_now_delinq,0)
cor(df$int_rate,df$acc_now_delinq)

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
NLP_dataset$int_rate<-df$int_rate
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
NLP_desc_dataset$int_rate<-df$int_rate
cor(NLP_desc_dataset$int_rate,NLP_desc_dataset)
Base_model<- lm(int_rate~.,data=NLP_desc_dataset)
summary(Base_model)
# in my opinion it is a bit redundant if we consider that we have also purpose, but may be some pattern also here

#Candidate New variables

#Class_Income
df$Class_Income<-ifelse(df$annual_inc<15000,0,ifelse(df$annual_inc>=15000 & df$annual_inc<35000,1,ifelse(df$annual_inc>=35000 & df$annual_inc<60000,2,ifelse(df$annual_inc>=60000 & df$annual_inc<100000,3,4))))
cor(df$int_rate,df$Class_Income)


#Has_something_wrong_done
df$Has_something_wrong_done<-ifelse(df$pub_rec==0 & df$delinq_2yrs==0 & df$mths_since_last_major_derog==0,0,1)
cor(df$int_rate,df$Has_something_wrong_done)

#months_since_bad_situation
df$months_since_bad_situation<-(df$mths_since_last_major_derog+df$mths_since_last_delinq+df$mths_since_last_record)
cor(df$months_since_bad_situation,df$int_rate)

#Loan_to_Wealth_index
df$Loan_to_Wealth_index<-df$loan_amnt/(1+df$tot_cur_bal+7*df$annual_inc)
cor(df$int_rate,df$Loan_to_Wealth_index)


# a closer look at the int_rate, which is the label attribute
# interest rate is in the range 5.32 - 28.99 --> we may divide by 100 - but if we do, the last step in our prediction of interest rate will be to multiply again with 100 to have same order of magnitude
summary(df$int_rate)
hist(df$int_rate, main="Interest Rates")
boxplot(df$int_rate, main="Interest Rates", ylab="Percent")
#tot_curr_bal (tried different replacement methods, mean has a better corr I kept it)
df$tot_cur_bal[is.na(df$tot_cur_bal)] <- mean(df$tot_cur_bal, na.rm = TRUE)
cor(df$int_rate,df$tot_cur_bal)

#tot_coll_amt (irrilevant, even with different replacement methods)
df$tot_coll_amt[is.na(df$tot_coll_amt)] <- mean(df$tot_coll_amt, na.rm = TRUE)
df$tot_coll_amt<-replace_na(df$tot_coll_amt,0)
cor(df$int_rate,df$tot_coll_amt)

#Good_employment
df_nlp$Good_employment<-ifelse(grepl("physician|chief|professor|attorney|scientist|advisory|executive|financial|project|consultant|teacher|software|president|manager|owner|director|analyst|engineer|senior|President|Manager|Owner|Director|Analyst|Engineer|Senior|Software|Teacher|Consultant|Project|Financial|Executive|Advisory|Scientist|Attorney|Professor|Chief|Physician", df_nlp$emp_title)==TRUE,1,0)
cor(df$int_rate,df_nlp$Good_employment)
df$Good_employment<-df_nlp$Good_employment

#(!) still possible to investigate how other variables can be combined to produce better predictors

# a closer look at the annual income
# range 0 - 9.5 mio., i.e. some one-sided huge outliers --> apply some threshold then scale
# 4 NAs --> apply some strategy to fill (mean / median / closest neighbour)
summary(df$annual_inc)
hist(df$annual_inc, main="Annual Income")
boxplot(df$annual_inc, main="Annual Income")


# we need to filter for numeric attrs to use corrplot

# corrplot(df)

glimpse(df)
summary(df)
skim(df)

#train split (choose the number of raw by changing the percentage in the row_train)

variables_for_prediction<-df[,c("Class_Income","emp_length2","total_acc","revol_bal","tot_cur_bal","purpose_car","Good_employment","inq_last_6mths","revol_util","Loan_to_Wealth_index","verification_status_Not Verified","declared_dti","Has_something_wrong_done","purpose_credit_card","purpose_debt_consolidation","purpose_house","purpose_medical","purpose_moving","purpose_other",
                                "purpose_small_business","int_rate")]
nrow_train<-round(nrow(variables_for_prediction)*0.75,0)
Train_data<-variables_for_prediction[0:nrow_train,]
Test_data<-variables_for_prediction[(nrow_train+1):nrow(variables_for_prediction),]

# Models

# Multiple Linear regression
Linear_regression<- lm(int_rate~.,data=Train_data)
Linear_regression_prediction<-predict(Linear_regression,Test_data)
print(paste0('MAE: ' , mae(Test_data$int_rate,Linear_regression_prediction)))
summary(Linear_regression)
vif(Base_model)

#Regression tree

Regression_tree <- rpart(int_rate ~., data=Train_data, control=rpart.control(cp=.0001))
Regression_tree_prediction<-predict(Regression_tree,Test_data)
print(paste0('MAE: ' , mae(Test_data$int_rate,Regression_tree_prediction)))

# Random Forest

random_forest <- randomForest(int_rate~., data=Train_data, maxnodes = 100, mtry=10, ntree = 200 )
random_forest_prediction<-predict(random_forest,Test_data)
print(paste0('MAE: ' , mae(Test_data$int_rate,random_forest_prediction)))

#AdaBoost
model_adaboost <- gbm(int_rate ~.,data = Train_data,
                      distribution = "gaussian",
                      cv.folds = 10,
                      shrinkage = .01,
                      n.minobsinnode = 10,
                      n.trees = 500)
model_adaboost_prediction<-predict(model_adaboost ,Test_data)
print(paste0('MAE: ' , mae(Test_data$int_rate,model_adaboost_prediction)))

# Multiple linear regression tree

