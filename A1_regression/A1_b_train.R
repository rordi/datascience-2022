# ---------------------------------------------------------------------
# Assignment 1 - REGRESSION - TRAIN
# ---------------------------------------------------------------------
#
# Group A2
# Dietrich Rordorf, Marco Lecci, Rizoanun Nasa, Sarah Castratori
#
# This script is used to train the final model (winning model) on all data
# which can later be used for predictions on unseen data!
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

# load packages
library("dplyr")
library("scales")
library("stringr")
library("data.table")
library("tidyverse")
library("skimr")
library("tm")
library("reshape")
library("Metrics")
library('xgboost')


# =====================================================================
# LOAD PREPROCESSED DATA
# =====================================================================

df<-fread("./A1_regression/LCdata_preprocessed.csv")

# verify that no error was introduced via CSV encoding/decoding
skim(df)


# =====================================================================
# TRAIN OUR BEST MODEL ON ALL DATA
# =====================================================================

# XG-Boost
#---------------------------------------------------------------------------

reg_xgboost<-function() {
  # xgboost expects a matrix not a dataframe
  labels<-data.matrix(df$int_rate)
  train_data<-data.matrix(df)
  train_data<-train_data[,colnames(train_data)!="int_rate"] # remove int_rate from train data

  # train winning model over all data with tuned parameters
  model_xgboost <- xgboost(
    data = train_data,
    label = labels,
    eta = 0.15,
    max.depth = 8,
    nthread = 2,
    nrounds = 90,
    lambda=0.2,
    objective = "reg:squarederror",
    eval_metric='rmse',
    verbose = 1
  )
  
  yhat<-predict(model_xgboost, train_data)
  mae<-mae(labels, yhat)
  mse<-mean((yhat-labels)^2)
  print(paste0('MAE (model_xgboost, all data): ' , mae))
  print(paste0('MSE (model_xgboost, all data): ' , mse))
  
  return (model_xgboost)
}

winning_model = reg_xgboost()

# [1] "MAE (model_xgboost, all data): 2.49276621735121"
# [1] "MSE (model_xgboost, all data): 10.0328253111851"

saveRDS(winning_model, file = "./A1_regression/winning_model.rda")
