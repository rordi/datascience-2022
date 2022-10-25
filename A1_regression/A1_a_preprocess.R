# ---------------------------------------------------------------------
# Assignment 1 - REGRESSION - PREPROCESS
# ---------------------------------------------------------------------
#
# Group A2
# Dietrich Rordorf, Marco Lecci, Rizoanun Nasa, Sarah Castratori
#
# This script is used to preprocess the raw data from CSV to useable
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
attributes(df)
