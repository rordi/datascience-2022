# ---------------------------------------------------------------------
# Assignment 1 - REGRESSION - TRAIN
# ---------------------------------------------------------------------
#
# Group A2
# Dietrich Rordorf, Marco Lecci, Rizoanun Nasa, Sarah Castratori
#
# This script is used to train a model that we can later use
# for prediction for the regression task.
#
# ---------------------------------------------------------------------
#
#   * Provide at least one working regression model that gives a recommendation
#     to lenders about the height of a suitable interest rate (int_rate) for a
#     given loan applicant, based on information that is available about the applicant.
#   * Compute training error, test error and cross validation error for the published
#     data using the mean squared error (MSE). You may additionally use other appropriate
#     metrics introduced in the lectures. Document your testing strategy, as well as your
#     interpretations of the resulting evaluation scores.
#   * Save your trained model: saveRDS(model, file = "filename") Also save your
#     preprocessed data that you used for model training and testing:
#     write.csv(dataset,"filename.csv")
#
# =====================================================================

# allow for reproducible results
set.seed(1)