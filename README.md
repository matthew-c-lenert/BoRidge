# The BoRidge Package
A library of functions for selecting features for and evaluating predictive models

## Package Definitions

This library standardizes data, selects features, and evaluates a model of the selected features corrected for Harrel's optimism. This function does not handle date features, please convert dates into a count of time since a reference date. This program implements the algorithm of Lenert, Matthew C., and Colin G. Walsh. "Balancing Performance and Interpretability: Selecting Features with Bootstrapped Ridge Regression." AMIA Annual Symposium Proceedings. Vol. 2018. American Medical Informatics Association, 2018.


## Package Functions and Parameters

### produce_mode()
dataFrame: input a data frame with NO missing values. Imputation should be done before running data through the BoRidge piepline. Use df.isna().sum() to count missing values.

responseVariableName: The name of column for the outcome (aka response) variable in the dataframe (CASE SENSITIVE)

outputType: the type of output you wish to recieve options are: 'data' to receive data frame of design matrix with BoRidge selected features, 'model' to recieve Scikit Learn fitted model object,  or 'coefficients' to recieve (Logit or Linear) regression coefficent and 95% confidence interval. The 'data' return type returns the dataframe used to fit the final model. The 'coefficients' return type returns a dataframe with the predictor name, the beta coefficient, the low 95% confidence interval, and the high 95% confidence interval. Default is 'coefficients'

interactionVariable: If there is a variable you wish to add an interaction term for with all other predictors provide the name of that column (CASE SENSITIVE). Only supports one column. Default is none

bootstraps: total number of samples with replacement taken during bootstrapping. Default is 100.

standardizeData: center numeric data at 0 and put on standard deviation scale. Split appart categorical data into (number of categories-1) dummy variables. Default is True

epvThreshold: guard rail for the number of observations per predictor. Default is 10

exploreTransforms: automatically adds non-linear forms of predictors such as log, square, square root, cubic, and cubic root in that prioritized order. The system will only add transforms if the number of observations per predictor is above the epvThreshold. Default is False

cStatisticThreshold: The minimum area under the ROC curve (classify) or the explained variance score (regress) required to report final model coefficients and condifence interval in a dataframe. Returns empty dataframe otherwise. Default is 0

brierScoreThreshold: The maximum Brier Score (classify) or mean square error (regress) allowed to report final model coefficients and condifence interval in a dataframe. Returns empty dataframe otherwise. Default is 1

bootstrapPercentage: the percentage of bootstraps a predictor must be found significant in to be included in the final model. Default is 1, accepted range is from [0,1].

cores: the number of threads you wish to use for bootstrapping. Default is 1

correlationThreshold: predictors that are highly correlated with one another can be automatically removed. Set the threshold for the correlation coefficient for predictors to be removed. Default is 1. Range is from [0,1]

typesOfModels: the types of models to evaluate performance on. This parameter requires an array of strings. Default is ['L1','RF','SVM']. L1 = Lasso Regression, RF=Random Forest, and SVM = support vector machine

printProgress: produce verbose output of where the program is in execution. Default is False

errorLogFile: the file where error/warning messages will be appended. Default is errorLog.txt

The output of this pipeline is the model performance characteristics printed to the command line, and the function returns a data frame with all the predictor coefficients and their confidence intervals.


### sampleDataFrame(x, n)

### splitCategoryIntoBinary(modelData,column)

### standardizeAndTransform(modelData,responseVariableName,exploreTransforms,interactionVariable,epvThreshold,errorLogFile)

### findLinearCombinations(features,correlationThreshold,errorLogFile,thisFileName)

### evaluate_model(currentModelData,responseVariableName,modelType,cores,bootstraps,outcomeType,printProgress,errorLogFile)


## Citing this Package

When using this package for research purposes, please cite “Lenert, Matthew C., and Colin G. Walsh. "Balancing Performance and Interpretability: Selecting Features with Bootstrapped Ridge Regression." AMIA Annual Symposium Proceedings. Vol. 2018. American Medical Informatics Association, 2018.”

