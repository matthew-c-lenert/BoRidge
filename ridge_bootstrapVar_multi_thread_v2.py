# Required dependancies: datetime, decimal, joblib, math, numpy, os, pandas, re, scipy, shutil, sklearn, sys, tempfile, time, traceback
#
# Version from 07-18-2018

import pandas as pd
import numpy as np
import re
import sys
import random
import sklearn.metrics as sklearn
import decimal
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
import scipy.stats as stats
import math
import time
from joblib import Parallel, delayed
from joblib import load, dump
import tempfile
import shutil
import os
from datetime import datetime, timedelta
import traceback


def about():
	print("dependancies()\n\nhelp()\n\nCIboundLogit(coefficient,stdError,sampleSize,isUpperBound=True,alpha=0.05)\n\nCIboundLinear(coefficient,stdError,sampleSize,isUpperBound=True,alpha=0.05)\n\nsampleDataFrame(x, n)\n\nsplitCategoryIntoBinary(modelData,column)\n\nstandardizeAndTransform(modelData,responseVariableName,exploreTransforms,interactionVariable,epvThreshold,errorLogFile)\n\nfindLinearCombinations(features,correlationThreshold,errorLogFile,thisFileName)\n\nevaluate_model(currentModelData,responseVariableName,modelType,cores,bootstraps,outcomeType,printProgress,errorLogFile)\n\nproduce_model(dataFrame,responseVariableName,outputType='coefficents',interactionVariable='',bootstraps=100,standardizeData=True,epvThreshold=10,exploreTransforms=False,cStatisticThreshold=0,brierScoreThreshold=1,bootstrapPercentage=1,cores=1,correlationThreshold=1,typesOfModels=['L1','RF','SVM'],printProgress=False,errorLogFile='errorLog.txt')")


def help():
	print("This library standardizes data, selects features, and evaluates a model of the selected features corrected for Harrel's optimism. This function does not handle date features, please convert dates into a count of time since a reference date. The following is a description of the parameters used in the produce_model() function:\n"+"1) dataFrame: input a data frame with NO missing values. Imputation should be done before running data through the BoRidge piepline. Use df.isna().sum() to count missing values.\n\n"+"2) responseVariableName: The name of column for the outcome (aka response) variable in the dataframe (CASE SENSITIVE)\n\n"+"3) outputType: the type of output you wish to recieve options are: 'data' to receive data frame of design matrix with BoRidge selected features, 'model' to recieve Scikit Learn fitted model object,  or 'coefficients' to recieve (Logit or Linear) regression coefficent and 95% confidence interval. The 'data' return type returns the dataframe used to fit the final model. The 'coefficients' return type returns a dataframe with the predictor name, the beta coefficient, the low 95% confidence interval, and the high 95% confidence interval. Default is 'coefficients'.\n\n"+"4) interactionVariable: If there is a variable you wish to add an interaction term for with all other predictors provide the name of that column (CASE SENSITIVE). Only supports one column. Default is none\n\n"+"5) bootstraps: total number of samples with replacement taken during bootstrapping. Default is 100.\n\n"+"6)standardizeData: center numeric data at 0 and put on standard deviation scale. Split appart categorical data into (number of categories-1) dummy variables. Default is True\n\n"+"7) epvThreshold: guard rail for the number of observations per predictor. Default is 10.\n\n"+"8) exploreTransforms: automatically adds non-linear forms of predictors such as log, square, square root, cubic, and cubic root in that prioritized order. The system will only add transforms if the number of observations per predictor is above the epvThreshold. Default is False\n\n"+"9) cStatisticThreshold: The minimum area under the ROC curve (classify) or the explained variance score (regress) required to report final model coefficients and condifence interval in a dataframe. Returns empty dataframe otherwise. Default is 0\n\n"+"10) brierScoreThreshold: The maximum Brier Score (classify) or mean square error (regress) allowed to report final model coefficients and condifence interval in a dataframe. Returns empty dataframe otherwise. Default is 1\n\n"+"11) bootstrapPercentage: the percentage of bootstraps a predictor must be found significant in to be included in the final model. Default is 1, accepted range is from [0,1].\n\n"+"12) cores: the number of threads you wish to use for bootstrapping. Default is 1.\n\n"+"13) correlationThreshold: predictors that are highly correlated with one another can be automatically removed. Set the threshold for the correlation coefficient for predictors to be removed. Default is 1. Range is from [0,1].\n\n"+"14) typesOfModels: the types of models to evaluate performance on. This parameter requires an array of strings. Default is ['L1','RF','SVM']. L1 = Lasso Regression, RF=Random Forest, and SVM = support vector machine\n\n"+"15) printProgress: produce verbose output of where the program is in execution. Default is False.\n\n"+"16) errorLogFile: the file where error/warning messages will be appended. Default is errorLog.txt\n\n"+"The output of this pipeline is the model performance characteristics printed to the command line, and the function returns a data frame with all the predictor coefficients and their confidence intervals.")


def dependancies():
	print("datetime\ndecimal\njoblib\nmath\nnumpy\nos\npandas\nre\nscipy\nshutil\nsklearn\nsys\ntempfile\ntime\ntraceback")

def printError(errorMessage,systemError,errorLine,errorFile):
	try:
		errorFileIO=""
		errorFileIO = open(errorFile,"a")
		traceBackMessage=""
		if errorLine != "":
			traceBackMessage=str(traceback.extract_tb(errorLine, limit=1)[-1][1])
		if systemError != "":
			errorFileIO.write(datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S') + ": " + errorMessage + " -- " + str(systemError) + " at line " + traceBackMessage + "\n")
			errorFileIO.close()
		else:
			errorFileIO.write(datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S') + ": " + errorMessage + "\n")
			errorFileIO.close()
	except:
		print(datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S') + ": " + errorMessage + " -- " + str(systemError) + " at line " + traceBackMessage + "\n")


def CIboundLogit(coefficient,stdError,sampleSize,isUpperBound=True,alpha=0.05):
	T=0

	if isUpperBound:
		T = stats.t.ppf(1-(alpha/2),sampleSize-1)

	else:
		T = stats.t.ppf(alpha/2,sampleSize-1)

	value=coefficient+(T*stdError/(sampleSize**(0.5)))

	if value>500:
		return(sys.float_info.max)

	return(math.exp(value))


def CIboundLinear(coefficient,stdError,sampleSize,isUpperBound=True,alpha=0.05):
	T=0

	if isUpperBound:
		T = stats.t.ppf(1-(alpha/2),sampleSize-1)

	else:
		T = stats.t.ppf(alpha/2,sampleSize-1)

	value=coefficient+(T*stdError/(sampleSize**(0.5)))

	return(value)


def drange(x, y, jump):
	returnArray=[]
	while x <= y:
		returnArray.append(float(x))
		x += decimal.Decimal(jump)
	return(returnArray)


def regularizeColumn(column):
	mean=column.mean()
	standardDeviation=column.std()
	return(column-mean)/standardDeviation


def sample_wr(population, k):
	n = len(population)
	_random, _int = random.random, int
	result = [None] * k
	for i in range(k):
		j = _int(_random() * n)
		result[i] = population[j]
	return(result)


def sampleDataFrame(x, n):
	return(x.iloc[sample_wr(x.index, n)])


def bootstrapPivotCI(bootstrapData,responseVariableName,bootstrapIterations,pairedInterventionDictionaryReplicates,secondLevelCoefficients,best_Alpha,correlationThreshold,outcomeType,printProgress):
	thisFileName="ridge_bootstrapVar_multi_thread"
	for replicateSample in bootstrapIterations:
		replicateData=sampleDataFrame(bootstrapData,bootstrapData.shape[0])
		response=replicateData[responseVariableName]
		safetyCount=1
		while response.nunique()<2:
			replicateData=sampleDataFrame(bootstrapData,bootstrapData.shape[0])
			response=replicateData[responseVariableName]
			if safetyCount>100:
				e = ""
				errorLine = ""
				printError(thisFileName+".py: Error Log: Safety count reached in 2nd level bootstrap. Cannot find more than 1 value after 100 resamples", e, errorLine, errorLogFile)
				break
			safetyCount+=1
		dropList=[]
		columnNames=list(replicateData.columns.values)
		features=replicateData.drop(responseVariableName,1)

		for column in columnNames:
			pairedInterventionDictionaryReplicates[column]=[]
			if column!=responseVariableName:
					if replicateData[column].nunique() < 2:
						dropList.append(column)

			#Check for randomly induced linear combinations


		#Remove columns with only one unique value and update columnNames
		replicateData=replicateData.drop(dropList,1)
		columnNames=list(replicateData.columns.values)

		features=replicateData.drop(responseVariableName,1)
		featureNames=features.columns.values

		model={}

		try:
			if outcomeType=="regress":
				model = Ridge(fit_intercept=True, alpha = best_Alpha,  tol=0.01)

			elif outcomeType=="classify":
				model = SGDClassifier(loss="log", penalty="l2", fit_intercept=True, alpha=best_Alpha,  tol=0.01, n_jobs=1)
				#LogisticRegression(penalty="l2", fit_intercept=True, solver='saga', C= best_Alpha,  tol=0.01, n_jobs=1)

			model.fit(features,response)
		except:
			#features.to_csv("troubleshootfit_bootstrapVar.csv")
			e = sys.exc_info()[0:1]
			errorLine = sys.exc_info()[-1]
			printError(thisFileName+".py: "+"Error fitting model in bootstrap variance loop, check for NaN's, Null's, or inf values: "+responseVariableName+" ~ " + str(list(features)), e, errorLine, errorLogFile)
			return(emptyFrame)
		#Store coefficients from 2nd level bootstrap model
		replicateCoefficients=list()
		if outcomeType=="regress":
			replicateCoefficients=model.coef_.tolist()
		else:
			replicateCoefficients=model.coef_[0].tolist()
		for regressorIndex in list(range(len(replicateCoefficients))):
			secondLevelCoefficients.at[replicateSample,featureNames[regressorIndex]]=replicateCoefficients[regressorIndex]
	return(secondLevelCoefficients)


def boridgeFeatureSelect(sample, bootstrapIterations, sharedData, sharedColNames, outputMatrix, responseVariableName, correlationThreshold, printProgress, errorLogFile, outcomeType):
	emptyFrame=pd.DataFrame(index=np.arange(0, 1), columns=('empty','frame'))
	thisFileName="ridge_bootstrapVar_multi_thread"
	modelData=pd.DataFrame(sharedData)
	bootstrapData=sampleDataFrame(modelData,modelData.shape[0])
	bootstrapData.columns=sharedColNames
	response=bootstrapData[responseVariableName]
	safetyCount=1
	while response.nunique()<2:
		bootstrapData=sampleDataFrame(modelData,modelData.shape[0])
		response=bootstrapData[responseVariableName]
		if safetyCount>100:
			e = ""
			errorLine = ""
			printError(thisFileName+".py: Error Log: Safety count reached. Cannot find more than 1 value after 100 resamples", e, errorLine, errorLogFile)
			break
		safetyCount+=1


	outOfBagData=modelData[~modelData.index.isin(bootstrapData.index)]
	outOfBagData.columns=sharedColNames
	aBigNumber=sys.float_info.max
	best_LogLoss=aBigNumber
	best_Alpha=0
	best_l2_ratio=1
	best_Model={}
	pairedInterventionDictionary={}
	pairedInterventionDictionaryReplicates={}
	dropList=[]
	columnNames=list(bootstrapData.columns.values)
	features=bootstrapData.drop(responseVariableName,1)
	outputArray=[0]*modelData.shape[1]
	reverseNameIndex={}

	#Adjust Alpha tunning precision based on data size
	dataSize=modelData.shape[0]*modelData.shape[1]


	if dataSize>100000000:
		dataSize=100000000
	numSteps=(100000000//dataSize)+1
	if numSteps>75:
		numSteps=75
	if numSteps<5:
		numSteps=5
	tunningAlphaValues = np.random.randint(1, 100, numSteps)
	l2RatioValues = np.random.randint(0,1,numSteps)

	for index in range(0,len(columnNames)):
		if columnNames[index]!=responseVariableName:
			reverseNameIndex[columnNames[index]]=index

	if printProgress:
		if sample % 10 == 0:
			print("BoRidge sample "+str(sample))

	for column in columnNames:
		pairedInterventionDictionary[column]=[]
		if column!=responseVariableName:
				if bootstrapData[column].nunique() < 2:
					dropList.append(column)

		#Check for randomly indueced linear combinations


	#Remove Columns with only one unique value and update columnNames
	bootstrapData=bootstrapData.drop(dropList,1)
	outOfBagData=outOfBagData.drop(dropList,1)
	outOfBagFeatures=outOfBagData.drop(responseVariableName,1)
	columnNames=list(bootstrapData.columns.values)
	outOfBagResponse=outOfBagData[responseVariableName]
	features=bootstrapData.drop(responseVariableName,1)
	featureNames=features.columns.values

	for tunningAlpha in tunningAlphaValues:
		model={}

		try:
			if outcomeType=="regress":
				model = Ridge(fit_intercept=True, alpha = tunningAlpha,  tol=0.01)

			elif outcomeType=="classify":
				model = SGDClassifier(loss="log", penalty="l2", fit_intercept=True, alpha=tunningAlpha,  tol=0.01, n_jobs=1)
				#LogisticRegression(penalty="l2", fit_intercept=True, solver='saga', C = tunningAlpha,  tol=0.01, n_jobs=1)

			model.fit(features,response)
		except:
			#features.to_csv("troubleshootfit.csv")
			e = sys.exc_info()[0:1]
			errorLine = sys.exc_info()[-1]
			printError(thisFileName+".py: "+"Error in model fitting, check for NaN's, Null's, or inf values: "+responseVariableName+" ~ " + str(list(features)), e, errorLine, errorLogFile)
			return(emptyFrame)

		try:
			predictedResponse=model.predict(outOfBagFeatures)

			log_loss=aBigNumber
			if outcomeType=="regress":
				log_loss=sklearn.mean_squared_error(outOfBagResponse,predictedResponse)
			elif outcomeType=="classify":
				log_loss=sklearn.log_loss(outOfBagResponse,predictedResponse)

			if best_LogLoss==aBigNumber:
				best_LogLoss = np.absolute(log_loss)
				best_Model = model
				best_Alpha = tunningAlpha
			elif np.absolute(log_loss) < best_LogLoss:
				best_LogLoss = np.absolute(log_loss)
				best_Model = model
				best_Alpha = tunningAlpha

		except:
			e = sys.exc_info()[0:1]
			errorLine = sys.exc_info()[-1]
			printError(thisFileName+".py: Error in model assesment used to tune regularization penalty parameter alpha", e, errorLine, errorLogFile)
			return(emptyFrame)
	coefficients=list()
	if outcomeType=="regress":
		coefficients=model.coef_.tolist()
	else:
		coefficients=model.coef_[0].tolist()
	secondLevelCoefficients=pd.DataFrame(0.0, index=np.arange(len(bootstrapIterations)), columns=list(featureNames))

	try:
		##### Calculate Bootstrap Pivot Variance ######
		#Pivot is Beta hat_n - Beta
		secondLevelCoefficients = bootstrapPivotCI(bootstrapData,responseVariableName,bootstrapIterations,pairedInterventionDictionaryReplicates,secondLevelCoefficients,best_Alpha,correlationThreshold,outcomeType,printProgress)

		for regressorIndex in list(range(len(coefficients))):
			#if log odds of lower bound (2*Beta - Beta'_1-alpha/2) is greater than 0, result is significant
			if 0 < (2*coefficients[regressorIndex]-secondLevelCoefficients[featureNames[regressorIndex]].quantile(0.975)):
				outputArray[reverseNameIndex[featureNames[regressorIndex]]]=1
				if len(pairedInterventionDictionary[featureNames[regressorIndex]])>0:
					for pairedRegressor in pairedInterventionDictionary[featureNames[regressorIndex]]:
						outputArray[reverseNameIndex[pairedRegressor]]=1
			#if log odds of upper bound (2*Beta - Beta'_alpha/2) is less than 0, result is significant
			elif 0 > (2*coefficients[regressorIndex]-secondLevelCoefficients[featureNames[regressorIndex]].quantile(0.025)):
				outputArray[reverseNameIndex[featureNames[regressorIndex]]]=1
				if len(pairedInterventionDictionary[featureNames[regressorIndex]])>0:
					for pairedRegressor in pairedInterventionDictionary[featureNames[regressorIndex]]:
						outputArray[reverseNameIndex[pairedRegressor]]=1

	except:
		e = sys.exc_info()[0:1]
		errorLine = sys.exc_info()[-1]
		printError(thisFileName+".py: Error calculating bootstrap variance in BoRidge loop", e, errorLine, errorLogFile)
		return(emptyFrame)

	# Write to Shared Memory - use sample # to prevent race conditions
	outputMatrix[sample]=outputArray


def valCheck(cell,val):
	if cell != cell:
		return(np.nan)
	else:
		return(int(cell==val))


def splitCategoryIntoBinary(modelData,column):
	skipLast=False
	count=0
	uniqueValues=modelData[column].unique()
	refString=""
	stopCount=0
	if uniqueValues[len(uniqueValues)-1]!=uniqueValues[len(uniqueValues)-1]:
		refString=str(uniqueValues[len(uniqueValues)-2])
		stopCount=len(uniqueValues)-2
	else:
		refString=str(uniqueValues[len(uniqueValues)-1])
		stopCount=len(uniqueValues)-1
	for value in uniqueValues:
		if skipLast:
			skipLast=True
		elif not value != value:
			valueString=column+"_is_"+str(value)+"_with_ref_"+refString
			modelData.loc[:,valueString]=modelData[column].apply(lambda x: valCheck(x,value))
		count+=1
		if count==stopCount:
			skipLast=True

	return(modelData.drop(column,1))


def standardizeAndTransform(modelData,responseVariableName,exploreTransforms,interactionVariable,epvThreshold,errorLogFile):
	# Factor variables should be strings (objects) and not integers or floats
	originalModelData = modelData.copy()
	groupColumnsOnDataType=modelData.columns.to_series().groupby(modelData.dtypes).groups
	groupColumnsOnDataType={k.name: v for k, v in groupColumnsOnDataType.items()}
	eventCount=modelData.shape[0]
	if responseVariableName!="":
		if modelData[responseVariableName].nunique()<3:
			eventCount=modelData[responseVariableName].sum()
	thisFileName="ridge_bootstrapVar_multi_thread"
	for dataType in groupColumnsOnDataType.keys():
		if (dataType!='object') & (dataType!='category'):
			for column in groupColumnsOnDataType[dataType]:
				# TODO Parallelize This
				if (column!=responseVariableName):
					if modelData[column].nunique()<10:
						if modelData[column].nunique()==2:
							if modelData[column].unique().sum()!=1:
								modelData=splitCategoryIntoBinary(modelData,column)
						else:
							modelData=splitCategoryIntoBinary(modelData,column)
					elif exploreTransforms:
						minimumValue=modelData[column].min()
						step=0
						if epvThreshold==0:
							step=12
						else:
							step = int(eventCount/epvThreshold) - modelData.shape[1]
						lower = -3
						upper = 3
						if step < 2:
							step=1
							upper=1
							lower=1
						elif step < 3:
							step=1
							upper=1
							lower=-1
						elif step < 5:
							step=1
							upper=2
							lower=-2
						elif step < 7:
							step=1
						elif step < 12:
							step=0.5
							upper=2
							lower=-2
						else:
							step=0.5
							upper=3
							lower=-3

						for transform in drange(lower,upper,step):
							if transform !=0:
								if transform != 1:
									if (transform == 2) | (minimumValue > 0) | (transform == 3):
										transformString1=column+"_^_"+str(transform)
										modelData[transformString1]=modelData[column].astype(float)**transform
										modelData[transformString1]=regularizeColumn(modelData[transformString1])
							elif originalModelData[column].min() > 0:
								transformString1=column+"_^_ln transform"
								modelData[transformString1]=np.log(originalModelData[column])
								modelData[transformString1]=regularizeColumn(modelData[transformString1])
						modelData[column]=regularizeColumn(modelData[column])
					else:
						modelData[column]=regularizeColumn(modelData[column])
		if (dataType=='object') | (dataType=='category'):
			for column in groupColumnsOnDataType[dataType]:
				if (column!=responseVariableName):
					if modelData[column].nunique()<10:
						modelData=splitCategoryIntoBinary(modelData,column)
					else:
						modelData=modelData.drop(column,1)
						e = ""
						errorLine = ""
						printError(thisFileName+".py: Audit Log: Cannot infer type of variable " + column + ". Variable dropped, please format as a numeric or a category", e, errorLine, errorLogFile)
				else:
					holder=modelData[responseVariableName].factorize()
					modelData[responseVariableName]=holder[0]

	if interactionVariable!="":
		for column in modelData.columns.values:
			if (column!=responseVariableName):
				interactionString=column+"*"+interactionVariable
				modelData[interactionString]=modelData[column].multiply(modelData[interactionVariable])
	return(modelData)


def findLinearCombinations(features,correlationThreshold,errorLogFile,thisFileName):
	dropList=[]
	columnNames=features.columns.values
	correlationMatrix=features.corr()
	for rowIndex in range(correlationMatrix.shape[0]):
		for columnIndex in range(rowIndex):
			if rowIndex != columnIndex:
				if np.absolute(correlationMatrix.iloc[rowIndex, columnIndex]) >= correlationThreshold:
					dropList.append(columnNames[columnIndex])
					printError(thisFileName+".py: Audit Log: Predictor: "+columnNames[columnIndex]+" dropped from consideration. The column is a linear combination of "+columnNames[rowIndex], "", "", errorLogFile)
	return(dropList)


def evaluate_model(currentModelData,responseVariableName,modelType,cores,bootstraps,outcomeType,printProgress,errorLogFile):
	#Adjust Alpha tunning precision based on data size
	features=currentModelData.drop(responseVariableName,1)
	response=currentModelData[responseVariableName]
	dataSize=currentModelData.shape[0]*currentModelData.shape[1]
	thisFileName="ridge_bootstrapVar_multi_thread"

	if dataSize>100000000:
		dataSize=100000000
	numSteps=(100000000//dataSize)+1
	if numSteps>25:
		numSteps=25
	if numSteps<5:
		numSteps=5
	tunningAlphaValues = []
	tunningBetaValues = []
	if modelType=="L2":
		tunningAlphaValues=np.random.randint(1, 100, numSteps)
		tunningBetaValues=[1]
	elif modelType=="SVM":
		standardDev=np.std(response.values)
		tunningAlphaValues=np.random.randint(1, 100, numSteps)
		if outcomeType=="regress":
			tunningBetaValues=np.linspace(standardDev/100.0, standardDev/10.0, numSteps)
		elif outcomeType=="classify":
			tunningBetaValues=['linear', 'poly', 'rbf', 'sigmoid']
	elif modelType=="RF":
		tunningAlphaValues=np.random.randint(10, 25, numSteps)
		tunningBetaValues=np.random.randint(10, 100, numSteps)

	l2RatioValues = np.random.randint(0,1,numSteps)
	aBigNumber=sys.float_info.max
	apparent_model={}
	best_LogLoss=aBigNumber
	best_Alpha=0
	best_Beta=0
	best_Model={}
	for tunningAlpha in tunningAlphaValues:
		for tunningBeta in tunningBetaValues:
			if True:
				if outcomeType=="regress":
					if modelType=="L2":
						apparent_model = Ridge(fit_intercept=True, alpha = tunningAlpha,  tol=0.01)
					elif modelType=="SVM":
						apparent_model = SVR(C = tunningAlpha, epsilon=tunningBeta,  tol=0.01)
					elif modelType=="RF":
						apparent_model = RandomForestRegressor(n_estimators=int(tunningAlpha), max_depth=int(tunningBeta), n_jobs=cores)

				elif outcomeType=="classify":
					if modelType=="L2":
						apparent_model = SGDClassifier(loss="log", penalty="l2", fit_intercept=True, alpha=tunningAlpha,  tol=0.01, n_jobs=cores)
						#LogisticRegression(penalty="l2", fit_intercept=True, solver='saga', C = tunningAlpha,  tol=0.01, n_jobs=cores)
					elif modelType=="SVM":
						apparent_model = SVC(C = tunningAlpha, kernel=tunningBeta,  tol=0.01)
					elif modelType=="RF":
						apparent_model = RandomForestClassifier(n_estimators=int(tunningAlpha), max_depth=int(tunningBeta), n_jobs=cores)

				apparent_model.fit(features,response)
			else:
				#features.to_csv("troubleshootBestFit.csv")
				e = sys.exc_info()[0:1]
				errorLine = sys.exc_info()[-1]
				printError(thisFileName+".py: "+"Error fitting best model, check for NaN's, Null's, or inf values: "+responseVariableName+" ~ " + str(list(features)), e, errorLine, errorLogFile)
				return([])

			try:
				apprentPredictions=apparent_model.predict(features)
				log_loss=aBigNumber
				if outcomeType=="regress":
					log_loss=sklearn.mean_squared_error(response,apprentPredictions)
				elif outcomeType=="classify":
					log_loss=sklearn.log_loss(response,apprentPredictions)
				if best_LogLoss==aBigNumber:
					best_LogLoss = np.absolute(log_loss)
					best_Alpha = tunningAlpha
					best_Beta = tunningBeta
				elif np.absolute(log_loss) < best_LogLoss:
					best_LogLoss = np.absolute(log_loss)
					best_Alpha = tunningAlpha
					best_Beta = tunningBeta
			except:
				e = sys.exc_info()[0:1]
				errorLine = sys.exc_info()[-1]
				printError(thisFileName+".py: Error in best model assesment used to tune regularization penalty parameter alpha", e, errorLine, errorLogFile)
				return([])

	apparent_model={}
	try:
		if outcomeType=="regress":
			if modelType=="L2":
				apparent_model = Ridge(fit_intercept=True, alpha = best_Alpha,  tol=0.01)
			elif modelType=="SVM":
				apparent_model = SVR(C = best_Alpha, epsilon=best_Beta,  tol=0.01)
			elif modelType=="RF":
				apparent_model = RandomForestRegressor(n_estimators=int(best_Alpha), max_depth=int(best_Beta), n_jobs=cores)

		elif outcomeType=="classify":
			if modelType=="L2":
				apparent_model = SGDClassifier(loss="log", penalty="l2", fit_intercept=True, alpha=best_Alpha,  tol=0.01, n_jobs=cores)
				#LogisticRegression(penalty="l2", fit_intercept=True, solver='saga', C = best_Alpha,  tol=0.01, n_jobs=cores)
			elif modelType=="SVM":
				apparent_model = SVC(C = best_Alpha, kernel=best_Beta,  tol=0.01)
			elif modelType=="RF":
				apparent_model = RandomForestClassifier(n_estimators=int(best_Alpha), max_depth=int(best_Beta), n_jobs=cores)


		apparent_model.fit(features,response)
	except:
		#features.to_csv("troubleshootBestFit.csv")
		e = sys.exc_info()[0:1]
		errorLine = sys.exc_info()[-1]
		printError(thisFileName+".py: "+"Error fitting best model with best alpha, check for NaN's, Null's, or inf values: "+responseVariableName+" ~ " + str(list(features)), e, errorLine, errorLogFile)
		return([])



	apprentPredictions=apparent_model.predict(features)
	apparentBrier=aBigNumber
	apparentMCC=0
	apparentC=0

	if outcomeType=="regress":
		apparentBrier = sklearn.mean_squared_error(response,apprentPredictions)
		apparentMCC = (sklearn.median_absolute_error(response,apprentPredictions)+1)**(-1)
		apparentC = sklearn.explained_variance_score(response,apprentPredictions)

	elif outcomeType=="classify":
		apparentBrier = sklearn.brier_score_loss(response,apprentPredictions)
		apparentMCC = sklearn.matthews_corrcoef(response,apprentPredictions)
		apparentC = sklearn.roc_auc_score(response,apprentPredictions)

	brierOptimism=[]
	mccOptimism=[]
	cOptimism=[]

	bootstrapIterations=range(0,bootstraps)

	for sample in bootstrapIterations:

		if printProgress:
			if sample % 10 == 0:
				print("Evaluation sample "+str(sample))

		bootstrapData = sampleDataFrame(currentModelData,currentModelData.shape[0])
		bootstrapFeatures = bootstrapData.drop(responseVariableName,1)
		bootstrapResponse = bootstrapData[responseVariableName]

		safetyCount=1
		while bootstrapResponse.nunique()<2:
			bootstrapData=sampleDataFrame(currentModelData,currentModelData.shape[0])
			bootstrapResponse = bootstrapData[responseVariableName]
			if safetyCount>100:
				e = ""
				errorLine = ""
				printError(thisFileName+".py: Error Log: Safety count reached in evaluation bootstrap. Cannot find more than 1 value after 100 resamples", e, errorLine, errorLogFile)
				break
			safetyCount+=1

		bootstrap_model = {}
		if outcomeType=="regress":
			if modelType=="L2":
				bootstrap_model = Ridge(fit_intercept=True, alpha = best_Alpha, tol=0.01)
			elif modelType=="SVM":
				bootstrap_model = SVR(C = best_Alpha, epsilon=best_Beta, tol=0.01)
			elif modelType=="RF":
				bootstrap_model = RandomForestRegressor(n_estimators=int(best_Alpha), max_depth=int(best_Beta), n_jobs=cores)

		elif outcomeType=="classify":
			if modelType=="L2":
				bootstrap_model = SGDClassifier(loss="log", penalty="l2", fit_intercept=True, alpha=best_Alpha,  tol=0.01, n_jobs=cores)
				#LogisticRegression(penalty="l2", fit_intercept=True, solver='saga', C = best_Alpha,  tol=0.01, n_jobs=cores)
			elif modelType=="SVM":
				bootstrap_model = SVC(C = best_Alpha, kernel=best_Beta, tol=0.01)
			elif modelType=="RF":
				bootstrap_model = RandomForestClassifier(n_estimators=int(best_Alpha), max_depth=int(best_Beta), n_jobs=cores)

		bootstrap_model.fit(bootstrapFeatures,bootstrapResponse)

		bootstrapPredictions = bootstrap_model.predict(bootstrapFeatures)
		originalPredictions = bootstrap_model.predict(features)
		if outcomeType=="regress":
			bootstrapBrier = sklearn.mean_squared_error(bootstrapResponse,bootstrapPredictions)
			bootstrapMCC = (sklearn.median_absolute_error(bootstrapResponse,bootstrapPredictions)+1)**(-1)
			bootstrapC = sklearn.explained_variance_score(bootstrapResponse,bootstrapPredictions)

			originalBrier = sklearn.mean_squared_error(response,originalPredictions)
			originalMCC = (sklearn.median_absolute_error(response,originalPredictions)+1)**(-1)
			originalC = sklearn.explained_variance_score(response,originalPredictions)

			brierOptimism.append(bootstrapBrier-originalBrier)
			mccOptimism.append(bootstrapMCC-originalMCC)
			cOptimism.append(bootstrapC-originalC)

		elif outcomeType=="classify":
			bootstrapBrier = sklearn.brier_score_loss(bootstrapResponse,bootstrapPredictions)
			bootstrapMCC = sklearn.matthews_corrcoef(bootstrapResponse,bootstrapPredictions)
			bootstrapC = sklearn.roc_auc_score(bootstrapResponse,bootstrapPredictions)

			originalBrier = sklearn.brier_score_loss(response,originalPredictions)
			originalMCC = sklearn.matthews_corrcoef(response,originalPredictions)
			originalC = sklearn.roc_auc_score(response,originalPredictions)

			brierOptimism.append(bootstrapBrier-originalBrier)
			mccOptimism.append(bootstrapMCC-originalMCC)
			cOptimism.append(bootstrapC-originalC)

	# Adjust for optimism
	brierScore = apparentBrier-(sum(brierOptimism)/len(brierOptimism))
	mcc = apparentMCC-(sum(mccOptimism)/len(mccOptimism))
	cStatistic = apparentC -(sum(cOptimism)/len(cOptimism))
	return([apparent_model,cStatistic,brierScore,mcc])

def produce_model(dataFrame,responseVariableName,outputType="coefficents",interactionVariable="",bootstraps=100,standardizeData=True,epvThreshold=10,exploreTransforms=False,cStatisticThreshold=0,brierScoreThreshold=1,bootstrapPercentage=1,cores=1,correlationThreshold=1,typesOfModels=["L2","SVM","RF"],printProgress=False,errorLogFile="errorLog.txt"):
	totalStart=time.time()
	outcomeType=""
	modelData=[]
	originalModelData=[]
	features=[]
	response=[]
	warnings=False
	originalColumnNames=[]
	eventCount=0
	aBigNumber=10000000
	thisFileName="ridge_bootstrapVar_multi_thread"
	outcomeType=outcomeType.lower()
	outputType=outputType.lower()

	if epvThreshold<0:
		epvThreshold=0

	if bootstrapPercentage > 1:
		bootstrapPercentage=1
	elif bootstrapPercentage < 0:
		bootstrapPercentage=0

	if (correlationThreshold>1) | (correlationThreshold<1):
		correlationThreshold=1
	elif correlationThreshold < 0:
		correlationThreshold=correlationThreshold*-1

	emptyFrame=pd.DataFrame(index=np.arange(0, 1), columns=('empty','frame'))
	modelData=dataFrame

	try:
		originalColumnNames=list(modelData.columns.values)
	except:
		e = sys.exc_info()[0:1]
		errorLine = sys.exc_info()[-1]
		printError(thisFileName+".py: "+"Error reading model data, make sure it is a Pandas dataframe", e, errorLine, errorLogFile)
		return(emptyFrame)

	eventCount=modelData.shape[0]

	#Filter out Variables with Missing Data
	index=0
	nullsInData=modelData.isnull().sum()
	nullKeys=nullsInData.keys()
	dropList=[]
	for column in nullsInData:
		if column > 0:
			columnName=nullKeys[index]
			dropList.append(columnName)
			e = ""
			errorLine = ""
			printError(thisFileName+".py: Regressor, " + columnName + " was dropped because it contains null values, please impute values in beforehand", e, errorLine, errorLogFile)
			index+=1
	modelData=modelData.drop(dropList,1)


	#Sanity checking for outcomeType
	if outcomeType=="classify":
		if modelData[responseVariableName].nunique()>2:
			e = ""
			errorLine = ""
			printError(thisFileName+".py: Outcome variable, " + responseVariableName + " is not a binary category, please adjust outcomeType or correct responseVariable data.", e, errorLine, errorLogFile)
			return(emptyFrame)
	elif outcomeType=="regress":
		if modelData[responseVariableName].nunique()<10:
			e = ""
			errorLine = ""
			printError(thisFileName+".py: Outcome variable, " + responseVariableName + " is not continuous (fewer than 10 unique values), please adjust outcomeType or correct responseVariable data.", e, errorLine, errorLogFile)
			return(emptyFrame)
	else:
		if modelData[responseVariableName].nunique()<3:
			outcomeType="classify"
		elif modelData[responseVariableName].nunique()>9:
			outcomeType="regress"
		else:
			e = ""
			errorLine = ""
			printError(thisFileName+".py: Outcome variable, " + responseVariableName + " cannot infer outcomeType. Please modify response variable to be binary category or continuous", e, errorLine, errorLogFile)
			return(emptyFrame)
	if standardizeData:
		if printProgress:
			print("Regularizing Data")

		try:
			standardizationTime=time.time()
			modelData=standardizeAndTransform(modelData,responseVariableName,exploreTransforms,interactionVariable,epvThreshold,errorLogFile)
		except:
			e = sys.exc_info()[0:1]
			errorLine = sys.exc_info()[-1]
			printError(thisFileName+".py: "+"Error during variable transformation and regularization. Make sure there are no date/date-time variables, and categories are coded as strings.", e, errorLine, errorLogFile)
			return(emptyFrame)
		if printProgress:
			print("Regularizing Time %s seconds" % (time.time() - standardizationTime))

	features=modelData.drop(responseVariableName,1)

	# Check for linear combinations
	colinearStart=time.time()
	dropList=findLinearCombinations(features,correlationThreshold,errorLogFile,thisFileName)
	if printProgress:
		print("Collinear Time %s seconds" % (time.time() - colinearStart))
	# Remove linear combinations
	modelData=modelData.drop(dropList,1)
	features=features.drop(dropList,1)
	feature_dictionary={}

	for regressor in list(features.columns.values):
		feature_dictionary[regressor]=0


	columnNames=modelData.columns.values
	bootstrapIterations=range(0,bootstraps)
	if printProgress:
		print("Begin BoRidge")
	boridgeTime=time.time()

	# Move data to shared memory map
	temp_folder = tempfile.mkdtemp(dir=os.path.dirname(os.path.realpath(__file__)))
	dataFile = os.path.join(temp_folder, 'L2_modelData.mmap')
	nameFile = os.path.join(temp_folder, 'L2_colNames')
	variableDictFile = os.path.join(temp_folder, 'L2_variableCounts')

	if os.path.exists(dataFile):
		os.unlink(dataFile)

	dump(modelData.as_matrix(), dataFile)

	if os.path.exists(nameFile):
		os.unlink(nameFile)

	dump(columnNames, nameFile)
	variableDict = np.memmap(variableDictFile, dtype='int32', mode='w+', shape=(bootstraps,modelData.shape[1]))

	modelData = load(dataFile, mmap_mode='r')
	columnNames = load(nameFile, mmap_mode='r')

	#Run the bootstraps in parallel
	Parallel(n_jobs=cores)(delayed(boridgeFeatureSelect)(sample,bootstrapIterations,modelData,columnNames,variableDict,responseVariableName,correlationThreshold,printProgress,errorLogFile,outcomeType)
		for sample in bootstrapIterations)

	if printProgress:
		print("BoRidge Time: %s" % (time.time() - boridgeTime))

	#Delete the memory map and temp folder
	try:
 	   	shutil.rmtree(temp_folder)
	except OSError:
	    	pass

	selectedModel=[responseVariableName]
	maxBootstrapCount=0
	bootstrapThreshold=round(bootstraps*bootstrapPercentage,0)

	#Find which variables are signficant
	for index in range(0,variableDict.shape[1]):
		if columnNames[index]!=responseVariableName:
			boridgeCount=np.sum(variableDict[:, index])
			feature_dictionary[columnNames[index]]=boridgeCount
			if boridgeCount >= bootstrapThreshold:
				selectedModel.append(columnNames[index])
			if boridgeCount>maxBootstrapCount:
				maxBootstrapCount=boridgeCount

	#If no predictors meet the threshold, reset the threshold as the maximum number of times a variable was significant
	if len(selectedModel)==1:
		for index in range(0,variableDict.shape[1]):
			if columnNames[index]!=responseVariableName:
				boridgeCount=np.sum(variableDict[:, index])
				feature_dictionary[columnNames[index]]=boridgeCount
				if boridgeCount >= maxBootstrapCount:
					selectedModel.append(columnNames[index])

	#if printProgress:
	#	print("Feature BoRidge Counts")
	#	print(feature_dictionary)


	#Make sure first order terms are included with higher order variable
	higherOrderPattern=re.compile('\_\^\_')

	for regressor in selectedModel:
		if higherOrderPattern.search(regressor)!=None:
			firstOrderVariable=higherOrderPattern.split(regressor)[0]
			if firstOrderVariable not in selectedModel:
				selectedModel.append(firstOrderVariable)

	modelData=pd.DataFrame(modelData)
	modelData.columns=columnNames
	selected_modelData=modelData[selectedModel]

	modelsForConsideration={}
	modelDataForConsideration={}
	#Check the number of Events Per Variable
	if ((selected_modelData.shape[1]-1)*epvThreshold) <= eventCount:
		for typeOfModel in typesOfModels:
			modelDataForConsideration[typeOfModel]=""


	if printProgress:
		print("Selected Model")
		print(list(selected_modelData.columns.values))

	if len(modelDataForConsideration.keys())==0:
		e = ""
		errorLine = ""
		printError(thisFileName+".py: The number of samples per column is too low to provide reasonable estimates: EPV < "+str(epvThreshold), e, errorLine, errorLogFile)
		return(emptyFrame)

	brierScores={}
	cStatistics={}
	votes={"L2":0.0,"SVM":0.0,"RF":0.0}
	bestBrier=[1.0, []]
	bestMCC=[0.0, []]
	bestC=[0.0, []]

	if printProgress:
		print("Begin Optimism Adjusted Evaluation")

	evalBootstrapTime=time.time()




	# Framework for comparing different types of models in the future, eg Elastic Net Logistic Regression vs Random Forest
	for model in modelDataForConsideration.keys():

		if printProgress:
			print("Evaluate Model "+str(model))


		results=evaluate_model(selected_modelData,responseVariableName,model,cores,bootstraps,outcomeType,printProgress,errorLogFile)

		if len(results)==0:
			return(emptyFrame)
		modelsForConsideration[model]=results[0]
		cStatistic=results[1]
		brierScore=results[2]
		mcc=results[3]


		brierScores[model]=brierScore
		cStatistics[model]=cStatistic

		if brierScore < bestBrier[0]:
			bestBrier[0]=brierScore
			bestBrier[1]=[model]
		elif brierScore == bestBrier[0]:
			bestBrier[1].append(model)

		if mcc > bestMCC[0]:
			bestMCC[0]=mcc
			bestMCC[1]=[model]
		elif mcc == bestMCC[0]:
			bestMCC[1].append(model)

		if cStatistic > bestC[0]:
			bestC[0]=cStatistic
			bestC[1]=[model]
		elif cStatistic == bestC[0]:
			bestC[1].append(model)

	if printProgress:
		print("Evaluation Time: %s" % (time.time() - evalBootstrapTime))

	for model in bestBrier[1]:
		votes[model]+=0.9

	for model in bestMCC[1]:
		votes[model]+=0.5

	for model in bestC[1]:
		votes[model]+=1

	bestModelVotes=0.0
	bestModelName=""
	for model in votes.keys():
		if votes[model]>bestModelVotes:
			bestModelVotes=votes[model]
			bestModelName=model

	brierDescript=""
	cDescript=""
	if outcomeType=="classify":
		brierDescript="Brier Score"
		cDescript="AUC"
	elif outcomeType=="regress":
		brierDescript="Mean Squared Error"
		cDescript="Explained Variance Score"

	print("Votes")
	print(votes)
	print(brierDescript)
	print(brierScores)
	print(cDescript)
	print(cStatistics)


	bestModelColumnNames=list(selected_modelData.columns.values)
	bestModelFeatureNames=list(selected_modelData.drop(responseVariableName,1).columns.values)


	try:
		for regressor in bestModelColumnNames:
			printError(thisFileName+".py: "+"Audit Log: regressor "+regressor+" selected for model by BoRidge feature selection", "", "", errorLogFile)
		printError(thisFileName+".py: "+"Audit Log: "+bestModelName+" is the Best Model", "", "", errorLogFile)
		printError(thisFileName+".py: "+"Audit Log: Best Model Optimism Adjusted "+brierDescript+" = "+str(brierScores[bestModelName]), "", "", errorLogFile)
		printError(thisFileName+".py: "+"Audit Log: Best Model Optimism Adjusted "+cDescript+" = "+str(cStatistics[bestModelName]), "", "", errorLogFile)
	except:
		e = sys.exc_info()[0:1]
		errorLine = sys.exc_info()[-1]
		printError(thisFileName+".py: Error writing BoRidge log", e, errorLine, errorLogFile)
		return(emptyFrame)

	#Do model perfomance chacks
	if cStatistics[bestModelName] <= cStatisticThreshold:
		e = ""
		errorLine = ""
		if outcomeType=="classify":
			printError(thisFileName+".py: Model discrimination is below the AUC threshold of "+str(cStatisticThreshold)+". Model features need redesign: "+str(bestModelColumnNames), e, errorLine, errorLogFile)
		elif outcomeType=="regress":
			printError(thisFileName+".py: Model fit is below the Explained Variance threshold of "+str(cStatisticThreshold)+". Model features need redesign: "+str(bestModelColumnNames), e, errorLine, errorLogFile)
		return(emptyFrame)

	if brierScores[bestModelName] >= brierScoreThreshold:
		e = ""
		errorLine = ""
		if outcomeType=="regress":
			printError(thisFileName+".py: Model error is above the MSE threshold of "+str(brierScoreThreshold)+". Model features need redesign: "+str(bestModelColumnNames), e, errorLine, errorLogFile)
		elif outcomeType=="classify":
			printError(thisFileName+".py: Model calibration is above the Brier score threshold of "+str(brierScoreThreshold)+". Model features need redesign: "+str(bestModelColumnNames), e, errorLine, errorLogFile)
			return(emptyFrame)

	if printProgress:
		print("total time: %s seconds" % (time.time() - totalStart))

	#Check for output type
	if outputType=="data":
		return(selected_modelData)
	elif outputType=="model":
		return(modelsForConsideration[bestModelName])
	else:

		regressorIndex=0
		modelOutputDataFrame=pd.DataFrame(columns=['predictor_name','coefficient','lower_bound','upper_bound'])

		features=selected_modelData.drop(responseVariableName,1)
		response=selected_modelData[responseVariableName]
		bestModel=modelsForConsideration["L2"]
		coefficients={}
		if outcomeType=="regress":
			coefficients=bestModel.coef_.tolist()
		else:
			coefficients=bestModel.coef_[0].tolist()

		standardErrors=[]

		# TODO  switch to Huber-White corrected standard errors
		if outcomeType=="regress":
			try:
				yVector = response.copy()
				yVector = yVector.as_matrix()
				designMatrix = features.copy()
				designMatrix.insert(0,'intercept',pd.Series([1]*features.shape[0],index=designMatrix.index))
				designMatrix = designMatrix.as_matrix()
				# H = X(X'X)^X'
				hatMatrix = np.dot(np.dot(designMatrix,np.linalg.inv(np.dot(designMatrix.T,designMatrix))),designMatrix.T)
				# MSE = y'y - y'Hy
				MSE=(np.dot(yVector.T,yVector)-np.dot(np.dot(yVector.T,hatMatrix),yVector))/float(features.shape[0]-features.shape[1])
				covarianceMatrix=np.dot(MSE,np.linalg.inv(np.dot(designMatrix.T,designMatrix)))
				standardErrors=np.sqrt(np.diag(covarianceMatrix))
			except:
				e = sys.exc_info()[0:1]
				errorLine = sys.exc_info()[-1]
				printError(thisFileName+".py: Error calculating covariance matrix for "+str(bestModelName)+" model", e, errorLine, errorLogFile)
				return(emptyFrame)
		else:
			try:
				predictedProbabilities = np.matrix(bestModel.predict_proba(features))
				# Design matrix -- add column of 1's at the beginning of your
				designMatrix = features.copy()
				designMatrix.insert(0,'intercept',pd.Series([1]*features.shape[0],index=designMatrix.index))
				designMatrix = designMatrix.as_matrix()

				# Initiate matrix of 0's, fill diagonal with each predicted observation's variance
				vMatrix = np.matrix(np.zeros(shape = (designMatrix.shape[0], designMatrix.shape[0])))

				np.fill_diagonal(vMatrix, np.multiply(predictedProbabilities[:,0], predictedProbabilities[:,1]).A1)

				#Add fudge factor for probabilities
				for i in range(vMatrix.shape[0]):
					if vMatrix[i,i]<np.float_(1.0e-16):
						vMatrix[i,i]=np.float_(1.0e-16)

				intermediate = np.dot(np.dot(designMatrix.T,vMatrix),designMatrix)

				covarianceMatrix = np.linalg.inv(intermediate)

				# Standard errors
				standardErrors=np.sqrt(np.diag(covarianceMatrix))

			except:
				e = sys.exc_info()[0:1]
				errorLine = sys.exc_info()[-1]
				printError(thisFileName+".py: Error calculating covariance matrix for "+str(bestModelName)+" model", e, errorLine, errorLogFile)
				return(emptyFrame)

		upperCI=[]
		lowerCI=[]
		inputIndex=0

		try:
			for regressorIndex in list(range(len(coefficients))):
				featureName=bestModelFeatureNames[regressorIndex]
				if outcomeType=="regress":
					upperCI.append(CIboundLinear(coefficients[regressorIndex],standardErrors[regressorIndex+1],bootstraps,True))
					lowerCI.append(CIboundLinear(coefficients[regressorIndex],standardErrors[regressorIndex+1],bootstraps,False))
					modelOutputDataFrame.loc[inputIndex]=[bestModelFeatureNames[regressorIndex],coefficients[regressorIndex],lowerCI[regressorIndex],upperCI[regressorIndex]]
					inputIndex+=1
				elif outcomeType=="classify":
					upperCI.append(CIboundLogit(coefficients[regressorIndex],standardErrors[regressorIndex+1],bootstraps,True))
					lowerCI.append(CIboundLogit(coefficients[regressorIndex],standardErrors[regressorIndex+1],bootstraps,False))
					modelOutputDataFrame.loc[inputIndex]=[bestModelFeatureNames[regressorIndex],math.exp(coefficients[regressorIndex]),lowerCI[regressorIndex],upperCI[regressorIndex]]
					inputIndex+=1



		except:
			e = sys.exc_info()[0:1]
			errorLine = sys.exc_info()[-1]
			printError(thisFileName+".py: Error building predictor output", e, errorLine, errorLogFile)
			return(emptyFrame)

		return(modelOutputDataFrame)
