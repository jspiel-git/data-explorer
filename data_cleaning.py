import numpy as np
import pandas as pd

train_data = pd.read_csv("data/train.csv")
ntrain = len(train_data)
test_data = pd.read_csv("data/test.csv")
full_data = pd.concat([train_data, test_data], sort = False, ignore_index = True)

dependent_variable = "SalePrice"

deleted_variables = ["Id", "MSSubClass", "LotFrontage", "Alley", "Utilities", "LandSlope", "Condition2", "YearRemodAdd", 
	"RoofMatl", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", 
	"BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "Heating", "2ndFlrSF", "LowQualFinSF", 
	"Electrical", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "BedroomAbvGr", "KitchenAbvGr", "Functional", "FireplaceQu", 
	"GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageQual", "GarageCond", "ScreenPorch", "PoolQC",
	"Fence", "MiscFeature", "MoSold", "YrSold", "SaleType"]

full_data = full_data.drop(deleted_variables, axis = 1)

# We drop MSSubClass since it represents categories created by house vendors base on
# age and type of building. Which are already in HouseStyle and Year

# For BldgType, we group (1Fam, TwnhsE) and (Duplex, Twnhs, 2fmCon) to have a single familiy variable

full_data["SingleFam"] = full_data["BldgType"].apply(lambda x: 1 if (x == "1Fam" or x == "TwnhsE") else 0)
full_data = full_data.drop("BldgType", axis = 1)

# OverallQual is categorical but ordered from 1 to 10

# YearBuilt we replace by age

full_data["Age"] = full_data["YearBuilt"].apply(lambda x: 2020 - x)
full_data = full_data.drop("YearBuilt", axis = 1)

# RoofStyle group (Gable, Gambrel, Mansard) and (Hip, Flat, Shed)

# Exterior1st One Nan value

# FullBath : unknown category in testing set, categorical ordered
# Also join FullBath + 0.5 * HalfBath

full_data["NumBath"] = full_data.apply(lambda x: int(x["FullBath"]) + 0.5 * int(x["HalfBath"]), axis = 1)
full_data = full_data.drop(["FullBath", "HalfBath"], axis = 1)

# TotalBsmtSF transform in categorical has basement or not

full_data["HasBasement"] = full_data["TotalBsmtSF"].apply(lambda x: 1 if (x > 0) else 0)
full_data = full_data.drop("TotalBsmtSF", axis = 1)

# We drop BedroomAbvGr and KitchenAbvGr since summarized in TotRmsAbvGrd

# We drop Functional since summarized in OverAllQual

# Fireplaces group by has fireplace or not

full_data["HasFireplace"] = full_data["Fireplaces"].apply(lambda x: 1 if (x > 0) else 0)
full_data = full_data.drop("Fireplaces", axis = 1)

# WoodDeckSF and OpenPorchSF and EnclosedPorch and PoolArea, do categorical, has or not

full_data["HasWoodDeck"] = full_data["WoodDeckSF"].apply(lambda x: 1 if (x > 0) else 0)
full_data = full_data.drop("WoodDeckSF", axis = 1)

full_data["HasOpenPorch"] = full_data["OpenPorchSF"].apply(lambda x: 1 if (x > 0) else 0)
full_data = full_data.drop("OpenPorchSF", axis = 1)

full_data["HasClosedPorch"] = full_data["EnclosedPorch"].apply(lambda x: 1 if (x > 0) else 0)
full_data = full_data.drop("EnclosedPorch", axis = 1)

full_data["HasSsnPorch"] = full_data["3SsnPorch"].apply(lambda x: 1 if (x > 0) else 0)
full_data = full_data.drop("3SsnPorch", axis = 1)

full_data["HasPool"] = full_data["PoolArea"].apply(lambda x: 1 if (x > 0) else 0)
full_data = full_data.drop("PoolArea", axis = 1)

# We complete the missing data by the most common categories

full_data["KitchenQual"] = full_data["KitchenQual"].fillna("TA")
full_data["GarageArea"] = full_data["GarageArea"].fillna("0")
full_data["MSZoning"] = full_data["MSZoning"].fillna("RL")
full_data["Exterior1st"] = full_data["Exterior1st"].fillna("VinylSd")

# We need to transform the categorical variables into 0 and 1

categorical_variables = ["Street", "CentralAir", "MSZoning", "LotShape", "LandContour", "LotConfig", "Neighborhood", "Condition1",
	"HouseStyle", "RoofStyle", "Exterior1st", "Foundation", "HeatingQC", "KitchenQual", "PavedDrive", 
	"SaleCondition"]

full_data_dummies = full_data

for column in categorical_variables:

	di = {}
	i = 0

	for ind in np.asarray(full_data[column].value_counts().index):

		di[str(ind)] = i
		i += 1
	
	full_data[column] = full_data[column].map(di)
	pclass_dummies = pd.get_dummies(full_data[column], prefix = column)
	full_data_dummies = pd.concat([full_data_dummies, pclass_dummies], axis = 1)

full_data_dummies = full_data_dummies.drop(categorical_variables, axis = 1)

train_data = full_data_dummies.iloc[:ntrain, :]
test_data = full_data_dummies.iloc[ntrain:, :]

train_data.to_csv("data/train_clean.csv", index = False)
test_data.to_csv("data/test_clean.csv", index = False)

