# Data exploration
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import io

from mdutils.mdutils import MdUtils

### Input : training data CSV file and name of dependent variable
### Output : Markdown file

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
full_data = pd.concat([train_data, test_data], sort = False, ignore_index = True)

full_data = full_data.drop(["Id"], axis = 1)

### Here give option to give columns to drop and to declare as dependent variable

# X_train = train_data.drop(["Id", "SalePrice"], axis = 1).values
# X_test = test_data.drop(["Id"], axis = 1).values

dependent_variable = "SalePrice"

# y_train = train_data[dependent_variable].values

ntrain = len(train_data)
ntest = len(test_data)
nfull = len(full_data)
nvariables = len(full_data.columns)

mdFile = MdUtils(file_name = "data_analysis", title = "Data set informations")

buffer = io.StringIO()

full_data.info(buf = buffer)
full_data_info = buffer.getvalue().split("\n")

mdFile.new_line("Number of variables (including dependent): {}".format(nvariables))
mdFile.new_line("Number of training examples: {}".format(ntrain))
mdFile.new_line("Number of testing examples: {}".format(ntest))

mdFile.new_header(level = 1, title = "Detail of variables")

i = 3

for column in full_data.columns:
	
	if column == dependent_variable:
		mdFile.new_header(level = 2, title = "Variable name: {} (dependent variable)".format(column))
	else:
		mdFile.new_header(level = 2, title = "Variable name: {}".format(column))

	# mdFile.new_line("**Description:** ")
	# mdFile.new_line("**Expectation:** ")
	# mdFile.new_line("**Comments:** ")

	column_info = []

	for info in full_data_info[i].split(" "):

		if info == "":
			pass
		else:
			column_info.append(info)

	i += 1

	column_type = column_info[3]

	column_notna = full_data[full_data[column].notnull()][column]

	mdFile.new_line("**Type:** {}".format(column_type))
	mdFile.new_line("**Number of NaNs in full data:** {}".format(nfull - int(column_info[1])))
	mdFile.write('  \n')
	# mdFile.new_line("First non-NaN element: {}".format(column_notna.values[0]))

	if column_type == "object":

		mdFile.new_line("**Variable is categorical with following levels:**")

		level_means = train_data.groupby(by = column)[dependent_variable].mean()

		column_levels = column_notna.value_counts()

		j = 0

		while j < len(column_levels):
			try:
				mdFile.new_line("Level {}: **{}**, number of non-NaN examples: {}, mean: {}".format(j + 1, 
					column_levels.index[j], column_levels.values[j], round(level_means[column_levels.index[j]],2)))
			except KeyError:
				mdFile.new_line("Level {}: **{}**, number of non-NaN examples: {}, mean unknown since category only in testing set".format(j + 1, 
					column_levels.index[j], column_levels.values[j]))
			
			j += 1

	elif column_type == "int64":

		mdFile.new_line("**Variable is integer-valued with following values:**")

		level_means = train_data.groupby(by = column)[dependent_variable].mean()

		column_levels = column_notna.value_counts()

		# Here maybe if len(column_levels) too big simply display plot.

		j = 0

		while j < len(column_levels):

			try:
				mdFile.new_line("Value: **{}**, number of non-NaN examples: {}, mean : {}".format(column_levels.index[j], 
					column_levels.values[j], round(level_means[column_levels.index[j]],2)))
			except KeyError:
				mdFile.new_line("Value: **{}**, number of non-NaN examples: {}, mean unknown since category only in testing set".format(column_levels.index[j], 
					column_levels.values[j]))

			if j >= 30:
				mdFile.write('  \n')
				#mdFile.new_line("**There are more {} values not displayed**".format(len(column_notna) - j))

				break
			
			j += 1

	elif column_type == "float64" and column != dependent_variable:

		mdFile.new_line("**Variable is real valued:**")
		mdFile.new_line("Sample mean: {}, Sample standard deviation: {}".format(round(np.mean(column_notna), 2), 
			round(np.std(column_notna), 2)))
		mdFile.new_line("Sample minimum: {}, Sample maximum: {}".format(round(min(column_notna), 2), 
			round(max(column_notna), 2)))

		train_column_notna = train_data[train_data[column].notnull()][column]
		y_notna = train_data[train_data[column].notnull()][dependent_variable]

		X = train_column_notna.values.reshape(-1,1)
		y = y_notna.values.reshape(-1,1)

		reg = LinearRegression().fit(X, y_notna)

		mdFile.new_line("Regression R-squared: {}".format(round(reg.score(X, y_notna), 2)))

	else:

		pass

corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.savefig('heatmap.png', bbox_inches='tight')

# Create markdown file

mdFile.create_md_file()

