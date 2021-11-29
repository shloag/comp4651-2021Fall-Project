import pyspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import logging

# Config
work_in_small = True
logging.basicConfig(level=logging.INFO)


def make_multiple_dfs(df):
	"""
	Assumed df in the format ['Date', 'ticker1', 'ticker2', ...]
	"""
	dfs = []

	col_names = df.schema.names

	first_col_name = col_names[0]
	tickers = col_names[1:]

	for ticker_name in tickers:
		current_df = df.select([first_col_name, ticker_name]).na.drop()
		dfs.append(current_df)

	return dfs


def fit_dfs(dfs, ticker_names):
	"""
	Fit N models to N df
	Return: pandas Dataframe
	"""
	all_models_dict = {}
	for i, ticker_name in enumerate(ticker_names):
		df = dfs[i]

		train, test = df.randomSplit([0.75, 0.25])
		lin_reg = LinearRegression(featuresCol='Feature', labelCol=ticker_name)
		assembler = VectorAssembler().setInputCols(
		    ['Date']).setOutputCol('Feature')
		linear_model = lin_reg.fit(assembler.transform(train))
		all_models_dict[ticker_name] = [linear_model.coefficients[0], linear_model.intercept]

	return pd.DataFrame(all_models_dict).transpose().rename(columns={0: 'm', 1: 'c'})

if __name__ == '__main__':
	spark = SparkSession.builder.appName('Financial data analyzer').getOrCreate()
	df = spark.read.option('header', 'true').csv('All_SP500_Companies_Close_Price_From_20000101_to_20200101.csv', inferSchema=True)

	if work_in_small:
		df = df.select(['Date', 'AAPL', 'A', 'AAL', 'MMM'])

	ticker_names = df.schema.names[1:]

	logging.info(f'The head of the current df is {df.head()}')
	dfs = make_multiple_dfs(df)

	models_df = fit_dfs(dfs, ticker_names)
	logging.INFO('Saving models to csv...')
	models_df.to_csv('models.csv')