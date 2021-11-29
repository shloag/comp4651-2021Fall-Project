import pyspark
import pandas as pd
import datetime as dt
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import logging

# Config
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
		lin_reg = LinearRegression(featuresCol='Feature', labelCol=ticker_name)
		assembler = VectorAssembler().setInputCols(['Date']).setOutputCol('Feature')

		try:
			linear_model = lin_reg.fit(assembler.transform(df))
			all_models_dict[ticker_name] = [linear_model.coefficients[0], linear_model.intercept]
			logging.info(f'Fitted {ticker_name}')
		except:
			logging.info(f'Cannot fit {ticker_name}')


		return pd.DataFrame(all_models_dict).transpose().rename(columns={0: 'm', 1: 'c'})

if __name__ == '__main__':
	spark = SparkSession.builder.appName('Financial data analyzer').getOrCreate()
	df = spark.read.option('header', 'true').csv('All_SP500_Companies_Close_Price_From_20000101_to_20200101.csv', inferSchema=True)

	ticker_names = df.schema.names[1:]

	dfs = make_multiple_dfs(df)
	models_df = fit_dfs(dfs, ticker_names)
	logging.info('Saving models to csv...')
	models_df.to_csv('models.csv')
	logging.info('Saved')