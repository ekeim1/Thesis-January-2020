# Import required libraries
import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import numpy as np


# FUNCTIONS___________________________________________________________________________________________________________
def make_df_pm2(filename):
  df = pd.read_table(filename, skiprows=[0, 1, 3])  # replace specific file with variable from upload
  # set date and time column as datetime type
  df['DATE AND TIME GMT'] = df['DATE AND TIME GMT'].astype('datetime64[ns]')
  # rename columns
  df.columns = ['Date and Time in GMT', 'Temperature (Degrees Fahrenheit)', 'Relative Humidity (%)']
  df = df.set_index('Date and Time in GMT')
  return df

# BODY________________________________________________________________________________________________________________
# create initial dataframes for rug room, textile room, costume room, and exterior weather file

# rug storage room
rug = "(OBIX) RugStorage.pm2"
dfR = make_df_pm2(rug)
# resample data to fill in any gaps and to ensure same time interval
upsampledTR = dfR['Temperature (Degrees Fahrenheit)'].resample('15min').mean()
interpolatedTR = upsampledTR.interpolate(method='linear')
upsampledRHR = dfR['Relative Humidity (%)'].resample('15min').mean()
interpolatedRHR = upsampledRHR.interpolate(method='linear')
# create new dataframe with the resampled values
frameR = {'Temp_Rug': interpolatedTR, 'RH_Rug': interpolatedRHR}
resultR = pd.DataFrame(frameR)
# select date range to be examined:
resultR = resultR.loc['2015-07-03 00:00' : '2020-02-06 00:00'] # user changes this as needed

# textile room
textile = "(OBIX) Textile Study (Needlework).pm2"
dfT = make_df_pm2(textile)
# resample
upsampledTT = dfT['Temperature (Degrees Fahrenheit)'].resample('15min').mean()
interpolatedTT = upsampledTT.interpolate(method='linear')
upsampledRHT = dfT['Relative Humidity (%)'].resample('15min').mean()
interpolatedRHT = upsampledRHT.interpolate(method='linear')
# create new dataframe with the resampled values
frameT = {'Temp_Textile': interpolatedTT, 'RH_Textile': interpolatedRHT}
resultT = pd.DataFrame(frameT)
# select date range to be examined:
resultT = resultT.loc['2015-07-03 00:00' : '2020-02-06 00:00'] # user changes this as needed

# costume storage
costume = "Costume Storage.pm2"
dfC = make_df_pm2(costume)
# resample
upsampledTC = dfC['Temperature (Degrees Fahrenheit)'].resample('15min').mean()
interpolatedTC = upsampledTC.interpolate(method='linear')
upsampledRHC = dfC['Relative Humidity (%)'].resample('15min').mean()
interpolatedRHC = upsampledRHC.interpolate(method='linear')
# create new dataframe with the resampled values
frameC = {'Temp_Costume': interpolatedTC, 'RH_Costume': interpolatedRHC}
resultC = pd.DataFrame(frameC)
# select date range to be examined:
resultC = resultC.loc['2015-07-03 00:00' : '2020-02-06 00:00'] # user changes this as needed

# now do the exterior weather data Excel spreadsheet
exteriorW = "exterior weather 2011-1-1-to-2020-3-1.csv"
dfE = pd.read_csv(exteriorW)  # replace specific file with variable from upload
# select columns that we want to work with:
dfE = dfE[['DATE','HourlyDewPointTemperature','HourlyDryBulbTemperature','HourlyPrecipitation',
          'HourlyRelativeHumidity','HourlyStationPressure','HourlyVisibility','HourlyWetBulbTemperature',
           'HourlyWindSpeed']]
# recast the columns as numbers instead of strings:
dfE['DATE'] = dfE['DATE'].astype('datetime64[ns]')
dfE['HourlyDewPointTemperature'] = dfE['HourlyDewPointTemperature'].astype('float64')
dfE['HourlyDryBulbTemperature'] = dfE['HourlyDryBulbTemperature'].astype('float64')
dfE['HourlyPrecipitation'] = dfE['HourlyPrecipitation'].astype('float64')
dfE['HourlyRelativeHumidity'] = dfE['HourlyRelativeHumidity'].astype('float64')
dfE['HourlyStationPressure'] = dfE['HourlyStationPressure'].astype('float64')
dfE['HourlyVisibility'] = dfE['HourlyVisibility'].astype('float64')
dfE['HourlyWetBulbTemperature'] = dfE['HourlyWetBulbTemperature'].astype('float64')
dfE['HourlyWindSpeed'] = dfE['HourlyWindSpeed'].astype('float64')
dfE = dfE.set_index('DATE') # set the index to the date

# resample
upsampledDPTE = dfE['HourlyDewPointTemperature'].resample('15min').mean()
interpolatedDPTE = upsampledDPTE.interpolate(method='linear')
upsampledDBTE = dfE['HourlyDryBulbTemperature'].resample('15min').mean()
interpolatedDBTE = upsampledDBTE.interpolate(method='linear')
upsampledHPE = dfE['HourlyPrecipitation'].resample('15min').mean()
interpolatedHPE = upsampledHPE.interpolate(method='linear')
upsampledHRHE = dfE['HourlyRelativeHumidity'].resample('15min').mean()
interpolatedHRHE = upsampledHRHE.interpolate(method='linear')
upsampledHSPE = dfE['HourlyStationPressure'].resample('15min').mean()
interpolatedHSPE = upsampledHSPE.interpolate(method='linear')
upsampledHVE = dfE['HourlyVisibility'].resample('15min').mean()
interpolatedHVE = upsampledHVE.interpolate(method='linear')
upsampledHWBTE = dfE['HourlyWetBulbTemperature'].resample('15min').mean()
interpolatedHWBTE = upsampledHWBTE.interpolate(method='linear')
upsampledHWSE = dfE['HourlyWindSpeed'].resample('15min').mean()
interpolatedHWSE = upsampledHWSE.interpolate(method='linear')

# create new dataframe with the resampled values
frameE = {'DP_Exterior': interpolatedDPTE, 'DryBulbTemp_Exterior': interpolatedDBTE, 'Precip_Exterior': interpolatedHPE,
          'RH_Exterior': interpolatedHRHE, 'StationPressure_Exterior': interpolatedHSPE, 'Visibility_Exterior': interpolatedHVE,
          'WetBulbTemp_Exterior': interpolatedHWBTE,'WindSpeed_Exterior': interpolatedHWSE}
resultE = pd.DataFrame(frameE)
# select date range to be examined:
resultE = resultE.loc['2015-07-03 00:00' : '2020-02-06 00:00'] # user changes this as needed

# combine the dataframes into one big overall one, adding on to resultE
resultE['Temp_Rug'] = resultR['Temp_Rug']
resultE['RH_Rug'] = resultR['RH_Rug']
resultE['Temp_Textile'] = resultT['Temp_Textile']
resultE['RH_Textile'] = resultT['RH_Textile']
resultE['Temp_Costume'] = resultC['Temp_Costume']
resultE['RH_Costume'] = resultC['RH_Costume']
# get rid of NaN values
resultE.dropna(inplace=True)

# FACTOR ANALYSIS_______________________________________________________________________________________________________
# calculate the number of variables
numVars = resultE.shape[1]
print(numVars) # = 14 for these data files

# # Bartlett's test
# # Is this data statistically significant?
# # A p-value < 0.05 means that data is statistically significant
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(resultE)
print(chi_square_value, p_value)
#
# # Kaiser-Meyer-Olkin test
# # Measures suitability of the data for factor analysis
# # 0 <= KMO <= 1, where KMO <= 0.6 is not adequate for factor analysis
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(resultE)
print(kmo_model)
#
# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.fit(resultE)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
print(ev) # the number of eigenvalues >= 1 is the max number of factors you need to potentially consider

# Create scree plot using matplotlib
# look at the difference between eigenvalues - we can see that after a certain point, they are super similar
plt.scatter(range(1,resultE.shape[1]+1),ev)
plt.plot(range(1,resultE.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# Create factor analysis object and perform factor analysis based on number of factors determined above
numFactors = 2 # USER CHANGES THIS LINE AS NEEDED
fa = FactorAnalyzer(numFactors, rotation="varimax")
fa.fit(resultE)
#print(fa.loadings_) # can uncomment to see loadings

# make the results of factor analysis understandable
L = np.array(fa.loadings_)
headings = list(resultE.columns)
factor_threshold = 0.25
for i, factor in enumerate(L.transpose()):
  descending = np.argsort(np.abs(factor))[::-1]
  contributions = [(np.round(factor[x],2),headings[x]) for x in descending if np.abs(factor[x])>factor_threshold]
  print('Factor %d:'%(i+1),contributions)

# Get variance of each factors
print(fa.get_factor_variance())