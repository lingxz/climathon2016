import xarray as xr
import pandas as pd
import seaborn as sns

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import scipy.stats
import scipy.optimize as spo

# convenience function for reshaping for use in sklearn linear regression
def reshape(xarray_col):
    return xarray_col.values.reshape((len(xarray_col.values), 1))

def chronological_split(df_xcol, df_ycol, test_size=0.2):
    assert len(df_xcol) == len(df_ycol)
    test_number = int(test_size * len(df_xcol))
    train_number = len(df_xcol) - test_number
    return df_xcol.head(train_number), df_xcol.tail(test_number), df_ycol.head(train_number), df_ycol.tail(test_number)


def convert_co(CO_working_mv=None, CO_aux_mv=None, temp_correction=+1):

    ANALOG_REF_VOLTAGE = 3.3

    # AFE board serial number 12-000027
    # sensor CO A4 serial number 132950238

    CO_WORKING_ELECTRODE_ELECTRONIC_ZERO_MV = 268
    CO_WORKING_ELECTRODE_SENSOR_ZERO_MV = 42

    CO_AUXILIARY_ELECTRODE_ELECTRONIC_ZERO_MV = 262
    CO_AUXILIARY_ELECTRODE_SENSOR_ZERO_MV = 10
    CO_SENSITIVITY = 0.228  # mv/ppb

    scaled_working_CO_nA = (CO_working_mv - CO_WORKING_ELECTRODE_ELECTRONIC_ZERO_MV) / 0.8
    scaled_aux_CO_nA = (CO_aux_mv - CO_AUXILIARY_ELECTRODE_ELECTRONIC_ZERO_MV) / 0.8 * temp_correction
    return (scaled_working_CO_nA - scaled_aux_CO_nA) / CO_SENSITIVITY

def convert_no2(NO2_working_mv=None, NO2_aux_mv=None, temp_correction=+1.09):
    # AFE board serial number 12-000027
    # sensor NO2 A43F serial number 212060336
    NO2_WORKING_ELECTRODE_ELECTRONIC_ZERO_MV = 295
    NO2_WORKING_ELECTRODE_SENSOR_ZERO_MV = 2
    NO2_AUXILIARY_ELECTRODE_ELECTRONIC_ZERO_MV = 295
    NO2_AUXILIARY_ELECTRODE_SENSOR_ZERO_MV = 0
    NO2_SENSITIVITY = 0.197  # mv/ppb

    # temp corretion is +1.09 up to 20 deg Cels, then 1.35, then 3 after 30C

    scaled_working_NO2_nA = (NO2_working_mv - NO2_WORKING_ELECTRODE_ELECTRONIC_ZERO_MV) / 0.8
    scaled_aux_NO2_nA = (NO2_aux_mv - NO2_AUXILIARY_ELECTRODE_ELECTRONIC_ZERO_MV) / 0.8 * temp_correction
    return (scaled_working_NO2_nA - scaled_aux_NO2_nA) / NO2_SENSITIVITY

# reverse to raw readings
def reverse_co(CO_working=None, CO_aux=None, ppbCO=None):
    ANALOG_REF_VOLTAGE = 3.3

    # sensor 1 values from mcu ( they are wrong and differ from datasheet)

    mcu_CO_WORKING_ELECTRODE_ZERO_OFFSET_MV = 310
    mcu_CO_AUXILIARY_ELECTRODE_ZERO_OFFSET_MV = 272
    mcu_CO_SENSITIVITY = 0.197  # it is wrong on the mcu(switched with no2)

    # float voltageCO_working = ANALOG_REF_VOLTAGE * raw_CO_working / 1024.0 * 1000.0;
    # sensor_readings->mvCO_working = voltageCO_working;

    raw_CO_working_1 = CO_working * 1024.0 / (1000.0 * ANALOG_REF_VOLTAGE)

    raw_CO_working_2 = (ppbCO * mcu_CO_SENSITIVITY + mcu_CO_WORKING_ELECTRODE_ZERO_OFFSET_MV) * 1024.0 / (
    1000.0 * ANALOG_REF_VOLTAGE)

    assert abs(raw_CO_working_1 - raw_CO_working_2) < 0.05

    # float voltageCO_aux = ANALOG_REF_VOLTAGE * raw_CO_aux / 1024.0 * 1000.0;
    #  sensor_readings->mvCO_aux = voltageCO_aux;

    raw_CO_aux = CO_aux * 1024.0 / (1000.0 * ANALOG_REF_VOLTAGE)

    return raw_CO_working_1, raw_CO_aux

# reverse to raw readings
def reverse_no2(NO2_working=None, NO2_aux=None, ppbNO2=None):
    ANALOG_REF_VOLTAGE = 3.3
    mcu_NO2_WORKING_ELECTRODE_ZERO_OFFSET_MV = 297
    mcu_NO2_AUXILIARY_ELECTRODE_ZERO_OFFSET_MV = 295
    mcu_NO2_SENSITIVITY = 0.228

    #  float voltageNO2_working = ANALOG_REF_VOLTAGE * raw_NO2_working / 1024.0 * 1000.0;
    #     //logMessage(VERBOSE, "Voltage NO2 Working Electrode (mV): " + String(voltageNO2_working));
    #     sensor_readings->mvNO2_working = voltageNO2_working;

    raw_NO2_working_1 = NO2_working * 1024.0 / (1000.0 * ANALOG_REF_VOLTAGE)

    #         float voltageNO2_working_corrected = voltageNO2_working - NO2_WORKING_ELECTRODE_ZERO_OFFSET_MV;
    #     //logMessage(VERBOSE, "Corrected Voltage NO2 Working Electrode (mV): " + String(voltageNO2_working_corrected));

    #     float ppbNO2 = voltageNO2_working_corrected / NO2_SENSITIVITY;
    #     logMessage(INFO, "NO2:" + String(ppbNO2));
    #     sensor_readings->ppbNO2 = ppbNO2;

    raw_NO2_working_2 = (ppbNO2 * mcu_NO2_SENSITIVITY + mcu_NO2_WORKING_ELECTRODE_ZERO_OFFSET_MV) * 1024.0 / (
    1000.0 * ANALOG_REF_VOLTAGE)

    assert abs(raw_NO2_working_1 - raw_NO2_working_2) < 0.05

    #     float voltageNO2_aux = ANALOG_REF_VOLTAGE * raw_NO2_aux / 1024.0 * 1000.0;
    #     //logMessage(VERBOSE, "Voltage NO2 Auxiliary Electrode (mV): " + String(voltageNO2_aux));
    #     sensor_readings->mvNO2_aux = voltageNO2_aux;

    raw_NO2_aux = NO2_aux * 1024.0 / (1000.0 * ANALOG_REF_VOLTAGE)

    return raw_NO2_working_1, raw_NO2_aux


class Colocation:

    def __init__(self, df, kings_df):
        self.df = df
        self.kings_df = kings_df

    def convert_co(self, working_col, aux_col, new_colname):
        self.df[new_colname] = self.df[[working_col, aux_col]].apply(lambda x: convert_co(x.values[0], x.values[1]),
                                                                  axis=1)

    def convert_no2(self, working_col, aux_col, new_colname):
        self.df[new_colname] = self.df[[working_col, aux_col]].apply(lambda x: convert_co(x.values[0], x.values[1]),
                                                                  axis=1)

    def reverse_no2(self, non_raw_working, non_raw_aux, ppb):
        new_raw_working = "raw_" + non_raw_working
        new_raw_aux = "raw_" + non_raw_aux
        self.df[new_raw_working] = self.df[[non_raw_working, non_raw_aux, ppb]].apply(
            lambda x: reverse_no2(x.values[0], x.values[1], x.values[2])[0], axis=1)

        self.df[new_raw_aux] = self.df[[non_raw_working, non_raw_aux, ppb]].apply(
            lambda x: reverse_no2(x.values[0], x.values[1], x.values[2])[1], axis=1)

    def reverse_co(self, non_raw_working, non_raw_aux, ppb):
        new_raw_working = "raw_" + non_raw_working
        new_raw_aux = "raw_" + non_raw_aux
        self.df[new_raw_working] = self.df[[non_raw_working, non_raw_aux, ppb]].apply(
            lambda x: reverse_co(x.values[0], x.values[1], x.values[2])[0], axis=1)

        self.df[new_raw_aux] = self.df[[non_raw_working, non_raw_aux, ppb]].apply(
            lambda x: reverse_co(x.values[0], x.values[1], x.values[2])[1], axis=1)


    def get_merged_ds(self):
        df_kings_vol = self.kings_df
        ds = xr.Dataset.from_dataframe(self.df.set_index('timestamp', 'id'))
        ds_mean_1h = ds.resample(freq='1H', dim='timestamp', how='mean', skipna=True)
        ds_kings_vol = xr.Dataset.from_dataframe(df_kings_vol)
        kings_prefix = dict([(v, 'kings_' + v) for v in ds_kings_vol.data_vars])
        ds_merged = xr.merge([ds_kings_vol.rename(kings_prefix), ds_mean_1h]
                             , join='inner')
        self.ds = ds_merged
        return ds_merged

    def compare(self, airpublic_measure, kings_measure):
        ds_merged = self.ds
        ds_merged_one = ds_merged[[airpublic_measure, kings_measure]]
        # ds_merged_one['error'] = ds_merged_one[kings_measure] - ds_merged[airpublic_measure]
        # ds_merged_one.sel(datetime=example_date).to_dataframe().plot()
        ds_merged_one.to_dataframe().plot()
        sns.pairplot(ds_merged_one.to_dataframe().dropna().reset_index(), vars=ds_merged_one.data_vars)
        print(ds_merged_one.to_dataframe().describe())
        # calculate r2 score
        r2 = r2_score(ds_merged_one.to_dataframe().dropna()[airpublic_measure],
                      ds_merged_one.to_dataframe().dropna()[kings_measure])
        print("r2 score: ", r2)

        # KS-test
        ks_statistic, p_value = scipy.stats.ks_2samp(ds_merged_one.to_dataframe().dropna()[airpublic_measure],
                                                     ds_merged_one.to_dataframe().dropna()[kings_measure])
        print("Probability that they belong to the same distribution (KS p-value): ", p_value)



    def linear_regr_fudge(self, airpublic_measure, kings_measure, test_size=0.2, split='random'):
        if split not in ['random', 'chronological']:
            raise ValueError("splitting method should either be random or chronological")
        ds_merged = self.ds
        df_merged_temp = ds_merged.to_dataframe().dropna(subset=[kings_measure, airpublic_measure])
        if split == 'random':
            kings_train, kings_test, ap_train, ap_test = \
                train_test_split(df_merged_temp[kings_measure], df_merged_temp[airpublic_measure], test_size=test_size)
        else:
            kings_train, kings_test, ap_train, ap_test = \
                chronological_split(df_merged_temp[kings_measure], df_merged_temp[airpublic_measure], test_size=test_size)

        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(reshape(ap_train), reshape(kings_train))
        # The coefficients
        print('Coefficients: \n', regr.coef_, regr.intercept_)
        # The mean squared error
        print("Mean squared error: %.2f"
              % np.mean((regr.predict(reshape(ap_test)) - reshape(kings_test)) ** 2))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % regr.score(reshape(ap_test), reshape(kings_test)))
        scores = cross_val_score(regr, reshape(ap_test), reshape(kings_test), scoring='r2')
        print("R2 score: %.3f" % scores[0])

        plt.scatter(ap_test, kings_test, color='black')
        plt.scatter(ap_train, kings_train, color='red')
        plt.plot(reshape(ap_train), regr.predict(reshape(ap_train)), color='blue',
                 linewidth=1)
        plt.show()
