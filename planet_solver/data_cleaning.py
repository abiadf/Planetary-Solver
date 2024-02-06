'''Module dealing with cleaning the data, by filling gaps, smoothening data, 
then makes derivatives of the data, and clips these derivatives (if too sharp)'''

import os

import numpy as np
import pandas as pd

class DataCleaner:
    '''Class that fills gaps in data, smoothens data'''

    @staticmethod
    def fill_data_gaps(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''Fills the NaN gaps in data without removing entries
        Input: dataframe containing [time, T, C, F] data
        Output: dataframe without NaN values'''

        df_filled = df.copy()

        time_column_id    = df.columns.get_loc(DfConstants.df_time_column)
        number_of_columns = df_filled.shape[1]
       
        # Possibilities to close data gaps: ffill, bfill, dropna, interpolate (linear/spline/polynomial), fillna, fillna(mean), rolling
        for i in range(time_column_id + 1, number_of_columns):
            df_filled.iloc[:, i] = df.iloc[:, i].fillna(method = 'bfill')

            # Handling NaN of a column's LAST value:
            if pd.isna(df_filled.iloc[-1, i]):
                last_non_null_value   = df_filled.iloc[:, i].dropna().iloc[-1]
                df_filled.iloc[-1, i] = last_non_null_value

        return df_filled


    @staticmethod
    def smoothen_data(df: pd.core.frame.DataFrame, window_size: int = 5) -> pd.core.frame.DataFrame:
        '''Smoothens the data per column (temp, cond, flow)
        INPUT: dataframe containing [time, T, C, F] data
        window_size: how big is the averaging window. Somewhere between 3-5 is good'''

        df_smooth         = df.copy()

        time_column_id    = df.columns.get_loc(DfConstants.df_time_column)
        number_of_columns = df_smooth.shape[1]

        for i in range(time_column_id + 1, number_of_columns):
            df_smooth.iloc[:, i] = df.iloc[:, i].rolling(window_size, min_periods = 1).mean()

        return df_smooth


    @staticmethod
    def make_derivatives(df: pd.core.frame.DataFrame, dx: int = 1) -> pd.core.frame.DataFrame:
        '''Computes 1st and 2nd derivative of dataframe
        INPUT:
              - df: input dataframe
              - dx: spacing for differentiation, default set to 1
        OUTPUT: 2 Dataframes with derived values, one for the 1st derivative, one for the 2nd derivative,
        having columns [time, temp, cond, flow]'''

        df_diff  = df.copy() # 1st derivative
        df_diff2 = df.copy() # 2nd derivative

        time_column_id       = df.columns.get_loc(DfConstants.df_time_column)
        number_of_columns    = df_diff.shape[1]
        coeff_to_offset_diff = -4 # offsets differentiation

        for i in range(time_column_id + 1, number_of_columns):
            df_diff.iloc[:, i]  = np.gradient(df.iloc[:, i], dx) # 1st derivative
            df_diff2.iloc[:, i] = np.gradient(df_diff.iloc[:, i], dx) # 2nd derivative

            # Every instance of differentiation causes an offset. Should be countered
            df_diff.iloc[:, i]  = np.roll(df_diff.iloc[:, i], coeff_to_offset_diff)
            df_diff2.iloc[:, i] = np.roll(df_diff2.iloc[:, i], coeff_to_offset_diff)
        
        return df_diff, df_diff2


    @staticmethod
    def clip_derivatives(df_diff: pd.core.frame.DataFrame, criterion: float = 0.005) -> pd.core.frame.DataFrame:
        '''Clipper functionality: if derivative is +ve and below X% of the max derivative value, 
        make it negative. This will make the derivative processing easier by getting rid of small bumps
        INPUT:
        - derivative dataframe
        - criterion: interpret as 'below which fraction of the max value do we neglect values?'
        OUTPUT: derivative dataframe with flipped small values'''

        dYdx = df_diff.copy()

        time_column_id = df_diff.columns.get_loc(DfConstants.df_time_column)
        num_of_columns = dYdx.shape[1]
        dY_max         = np.zeros(num_of_columns)

        for column in range(time_column_id + 1, num_of_columns):
            dY_max_val= dYdx.iloc[:, column].max()
            mask      = (dYdx.iloc[:, column] < (criterion * dY_max_val)) & (dYdx.iloc[:, column] > 0) # if derivative is +ve AND below X% of the max derivative value
            dYdx.loc[mask, dYdx.columns[column]] *= -1

        return dYdx



