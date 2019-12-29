#
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for processing OBS data.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
from collections import OrderedDict
from mosx.MesoPy import Meso
#from metpy.io import get_upper_air_data
from metpy.calc import interp
from mosx.util import generate_dates, get_array, read_pkl


def upper_air(config, station_id, sounding_station_id, date, use_nan_sounding=False, use_existing=True, save=True):
    """
    Retrieves upper-air data and interpolates to pressure levels. If use_nan_sounding is True, then if a retrieval
    error occurs, a blank sounding will be returned instead of an error.
    :param config:
    :param station_id: station ID of surface station used
    :param sounding_station_id: station ID of sounding station to use
    :param date: datetime
    :param use_nan_sounding: bool: if True, use sounding of NaNs instead of raising an error
    :param use_existing: bool: preferentially use existing soundings in sounding_data_dir
    :param save: bool: if True, save processed soundings to sounding_data_dir
    :return:
    """
    variables = ['height', 'temperature', 'dewpoint', 'u_wind', 'v_wind']
    
    # Define levels for interpolation: same as model data, except omitting lowest_p_level
    plevs = [600, 750, 850, 925]
    pres_interp = np.array([p for p in plevs if p <= config['lowest_p_level']])
    
    # Try retrieving the sounding, first checking for existing
    if config['verbose']:
        print('upper_air: retrieving sounding for %s' % datetime.strftime(date, '%Y%m%d%H'))
    nan_sounding = False
    retrieve_sounding = False
    sndg_data_dir = config['Obs']['sounding_data_dir']
    if not(os.path.isdir(sndg_data_dir)):
        os.makedirs(sndg_data_dir)
    sndg_file = '%s/%s_SNDG_%s.pkl' % (sndg_data_dir, station_id, datetime.strftime(date, '%Y%m%d%H'))
    if use_existing:
        try:
            data = read_pkl(sndg_file)
            if config['verbose']:
                print('    Read from file.')
        except:
            retrieve_sounding = True
    else:
        retrieve_sounding = True
    if retrieve_sounding:
        try:
            dset = get_upper_air_data(date, sounding_station_id)
        except:
            # Try again
            try:
                dset = get_upper_air_data(date, sounding_station_id)
            except:
                if use_nan_sounding:
                    if config['verbose']:
                        print('upper_air: warning: unable to retrieve sounding; using nan.')
                    nan_sounding = True
                else:
                    raise ValueError('error retrieving sounding for %s' % date)
    
        # Retrieve pressure for interpolation to fixed levels
        if not nan_sounding:
            pressure = dset.variables['pressure']
            pres = np.array([p.magnitude for p in list(pressure)])  # units are hPa
    
        # Get variables and interpolate; add to dictionary
        data = OrderedDict()
        for var in variables:
            if not nan_sounding:
                var_data = dset.variables[var]
                var_array = np.array([v.magnitude for v in list(var_data)])
                var_interp = interp(pres_interp, pres, var_array)
                data[var] = var_interp.tolist()
            else:
                data[var] = [np.nan] * len(pres_interp)
    
        # Save
        if save and not nan_sounding:
            with open(sndg_file, 'wb') as handle:
                pickle.dump(data, handle, protocol=2)

    return data


def get_obs_hourly(config, station_id, api_dates, vars_api, units):
    """
    Retrieve hourly obs data in a pd dataframe. In order to ensure that there is no missing hourly indices, use
    dataframe.reindex on each retrieved dataframe.
    :param station_id: station ID to obtain data from
    :param api_dates: dates from generate_dates
    :param vars_api: str: string formatted for api call var parameter
    :param units: str: string formatted for api call units parameter
    :return: pd.DataFrame: formatted hourly obs DataFrame
    """
    # Initialize Meso
    m = Meso(token=config['meso_token'])
    if config['verbose']:
        print('get_obs_hourly: MesoPy initialized for station %s' % station_id)

    # Retrieve data
    obs_final = pd.DataFrame()
    for api_date in api_dates:
        if config['verbose']:
            print('get_obs_hourly: retrieving data from %s to %s' % api_date)
        obs = m.timeseries(stid=station_id, start=api_date[0], end=api_date[1], vars=vars_api, units=units,
                           hfmetars='0')
        obspd = pd.DataFrame.from_dict(obs['STATION'][0]['OBSERVATIONS'])

        # Rename columns to requested vars
        obs_var_names = obs['STATION'][0]['SENSOR_VARIABLES']
        obs_var_keys = list(obs_var_names.keys())
        col_names = list(map(''.join, obspd.columns.values))
        for c in range(len(col_names)):
            col = col_names[c]
            for k in range(len(obs_var_keys)):
                key = obs_var_keys[k]
                if col == list(obs_var_names[key].keys())[0]:
                    col_names[c] = key
        obspd.columns = col_names

        # Change datetime column to datetime object
        dateobj = pd.to_datetime(obspd['date_time'])
        obspd['date_time'] = dateobj
        datename = 'date_time'
        obspd = obspd.rename(columns={'date_time': datename})

        # Reformat data into hourly obs
        # Find mode of minute data: where the hourly metars are
        if config['verbose']:
            print('get_obs_hourly: finding METAR observation times...')
        minutes = []
        for row in obspd.iterrows():
            date = row[1][datename]
            minutes.append(date.minute)
        minute_count = np.bincount(np.array(minutes))
        rev_count = minute_count[::-1]
        minute_mode = minute_count.size - rev_count.argmax() - 1

        if config['verbose']:
            print('get_obs_hourly: finding hourly data...')
        obs_hourly = obspd[pd.DatetimeIndex(obspd[datename]).minute == minute_mode]
        obs_hourly.date_time = pd.to_datetime(obs_hourly[datename].values)
        obs_hourly = obs_hourly.set_index(datename)
        if 'precip_accum_one_hour' in vars_api:
            # May not have precip if none is recorded
            try:
                obs_hourly['precip_accum_one_hour'].fillna(0.0, inplace=True)
            except KeyError:
                obs_hourly['precip_accum_one_hour'] = 0.0

        # Need to reorder the column names
        obs_hourly.sort_index(axis=1, inplace=True)

        # Remove any duplicate rows
        obs_hourly = obs_hourly[~obs_hourly.index.duplicated(keep='last')]

        # Re-index by hourly. Fills missing with NaNs.
        expected_start = datetime.strptime(api_date[0], '%Y%m%d%H%M').replace(minute=minute_mode)
        expected_end = datetime.strptime(api_date[1], '%Y%m%d%H%M')
        expected_times = pd.date_range(expected_start, expected_end, freq='H').to_pydatetime()
        obs_hourly = obs_hourly.reindex(expected_times)
        var_list = vars_api.split(',')
        obs_final = pd.concat((obs_final, obs_hourly))

    # Remove any duplicate rows from concatenation
    obs_final = obs_final[~obs_final.index.duplicated(keep='last')]

    return obs_final


def reindex_hourly(df, start, end, interval, end_23z=False, use_rain_max=False):

    def last(values):
        return values.iloc[-1]

    if end_23z:
        new_end = pd.Timestamp(end.to_pydatetime() - timedelta(hours=1))
    else:
        new_end = end
    period = pd.date_range(start, end, freq='%dH' % interval)

    # Create a column with the new index an ob falls into
    if type(df.index.values[0]) == np.int64: #observations from csv file
        df.date_time=np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in df['date_time'].values],dtype='datetime64[s]')
        df.set_index('date_time',inplace=True)
    df['period'] = (df.index.values > period.values[..., np.newaxis]).sum(0)
    df['DateTime'] = df.index.values
    aggregate = OrderedDict()
    col_names = df.columns.values
    for col in col_names:
        if not(col.lower().startswith('precip')) and not(col.lower().startswith('rain')):
            aggregate[col] = last
        else:
            if use_rain_max:
                aggregate[col] = np.max
            else:
                aggregate[col] = np.sum
    df_reindex = df.loc[start:new_end].groupby('period').agg(aggregate)
    try:
        df_reindex = df_reindex.drop('period', axis=1)
    except (ValueError, KeyError):
        pass
    df_reindex = df_reindex.set_index('DateTime')
    return df_reindex


def obs(config, output_files=None, csv_files=None, num_hours=24, interval=3,use_nan_sounding=False, use_existing_sounding=True):
    """
    Generates observation data from MesoWest and UCAR soundings and saves to a file, which can later be retrieved for
    either training data or model run data.
    :param config:
    :param output_files: str: output file path if just one station, or list of output file paths if multiple stations
    :param csv_files: str: path to csv file containing observations if just one station, or list of paths to csv files if multiple stations
    :param num_hours: int: number of hours to retrieve obs
    :param interval: int: retrieve obs every 'interval' hours
    :param use_nan_sounding: bool: if True, uses a sounding of NaNs rather than omitting a day if sounding is missing
    :param use_existing_sounding: bool: if True, preferentially uses saved soundings in sounding_data_dir
    :return:
    """
    if config['multi_stations']: #Train on multiple stations
        station_ids = config['station_id']
        if len(station_ids) != len(output_files): #There has to be the same number of output files as station IDs, so raise error if not
            raise ValueError("There must be the same number of output files as station IDs")
        if len(station_ids) != len(csv_files): #There has to be the same number of output files as station IDs, so raise error if not
            raise ValueError("There must be the same number of csv files as station IDs")
    else:
        station_ids = [config['station_id']]
        if output_files is not None:
            output_files = [output_files]
        if csv_files is not None:
            csv_files = [csv_files]
    
    for i in range(len(station_ids)):
        station_id = station_ids[i]
        if output_files is None:
            output_file = '%s/%s_obs.pkl' % (config['SITE_ROOT'], station_id)
        else:
            output_file = output_files[i]
            
        if csv_files is None:
            csv_file = '%s/%s_obs.csv' % (config['SITE_ROOT'], station_id)
        else:
            csv_file = csv_files[i]
    
        start_date = datetime.strptime(config['data_start_date'], '%Y%m%d') - timedelta(hours=num_hours)
        dates = generate_dates(config)
        api_dates = generate_dates(config, api=True, start_date=start_date)

    
        # Retrieve station data
        if not os.path.exists(csv_file): #no observations saved yet
            # Look for desired variables
            vars_request = []
            vars = ['air_temp', 'altimeter', 'precip_accum_one_hour', 'relative_humidity','wind_speed', 'wind_direction','air_temp_low_6_hour', 'air_temp_high_6_hour', 'precip_accum_six_hour']
            m = Meso(token=config['meso_token'])
            if config['verbose']:
                print('obs: MesoPy initialized for station %s' % config['station_id'])
                print('obs: retrieving latest obs and metadata')
            latest = m.latest(stid=station_id)
            obs_list = list(latest['STATION'][0]['SENSOR_VARIABLES'].keys())
            # Add variables to the api request if they exist
            if config['verbose']:
                print('obs: searching for 6-hourly variables...')
            for var in vars:
                if var in obs_list:
                    if config['verbose']:
                        print('obs: found variable %s, adding to data' % var)
                    vars_request += [var]
    
            # Add variables to the api request
            vars_api = ''
            for var in vars_request:
                vars_api += var + ','
            vars_api = vars_api[:-1]
    
            # Units
            units = 'temp|f,precip|in,speed|kts'
            all_obs_hourly = get_obs_hourly(config, station_id, api_dates, vars_api, units)
            try:
                all_obs_hourly.to_csv(csv_file)
                if config['verbose']:
                    print('obs: saving observations to csv file succeeded')
            except BaseException as e:
                if config['verbose']:
                    print("obs: warning: '%s' while saving observations" % str(e))
            if 'precip_accum_one_hour' in vars_request:
                obs_hourly = all_obs_hourly[['air_temp','altimeter','precip_accum_one_hour','relative_humidity','wind_speed','wind_direction']] #subset of data used as predictors
            else:
                obs_hourly = all_obs_hourly[['air_temp','altimeter','relative_humidity','wind_speed','wind_direction']] #subset of data used as predictors
        else:
            if config['verbose']:
                print('obs: obtaining observations from csv file') 
            all_obs_hourly = pd.read_csv(csv_file)
            vars_request=['air_temp','altimeter','precip_accum_one_hour','relative_humidity','wind_speed', 'wind_direction']
            for var in vars_request[:]: #see if variable is available, and remove from vars_request list if not
                try:
                    obs_hourly = all_obs_hourly[[var]]
                except KeyError: #no such variable, so remove from vars_request list 
                    vars_request.remove(var)
            obs_hourly = all_obs_hourly[['date_time']+vars_request] #subset of data used as predictors
    
        # Retrieve upper-air sounding data
        soundings = OrderedDict()
        if config['Obs']['use_soundings']:
            if config['verbose']:
                print('obs: retrieving upper-air sounding data')
            for date in dates:
                soundings[date] = OrderedDict()
                start_date = date - timedelta(days=1)  # get the previous day's soundings
                for hour in [0, 12]:
                    sounding_date = start_date + timedelta(hours=hour)
                    try:
                        sounding = upper_air(config, station_id, sounding_station_id, sounding_date, use_nan_sounding, use_existing=use_existing_sounding)
                        soundings[date][sounding_date] = sounding
                    except:
                        print('obs: warning: problem retrieving soundings for %s' % datetime.strftime(date, '%Y%m%d'))
                        soundings.pop(date)
                        break
    
        # Create dictionary of days
        if config['verbose']:
            print('obs: converting to output dictionary')
        obs_export = OrderedDict({'SFC': OrderedDict(),
                                  'SNDG': OrderedDict()})
        for date in dates:
            if config['Obs']['use_soundings'] and date not in soundings.keys():
                continue
            # Need to ensure we use the right intervals to have 22:5? Z obs
            start = pd.Timestamp(date - timedelta(hours=num_hours,minutes=-1))
            end = pd.Timestamp(date)
            obs_export['SFC'][date] = reindex_hourly(obs_hourly, start, end, interval,
                                                     end_23z=True).to_dict(into=OrderedDict)
            if config['Obs']['use_soundings']:
                obs_export['SNDG'][date] = soundings[date]
    
        # Export final data
        if config['verbose']:
            print('obs: -> exporting to %s' % output_file)
        with open(output_file, 'wb') as handle:
            pickle.dump(obs_export, handle, protocol=2)

    return

def process(config, obs_list):
    """
    Returns a numpy array of obs for use in mosx_predictors. The first dimension is date; all other dimensions are
    serialized.
    :param config:
    :param obs_list: list of dictionaries of processed obs data for the different stations
    :return:
    """
    # Surface observations
    if config['verbose']:
        print('obs.process: processing array for obs data...')
        
    for i in range(len(obs_list)):
        obs = obs_list[i]
        try:
            sfc = obs['SFC']
        except KeyError:
            sfc = obs[b'SFC']
        num_days = len(sfc.keys())
        variables = sorted(sfc[list(sfc.keys())[0]].keys())
        sfc_array = get_array(sfc)
        sfc_array_r = np.reshape(sfc_array, (num_days, -1))

        # Sounding observations
        if config['Obs']['use_soundings']:
            try:
                sndg_array = get_array(obs['SNDG'])
            except KeyError:
                sndg_array = get_array(obs[b'SNDG'])
            # num_days should be the same first dimension
            sndg_array_r = np.reshape(sndg_array, (num_days, -1))
            obs_one_array = np.hstack((sfc_array_r, sndg_array_r))
            if i == 0: #first station
                obs_array = obs_one_array
            else:
                obs_array = np.hstack((obs_array,obs_one_array))
        else:
            if i == 0: #first station
                obs_array = sfc_array_r
            else:
                obs_array = np.hstack((obs_array,sfc_array_r))
    return obs_array
