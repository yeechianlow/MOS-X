
# Copyright (c) 2018 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Utilities for the MOS-X model.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle


# ==================================================================================================================== #
# Classes
# ==================================================================================================================== #


# ==================================================================================================================== #
# Config functions
# ==================================================================================================================== #

def walk_kwargs(section, key):
    value = section[key]
    try:
        section[key] = int(value)
    except (TypeError, ValueError):
        try:
            section[key] = float(value)
        except (TypeError, ValueError):
            pass


def get_config(config_path):
    """
    Retrieve the config object from config_path.
    :param config_path: str: full path to config file
    :return:
    """
    import configobj
    from validate import Validator

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_spec = '%s/configspec' % dir_path

    try:
        config = configobj.ConfigObj(config_path, configspec=config_spec, file_error=True)
    except IOError:
        try:
            config = configobj.ConfigObj(config_path+'.config', configspec=config_spec, file_error=True)
        except IOError:
            print('Error: unable to open configuration file %s' % config_path)
            raise
    except configobj.ConfigObjError as e:
        print('Error while parsing configuration file %s' % config_path)
        print("*** Reason: '%s'" % e)
        raise

    config.validate(Validator())

    # Make sure site_directory is there
    if config['SITE_ROOT'] == '':
        config['SITE_ROOT'] = '%(MOSX_ROOT)s/site_data'

    # Make sure BUFR parameters have defaults
    if config['BUFR']['bufr_station_id'] == '':
        config['BUFR']['bufr_station_id'] = '%(station_id)s'
    if config['BUFR']['bufr_data_dir'] == '':
        config['BUFR']['bufr_data_dir'] = '%(SITE_ROOT)s/bufkit'
    if config['BUFR']['bufrgruven'] == '':
        config['BUFR']['bufrgruven'] = '%(BUFR_ROOT)s/bufr_gruven.pl'

    # Make sure Obs parameters have defaults
    if config['Obs']['sounding_data_dir'] == '':
        config['Obs']['sounding_data_dir'] = '%(SITE_ROOT)s/soundings'

    # Add in a list for BUFR models
    config['BUFR']['bufr_models'] = []
    for model in config['BUFR']['models']:
        if model.upper() == 'GFS':
            config['BUFR']['bufr_models'].append(['gfs3', 'gfs'])
        else:
            config['BUFR']['bufr_models'].append(model.lower())

    # Convert kwargs, Rain tuning, Ada boosting, and Bootstrapping, if available, to int or float types
    config['Model']['Parameters'].walk(walk_kwargs)
    try:
        config['Model']['Ada boosting'].walk(walk_kwargs)
    except KeyError:
        pass
    try:
        config['Model']['Rain tuning'].walk(walk_kwargs)
    except KeyError:
        pass
    try:
        config['Model']['Bootstrapping'].walk(walk_kwargs)
    except KeyError:
        pass

    return config


# ==================================================================================================================== #
# Utility functions
# ==================================================================================================================== #

def get_object(module_class):
    """
    Given a string with a module class name, it imports and returns the class.
    This function (c) Tom Keffer, weeWX.
    """

    # Split the path into its parts
    parts = module_class.split('.')
    # Strip off the classname:
    module = '.'.join(parts[:-1])
    # Import the top level module
    mod = __import__(module)
    # Recursively work down from the top level module to the class name.
    # Be prepared to catch an exception if something cannot be found.
    try:
        for part in parts[1:]:
            mod = getattr(mod, part)
    except AttributeError:
        # Can't find something. Give a more informative error message:
        raise AttributeError("Module '%s' has no attribute '%s' when searching for '%s'" %
                             (mod.__name__, part, module_class))
    return mod


def generate_dates(config, api=False, start_date=None, end_date=None, api_add_hour=0):
    """
    Returns all of the dates requested from the config. If api is True, then returns a list of (start_date, end_date)
    tuples split by year in strings formatted for the MesoWest API call. If api is False, then returns a list of all
    dates as datetime objects. start_date and end_date are available as options as certain calls require addition of
    some data for prior days.
    :param config:
    :param api: bool: if True, returns dates formatted for MesoWest API call
    :param start_date: str: starting date in config file format (YYYYMMDD)
    :param end_date: str: ending date in config file format (YYYYMMDD)
    :param api_add_hour: int: add this number of hours to the end of the call, useful for getting up to 6Z on last day
    :return:
    """
    if start_date is None:
        start_date = datetime.strptime(config['data_start_date'], '%Y%m%d')
    if end_date is None:
        end_date = datetime.strptime(config['data_end_date'], '%Y%m%d')
    start_dt = start_date
    end_dt = end_date
    if start_dt > end_dt:
        raise ValueError('Start date must be before end date; check MOSX_INFILE.')
    end_year = end_dt.year + 1
    time_added = timedelta(hours=api_add_hour)
    all_dates = []
    if config['is_season']:
        if start_dt > datetime(start_dt.year, end_dt.month, end_dt.day):
            # Season crosses new year
            end_year -= 1
        for year in range(start_dt.year, end_year):
            if start_dt > datetime(start_dt.year, end_dt.month, end_dt.day):
                # Season crosses new year
                year2 = year + 1
            else:
                year2 = year
            if api:
                time_start = datetime(year, start_dt.month, start_dt.day)
                time_end = datetime(year2, end_dt.month, end_dt.day) + time_added
                year_start = datetime.strftime(time_start, '%Y%m%d0000')
                year_end = datetime.strftime(time_end, '%Y%m%d%H00')
                if time_start != time_end: #if there's a non-zero time range 
                    all_dates.append((year_start, year_end))
            else:
                year_dates = pd.date_range(datetime(year, start_dt.month, start_dt.day),
                                           datetime(year2, end_dt.month, end_dt.day), freq='D')
                for date in year_dates:
                    all_dates.append(date.to_pydatetime())

    else:
        if api:
            for year in range(start_dt.year, end_year):
                if year == start_dt.year:
                    time_start = datetime(year, start_dt.month, start_dt.day)
                    year_start = datetime.strftime(time_start, '%Y%m%d0000')
                else:
                    time_start = datetime(year, 1, 1)
                    year_start = datetime.strftime(time_start, '%Y%m%d0000')
                if year == end_dt.year:
                    time_end = datetime(year, end_dt.month, end_dt.day) + time_added
                    year_end = datetime.strftime(time_end, '%Y%m%d%H00')
                else:
                    time_end = datetime(year+1, 1, 1) + time_added
                    year_end = datetime.strftime(time_end, '%Y%m%d%H00')
                if time_start != time_end: #if there's a non-zero time range 
                    all_dates.append((year_start, year_end))
        else:
            pd_dates = pd.date_range(start_dt, end_dt, freq='D')
            for date in pd_dates:
                all_dates.append(date.to_pydatetime())
    return all_dates

def find_matching_dates(bufr, obs, verif, return_data=False):
    """
    Finds dates which match in all three dictionaries. If return_data is True, returns the input dictionaries with only
    common dates retained. verif may be None if running the model.
    :param bufr: dict: dictionary of processed BUFR data if just one station, or list of dictionaries if multiple stations
    :param obs: dict: dictionary of processed OBS data if just one station, or list of dictionaries if multiple stations
    :param verif: dict: dictionary of processed VERIFICATION data if just one station, or list of dictionaries if multiple stations
    :param return_data: bool: if True, returns edited data dictionaries if just one station, or list of dictionaries if multiple stations containing only matching dates' data
    :return: list of dates[, new BUFR, OBS, and VERIF dictionaries or list of dictionaries]
    """
    if type(bufr) != list: #Just one station
        bufr = [bufr]
        obs = [obs]
        if verif is not None:
            verif = [verif]
    for i in range(len(bufr)):
        bufr_one = bufr[i]
        obs_one = obs[i]
        if verif is not None:
            verif_one = verif[i]
        try:
            obs_dates = obs_one['SFC'].keys()
        except KeyError:
            obs_dates = obs_one[b'SFC'].keys()
        if verif is not None:
            verif_dates = verif_one.keys()
        # For BUFR dates, find for all models
        items = list(bufr_one.items())
        for item in items:
            if item[0] == 'SFC' or item[0] == b'SFC':
                bufr_sfc = item[1]
        bufr_dates_list = [bufr_sfc[key].keys() for key in bufr_sfc.keys()]
        bufr_dates = bufr_dates_list[0] 
        for m in range(1, len(bufr_dates_list)):
            bufr_dates = set(bufr_dates).intersection(set(bufr_dates_list[m]))
        if i == 0: #first station, so initialize all matching dates
            all_dates = set(bufr_dates)
        if verif is not None:
            all_dates = all_dates.intersection((set(verif_dates).intersection(set(obs_dates))).intersection(bufr_dates))
        else:
            all_dates = all_dates.intersection(set(obs_dates).intersection(bufr_dates))
    if len(all_dates) == 0:
        raise ValueError('Sorry, no matching dates found in data!')
    print('find_matching_dates: found %d matching dates.' % len(all_dates))
    if return_data:
        for i in range(len(bufr)):
            bufr_one = bufr[i]
            obs_one = obs[i]
            if verif is not None:
                verif_one = verif[i]
            try:
                obs_dates = obs_one['SFC'].keys()
            except KeyError:
                obs_dates = obs_one[b'SFC'].keys()
            if verif is not None:
                verif_dates = verif_one.keys()
            for lev in ['SFC', 'PROF', 'DAY', b'SFC', b'PROF', b'DAY']:
                items = list(bufr_one.items())
                for item in items:
                    if item[0] == lev:
                        bufr_lev = item[1]
                        break
                    else:
                        bufr_lev = None
                if bufr_lev != None:
                    for model in bufr_lev.keys():
                        for date in list(bufr_lev[model].keys()):
                            if date not in all_dates:
                                bufr_lev[model].pop(date, None)
            for date in list(obs_dates):
                if date not in all_dates:
                    try:
                        obs_one['SFC'].pop(date, None)
                    except KeyError:
                        obs_one[b'SFC'].pop(date, None)
                    try:
                        obs_one['SNDG'].pop(date, None)
                    except KeyError:
                        obs_one[b'SNDG'].pop(date, None)
            if verif is not None:
                for date in list(verif_dates):
                    if date not in all_dates:
                        verif_one.pop(date, None)
                verif[i] = verif_one
            bufr[i] = bufr_one
            obs[i] = obs_one
        return bufr, obs, verif, sorted(list(all_dates))
    else:
        return sorted(list(all_dates))
        
def get_array(dictionary):
    """
    Transforms a nested dictionary into an nd numpy array, assuming that each nested sub-dictionary has the same
    structure and that the values elements of the innermost dictionary is either a list or a float value. Function
    _get_array is its recursive sub-function.
    :param dictionary:
    :return:
    """
    dim_list = []
    d = dictionary
    while isinstance(d, dict):
        dim_list.append(len(d.keys()))
        d = list(d.values())[0]
    try:
        dim_list.append(len(d))
    except:
        pass
    out_array = np.full(dim_list, np.nan, dtype=np.float64)
    _get_array(dictionary, out_array)
    return out_array


def _get_array(dictionary, out_array):
    if dictionary == {}:  # in case there's an empty dict for any reason
        return
    if isinstance(list(dictionary.values())[0], list):
        for i, L in enumerate(dictionary.values()):
            out_array[i, :] = np.asarray(L)
    elif isinstance(list(dictionary.values())[0], float):
        for i, L in enumerate(dictionary.values()):
            out_array[i] = L
    else:
        for i, d in enumerate(dictionary.values()):
            _get_array(d, out_array[i, :])

def unpickle(bufr_files, obs_files, verif_files):
    """
    Shortcut function to unpickle bufr, obs, and verif files all at once. verif_files may be None if running the model.
    :param bufr_files: str: full path to pickled BUFR data file if just one station, or list of paths if multiple stations
    :param obs_files: str: full path to pickled OBS data file if just one station, or list of paths if multiple stations
    :param verif_files: str: full path to pickled VERIFICATION data file if just one station, or list of paths if multiple stations
    :return:
    """
    if type(bufr_files) != list: #just one station
        bufr_files = [bufr_files]
        obs_files = [obs_files]
        if verif_files is not None:
            verif_files = [verif_files]
    bufr = []
    obs = []
    verif = []
    for i in range(len(bufr_files)):
        bufr_file = bufr_files[i]
        obs_file = obs_files[i]
        print('util: loading BUFKIT data from %s' % bufr_file)
        bufr.append(read_pkl(bufr_file))
        print('util: loading OBS data from %s' % obs_file)
        obs.append(read_pkl(obs_file))
        if verif_files is not None:
            verif_file = verif_files[i]
            print('util: loading VERIFICATION data from %s' % verif_file)
            verif.append(read_pkl(verif_file))
        else:
            verif = None
    return bufr, obs, verif
    
def get_ghcn_stid(config, stid):
    """
    After code by Luke Madaus, modified by Yeechian Low to extend functionality to stations outside the U.S.
    Gets the GHCN station ID from the 4-letter station ID.
    :param stid: station ID to obtain data for
    """
    stid = stid.upper()
    main_addr = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa'

    site_directory = config['SITE_ROOT']
    # Check to see that ish-history.txt exists
    stations_file = 'isd-history.txt'
    stations_filename = '%s/%s' % (site_directory, stations_file)
    if not os.path.exists(stations_filename):
        print('get_ghcn_stid: downloading site name database')
        try:
            from urllib.request import urlopen
            response = urlopen('%s/%s' % (main_addr, stations_file))
            print('%s/%s' % (main_addr, stations_file))
            f = open(stations_filename, 'wb')
            f.write(response.read())
            f.close()
        except BaseException as e:
            print('get_ghcn_stid: unable to download site name database')
            print("*** Reason: '%s'" % str(e))

    # Now open this file and look for lat/lon coordinates of station
    station_isds = []
    station_isds_99999 = [] #has '99999' in one of the IDs, so not ideal; used only if no station in station_isds
    infile = open(stations_filename, 'r')
    for line in infile:
        linesp = line.split()
        if stid in linesp:
            try: 
                lat = float(line[57:64])
                lon = float(line[65:73])
                if (not linesp[0].startswith('99999') and not linesp[1].startswith('99999')):
                    station_isds.append(int(linesp[1]))
                    break
                else:
                    station_isds_99999.append(int(linesp[1]))
            except:
                continue
    if len(station_isds) == 0 and len(station_isds_99999) == 0: #no station found
        raise ValueError('get_ghcn_stid error: no station found for %s' % stid)
    elif len(station_isds) == 0: #only station with '99999' in one of the IDs is found
        station_isd = station_isds_99999[0]
    else: #use regular station without '99999' in any ID
        station_isd = station_isds[0]   

    if stid[0] == 'K' or stid[0:2] == 'PA' or stid[0:2] == 'PF' or stid[0:2] == 'PG' or stid[0:2] == 'PH' or stid[0:2] == 'PJ' or stid[0:2] == 'PK' or stid[0:2] == 'PM' or stid[0:2] == 'PO' or stid[0:2] == 'PP' or stid[0:2] == 'PW': #U.S. stations
        usw_format = 'USW000%05d'
        print("get_ghcn_stid: found long station id: "+usw_format % station_isd)
        return usw_format % station_isd

    ghcnd_file = 'ghcnd-inventory.txt'
    ghcnd_filename = '%s/%s' % (site_directory, ghcnd_file)
    if not os.path.exists(ghcnd_filename):
        print('get_ghcn_stid: downloading site name database')
        try:
            from urllib.request import urlopen
            response = urlopen('%s/%s/%s' % (main_addr[:-4], "ghcn/daily", ghcnd_file))
            print('%s/%s/%s' % (main_addr[:-4], "ghcn/daily", ghcnd_file))
            f = open(ghcnd_filename, 'wb')
            f.write(response.read())
            f.close()
        except BaseException as e:
            print('get_ghcn_stid: unable to download ghcnd name database')
            print("*** Reason: '%s'" % str(e))

    # Now open ghcnd file and look for full siteid using lat/lon coordinates
    best_stid_ghcnd = None
    best_num_years = -1
    stid_ghcnd = None
    infile = open(ghcnd_filename, 'r')
    for line in infile:
        linesp = line.split()
        try:
            lat_found = round(float(linesp[1]),3)
            lon_found = round(float(linesp[2]),3)
            if (stid_ghcnd is None or stid_ghcnd != linesp[0]) and abs(lat_found-lat) < 0.05 and abs(lon_found-lon) < 0.05: #found station
                stid_ghcnd = linesp[0]
                tmax_years = [0,0]
                tmin_years = [0,0]
                precip_years = [0,0]
            if stid_ghcnd == linesp[0]:
                if linesp[3] == 'TMAX':
                    tmax_years = [int(linesp[4]),int(linesp[5])]
                elif linesp[3] == 'TMIN':
                    tmin_years = [int(linesp[4]),int(linesp[5])]
                elif linesp[3] == 'PRCP':
                    precip_years = [int(linesp[4]),int(linesp[5])]
                if tmax_years[0] != 0 and tmin_years[0] != 0 and precip_years[0] != 0:
                    num_years = min(tmax_years[1],tmin_years[1],precip_years[1])-max(2010,tmax_years[0],tmin_years[0],precip_years[0])
                    if (num_years > best_num_years): #found a better station (more available years)
                        best_stid_ghcnd = stid_ghcnd
                        best_num_years = num_years
        except:
            continue
    if best_stid_ghcnd is None:
        raise ValueError('get_ghcn_stid error: no station found for %s' % stid)
    else:
        print('get_ghcn_stid: found station id: '+str(best_stid_ghcnd))
        return best_stid_ghcnd
# ==================================================================================================================== #
# Conversion functions
# ==================================================================================================================== #

def dewpoint(T, RH):
    """
    Calculates dewpoint from T in Fahrenheit and RH in percent.
    """

    def FtoC(T):
        return (T - 32.) / 9. * 5.

    def CtoF(T):
        return 9. / 5. * T + 32.

    b = 17.67
    c = 243.5  # deg C

    def gamma(T, RH):
        return np.log(RH/100.) + b * T/ (c + T)

    T = FtoC(T)
    TD = c * gamma(T, RH) / (b - gamma(T, RH))
    return CtoF(TD)


def to_bool(x):
    """Convert an object to boolean.
    Examples:
    >>> print to_bool('TRUE')
    True
    >>> print to_bool(True)
    True
    >>> print to_bool(1)
    True
    >>> print to_bool('FALSE')
    False
    >>> print to_bool(False)
    False
    >>> print to_bool(0)
    False
    >>> print to_bool('Foo')
    Traceback (most recent call last):
    ValueError: Unknown boolean specifier: 'Foo'.
    >>> print to_bool(None)
    Traceback (most recent call last):
    ValueError: Unknown boolean specifier: 'None'.
    This function (c) Tom Keffer, weeWX.
    """
    try:
        if x.lower() in ['true', 'yes']:
            return True
        elif x.lower() in ['false', 'no']:
            return False
    except AttributeError:
        pass
    try:
        return bool(int(x))
    except (ValueError, TypeError):
        pass
    raise ValueError("Unknown boolean specifier: '%s'." % x)
    
def read_pkl(filename):
    '''
    Reads a pickle file from filename according to whether we are using Python 2 or 3.
    '''
    try:
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
    except UnicodeDecodeError: #Python 3 requires you to explicitly tell it to use byte encoding when opening Python 2 made pickle files 
        with open(filename, 'rb') as handle:
            data = pickle.load(handle, encoding='bytes')
    return data
