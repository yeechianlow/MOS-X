def convert_MesoWest_data(filename1, filename2, minute):
    '''
    Program to convert data downloaded directly from Synoptic website into csv file suitable for use by MOS-X. Data downloaded must go back to at least January 1, 2010 00 UTC and must have at least the following variables:
    - Temperature
    - Relative Humidity
    - Altimeter
    - Wind Speed
    - Wind Direction
    - 6 Hr Low Temperature
    - 6 Hr High Temperature
    - Precipitation 1hr
    - Precipitation 6hr

    :filename1: filename of data downloaded directly from Synoptic website
    :filename2: filename of processed csv file suitable for use by MOS-X
    :minute: the minute of hourly observations
    '''
    import datetime
    f1 = open(filename1,'r')
    f2 = open(filename2,'w')
    f2.write('date_time,air_temp,air_temp_high_6_hour,air_temp_low_6_hour,altimeter,precip_accum_one_hour,precip_accum_six_hour,relative_humidity,wind_direction,wind_speed\n')
    labels = f1.readline().split(",")
    lines = f1.readlines()
    t = datetime.datetime(2010,1,1,0,minute,0) #first time we want in csv file
    prev_hour = None
    found_first_t = False #keep track if we have found the timestamp equal to the first time we want
    for line in lines:
        line_split = line.split(",")
        date_time = line_split[labels.index('Date_Time')]
        if not found_first_t:
            if t == datetime.datetime.strptime(date_time,'%Y-%m-%dT%H:%M:%SZ'): #found first wanted timestamp
                found_first_t = True
            else:
                continue #skip to next line since we still haven't found the first wanted timestamp
        if (prev_hour != date_time[:-7]) and (date_time[-6:-4] == str(minute)): #only keep hourly observations
            while t != datetime.datetime.strptime(date_time,'%Y-%m-%dT%H:%M:%SZ'): #fill in missing hours with blank data
                print("Missing data for "+str(t))
                f2.write(str(t)+",,,,,,,,,\n")
                t += datetime.timedelta(hours=1)
            if line_split[labels.index('altimeter_set_1')] == '': #missing data:
                altimeter = ''
            else:
                altimeter = str(float(line_split[labels.index('altimeter_set_1')])*3386.39) #convert from inHg to hPa
            new_line = str(datetime.datetime.strptime(date_time,'%Y-%m-%dT%H:%M:%SZ'))+","+line_split[labels.index('air_temp_set_1')]+","+line_split[labels.index('air_temp_high_6_hour_set_1')]+","+line_split[labels.index('air_temp_low_6_hour_set_1')]+","+altimeter+","+line_split[labels.index('precip_accum_one_hour_set_1')]+","+line_split[labels.index('precip_accum_six_hour_set_1')]+","+line_split[labels.index('relative_humidity_set_1')]+","+line_split[labels.index('wind_direction_set_1')]+","+line_split[labels.index('wind_speed_set_1')]+"\n"
            f2.write(new_line)
            t += datetime.timedelta(hours=1)
            prev_hour = date_time[:-7]
    f1.close()
    f2.close()
