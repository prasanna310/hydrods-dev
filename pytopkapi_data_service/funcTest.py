import numpy as np
import os, json
from osgeo import gdal
import createpytopkapiforcingfile
import h5py
import numpy
def get_raster_detail(tif_file):
    dx= None
    ncol =None
    nrow = None
    bands= None
    try:

        from osgeo import gdal
        ds = gdal.Open(tif_file)

        x0, dx, fy, y0, fx, dy = ds.GetGeoTransform()
        ncol= ds.RasterXSize
        nrow = ds.RasterYSize
        bands= ds.RasterCount

        print ('Progress --> Cell size calculated is %s m' % dx)
    except:
        dx = ""
        print ("Either no GDAL, or no tiff file")
    return {'cell_size':dx, 'x':dx, 'ncol':ncol, 'nrow':nrow, 'bands':bands}
# import datetime
from datetime import timedelta, datetime
import pandas as pd
import  json, math
# import ueb_utils

def change_date_from_mmddyyyy_to_yyyyddmm(in_date):
    '''
    :param in_date:         accepts date of formate '01/25/2010'
    :return:                converts the date to formate: '2010-01-25'
    '''
    from datetime import datetime
    in_date_element = datetime.strptime(in_date , '%m/%d/%Y')
    out_date = "%s-%s-%s"%(in_date_element.year, in_date_element.month, in_date_element.day)
    return out_date

def downloadandresampleusgsdischarge(USGS_Gage, begin_date='10/01/2010', end_date='12/30/2010',out_fname='q_obs_cfs.txt',
                                         output_unit='cfs',resampling_time = '1D', resampling_method='mean'):
    """
    Downloads, and then resamples the discharge data from USGS using the url of the format:
    http://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&format=rdb&site_no=10109000&period=&begin_date=2015-10-01&end_date=2015-10-31
    INPUT:
    USGS_Gage :     string, e.g. 10109000
    begin_date=     string, e.g. '10/01/2010'
    end_date=       string, e.g. ''12/30/2010'
    out_fname=      string, e.g. 'Q_cfs.txt'
    output_unit=    string, e.g. 'cfs' or 'cumecs'
    resampling_time=  string, e.g. '1D'
    resampling_method=string, e.g.'mean'
    """

    import urllib
    import pandas as pd

    print ('Input begin date',begin_date)
    begin_date = change_date_from_mmddyyyy_to_yyyyddmm(in_date=begin_date)
    end_date = change_date_from_mmddyyyy_to_yyyyddmm(in_date=end_date)
    print ('Edited begin date', begin_date)
    print ('Required format is yyyy-mm-dd')
    urlString3 = 'http://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&format=rdb&site_no=%s&period=&begin_date=%s&end_date=%s'%(USGS_Gage, begin_date, end_date)

    response = urllib.request.urlopen(urlString3)  # instance of the file from the URL
    html = response.read()                  # reads the texts into the variable html

    print ('Progress --> HTML read for the observed timeseries')
    with open('Q_raw.txt', 'wb') as f:
        f.write(html)

    df = pd.read_csv('Q_raw.txt', delimiter='\t' , skiprows=28, names=['agency_cd', 'USGS_Station_no', 'datatime', 'timezone', 'Q_cfs','Quality'])

    # convert datetime from string to datetime
    df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2],errors='ignore')

    # create a different dataframe with just the values and datetime
    df_datetime_val = df[['datatime', 'Q_cfs']]

    # convert the values to series
    values = []
    dates = []

    # add values to the list a
    multiplier = 1.0
    for v in df_datetime_val.iloc[:,1]:
        if output_unit.lower()=='cumecs' or 'cumec':
            multiplier =  0.028316846592
        values.append(float(v)* multiplier)

    # add datatime to list b
    for v in df_datetime_val.iloc[:, 0]:
        dates.append(v)

    # prepare a panda series
    ts = pd.Series(values, index=dates)

    # resample to daily or whatever
    # ts_mean = ts.resample('1D', how='mean') #or
    # ts_mean = ts.resample('1D').mean()
    ts_mean = ts.resample(resampling_time, how=resampling_method)


    # save
    ts_mean.to_csv(out_fname)
    print ('Progress --> Output creatad for observed file at %s'%out_fname)
    return {'success':True, 'output_file':out_fname}



def read_values_from_results(results=None, outlet_id=None, rain_h5=None, simulation_start_date=None, timestep=24,
                                 output_qsim=None, input_q_obs=None, ):
    '''
    The output file contains array of the format:
            YYYY  MM  DD  hh  mm  q_simulated q_observed
            Both q_simulated and q_observed are in cfs
    :param results:
    :param outlet_id:
    :param simulation_start_date:
    :param timestep:
    :param output_qsim:
    :param input_q_obs:
    :return:
    '''

    # default results location
    if results == None:
        results = "results.h5"
    # output path
    if output_qsim == None:
        output_qsim =  "q_sim_cfs.txt"


    # read h5 file
    f = h5py.File(results, 'r')

    # channel flow
    ndar_Qc_out = f['Channel/Qc_out'][:]
    ar_Qsim = ndar_Qc_out[:, int(outlet_id)]

    # ET actual out
    ET_out_ar = f['ET_out'][:]
    eta_ar = numpy.average(ET_out_ar, axis=1)

    # overland water vol
    Vo = f['Overland']['V_o'][:]
    vo_ar = numpy.average(Vo, axis=1)

    # soil water vol
    V_s = f['Soil']['V_s'][:]
    vs_ar = numpy.average(V_s, axis=1)

    # channel water vol
    V_c = f['Channel']['V_c'][:]
    vc_ar = numpy.average(V_c, axis=1)

    print ('WARNING: Total nan simulated discharge (converted to 0)', numpy.count_nonzero(~numpy.isnan(ar_Qsim)))
    ar_Qsim[numpy.isnan(ar_Qsim)] = 0  # line added

    f.close()

    # create an array with first line as
    # 2011  01  05  0   0   15

    s = datetime.strptime(simulation_start_date, "%m/%d/%Y")
    timestep = timedelta(hours=int(timestep))
    final_array = []
    for i in range(len(
            ar_Qsim) - 1):  # for some reason, simulated values are one more than the observed.. :TODO, fix this
        one_timestep = [s.year, s.month, s.day, s.hour, s.minute, ar_Qsim[i] / 0.028316846592,eta_ar[i], vo_ar[i], vs_ar[i], vc_ar[i]  ]  # with the multiplier, output is now in cfs
        final_array.append(one_timestep)
        s = s + timestep

    # replace nans with 0
    # final_array[numpy.isnan(final_array)] = 0
    # final_array = numpy.nan_to_num(final_array)
    # print (final_array)

    # numpy.savetxt(output_qsim, X=final_array, fmt='%2d %2d %2d %2d %2d %7.3f',delimiter=',')  # was 5.1f
    numpy.savetxt(output_qsim, X=final_array, fmt='%2d,%2d,%2d,%2d,%2d,%7.3f,%7.3f,%7.3f,%7.3f,%7.3f', delimiter=',')  # was 5.1f

    print ('Progress --> Hydrograph results Read')
    return

def array_to_arrayAndDate(array=[1,2,1,2,5,3.,24,5], simulation_start_date='01/01/2010', timestep=24):
    s = datetime.strptime(simulation_start_date, "%m/%d/%Y")
    timestep = timedelta(hours=int(timestep))

    ar_value = []
    ar_date = []
    final_array = []

    for i in range(len(array) ):  # for some reason, simulated values are one more than the observed.. :TODO, fix this
        one_timestep = [s.year, s.month, s.day, s.hour, s.minute,
                        array[i] ]
        ar_value.append( array[i])
        ar_date.append(s)
        final_array.append(one_timestep)
        s = s + timestep

    return  ar_value, ar_date, final_array


def add_an_element_to_json(json_file, elementName, elementValue):
    with open(json_file, 'r') as oldfile:
        old_data = json.load(oldfile)
    with open(json_file, 'w') as updatefile:
        old_data['runs'][-1][elementName] = elementValue
        # old_data[elementName] = elementValue
        json.dump(old_data, updatefile, indent=4)


def vol_balance(result_fname, precip_fname, delta_t, X, cell_id, A):
    import pytopkapi
    import sys
    sys.path.append('/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/PyTOPKAPI/pytopkapi/tests/test_continuity/')

    import continuity_tests as ct

    ini_fname = 'TOPKAPI.ini'
    channel_indices = [cell_id]
    group_name = 'sample_event'

    # hdf_ar = ct.read_hdf5_array(fname, dset_string='ETout')
    precip_vol = ct.compute_precip_volume(precip_fname, group_name, X)                  / A * 1000.
    evapot_vol = ct.compute_evapot_volume(result_fname, X)                              /A *1000.
    # evapot_vol = 'xx'
    evap_from_channelCell_vol = ct.compute_evap_volume(result_fname, channel_indices)   /A *1000.
    storage = ct.compute_storage(result_fname)                                          #/A *1000.
    hdf_ar = ct.compute_channel_runoff(result_fname, delta_t, cell_id)
    overland_runoff = ct.compute_overland_runoff(result_fname, delta_t, cell_id)        /A *1000.
    soil_drainage = ct.compute_soil_drainage(result_fname, delta_t, cell_id)            /A *1000.
    down_drainage = ct.compute_down_drainage(result_fname, delta_t, cell_id)            /A *1000.
    continuity_err = ct.continuity_error(ini_fname, delta_t, cell_id, X, channel_indices)

    print (
    '''
    precip_vol = %s
    evapot_vol = %s
    evap_from_channelCell_vol = %s
    storage = %s
    hdf_ar = %s
    overland_runoff = %s
    soil_drainage = %s
    down_drainage = %s
    continuity_error =%s
    ''' % (
    precip_vol, evapot_vol, evap_from_channelCell_vol, storage, hdf_ar, overland_runoff, soil_drainage, down_drainage,
    continuity_err) )



    # Continuity tests and test environment
    def setup():
        "set up test fixtures"
        os.chdir(os.path.join(os.getcwd(), 'pytopkapi/tests/test_continuity'))

        old_settings = np.seterr(all='ignore')

    def teardown():
        "tear down test fixtures"
        os.chdir('../../..')
        np.seterr(all=old_settings)

    def test_4cell_continuity():
        """Test continuity on the 4 cell catchment (Parak, 2006).

        """
        ini_fname = '4cells.ini'
        delta_t = 3600.0
        cell_id = 3
        X = 1000.0
        channel_indices = [1, 2, 3]

        error, precip_error, stor_error = continuity_error(ini_fname,
                                                           delta_t,
                                                           cell_id, X,
                                                           channel_indices)

        assert precip_error < 2.8e-05
        assert stor_error < 2.1e-03

    def test_d8_continuity():
        """Test continuity on a generic catchment.

        The cell connectivity is D8 and evaporative forcing is applied in addition
        to rainfall.

        """
        ini_fname = 'D8.ini'
        delta_t = 21600.0
        cell_id = 0
        X = 1000.0
        channel_indices = [0, 1, 4]

        error, precip_error, stor_error = continuity_error(ini_fname,
                                                           delta_t,
                                                           cell_id, X,
                                                           channel_indices)

        assert precip_error < 3.6e-04
        assert stor_error < 1.3e-03

    def test_TOPKAPI_continuity(cell_id, cell_size):
        """Test continuity on a sub-catchment of Liebenbergsvlei.

        The cell connectivity is D8, rainfall and evaporative forcing are
        both zero.

        """
        ini_fname = 'TOPKAPI.ini'
        delta_t = 21600.0 * 4  # because our simulation is daily
        # cell_id = 0
        X = 100.0
        channel_indices = [0]

        error, precip_error, stor_error = ct.continuity_error(ini_fname,
                                                           delta_t,
                                                           cell_id, X,
                                                           channel_indices)
        assert precip_error == None
        assert stor_error < 1.5e-05

    # test_TOPKAPI_continuity(cell_id, cell_size=100)

def read_data_from_json(json_fname):
    import json
    import datetime
    with open(json_fname) as json_file:
        data = json.load(json_file)
        try:
            hs_resource_id_created = data['hs_res_id_created']
        except:
            hs_resource_id_created = None

        calib_parameter = None
        numeric_param = None
        hydrograph_series_sim = []
        hydrograph_series_obs = []
        eta = []
        vs = []
        vc = []
        vo = []
        ppt = []

        if 'observed_discharge' in data:
            yr_mon_day_hr_min_discharge_list = data['observed_discharge']

            for yr, mon, day, hr, min, q in yr_mon_day_hr_min_discharge_list:
                date = datetime.datetime(year=int(yr), month=int(mon), day=int(day), hour=int(hr), minute=int(min))
                hydrograph_series_obs.append([date, float(q)])

        if 'ppt' in data:
            yr_mon_day_hr_min_ppt = data['ppt']
            for yr, mon, day, hr, min, val in yr_mon_day_hr_min_ppt:
                date = datetime.datetime(year=int(yr), month=int(mon), day=int(day), hour=int(hr), minute=int(min))
                ppt.append([date, float(val)])

        if 'runs' in data:
            if 'simulated_discharge' in data['runs'][-1]:
                yr_mon_day_hr_min_discharge_list = data['runs'][-1]['simulated_discharge']  # of the last run
                for yr, mon, day, hr, min, q in yr_mon_day_hr_min_discharge_list:
                    date = datetime.datetime(year=int(yr), month=int(mon), day=int(day), hour=int(hr), minute=int(min))
                    hydrograph_series_sim.append([date, float(q)])

            if 'et_a' in data['runs'][-1]:
                yr_mon_day_hr_min_eta = data['runs'][-1]['et_a']
                for yr, mon, day, hr, min, val in yr_mon_day_hr_min_eta:
                    date = datetime.datetime(year=int(yr), month=int(mon), day=int(day), hour=int(hr), minute=int(min))
                    eta.append([date, float(val)])
                eta = [[item[0], 0] if np.isnan(item[-1]) else item for item in eta]  # replace nan to 0

            if 'vc' in data['runs'][-1]:
                yr_mon_day_hr_min_eta = data['runs'][-1]['vc']
                for yr, mon, day, hr, min, val in yr_mon_day_hr_min_eta:
                    date = datetime.datetime(year=int(yr), month=int(mon), day=int(day), hour=int(hr), minute=int(min))
                    vc.append([date, float(val)])
                vc = [[item[0], 0] if np.isnan(item[-1]) else item for item in vc]  # replace nan to 0

            if 'vs' in data['runs'][-1]:
                yr_mon_day_hr_min_eta = data['runs'][-1]['vs']
                for yr, mon, day, hr, min, val in yr_mon_day_hr_min_eta:
                    date = datetime.datetime(year=int(yr), month=int(mon), day=int(day), hour=int(hr), minute=int(min))
                    vs.append([date, float(val)])
                vs = [[item[0], 0] if np.isnan(item[-1]) else item for item in vs]  # replace nan to 0

            if 'vo' in data['runs'][-1]:
                yr_mon_day_hr_min_eta = data['runs'][-1]['vo']
                for yr, mon, day, hr, min, val in yr_mon_day_hr_min_eta:
                    date = datetime.datetime(year=int(yr), month=int(mon), day=int(day), hour=int(hr), minute=int(min))
                    vo.append([date, float(val)])
                vo = [[item[0], 0] if np.isnan(item[-1]) else item for item in vo]  # replace nan to 0

            # read numeric and calib parameters:
            try:
                # calib_parameter= {"fac_l": 1.0, "fac_n_o": 1.0, "fac_n_c": 1.0, "fac_th_s": 1.0, "fac_ks": 1.0},
                # numeric_param= {"pvs_t0": 50, "vo_t0": 750.0, "qc_t0": 0.0, "kc": 1.0},
                calib_parameter = data['runs'][-1]['calib_parameter']
                numeric_param = data['runs'][-1]['numeric_param']

            except:
                calib_parameter = None
                numeric_param = None

    # for i in range(len(data['runs'])):
    #     print 'run:',i, data['runs'][i]['simulated_discharge'][:5]


    return_dict = {'hs_res_id_created': hs_resource_id_created, 'hydrograph_series_obs': hydrograph_series_obs,
                   'hydrograph_series_sim': hydrograph_series_sim,
                   'eta': eta, 'vs': vs, 'vo': vo, 'vc': vc, 'ppt': ppt, 'calib_parameter': calib_parameter,
                   'numeric_param': numeric_param}

    # for key in return_dict:
    #     if key =='hs_res_id_created':
    #         print str( key) +  str( type(return_dict[key])  )
    #
    #     elif key == 'numeric_param' or key == 'calib_parameter':
    #         print str(key) + str(return_dict[key])
    #     else:
    #         print str( key) + " : Length = " + str( len(return_dict[key])  )

    # vol balance in average mm
    # vc[-1]+ vs[-1]+ [vo][-1] + rain_summation - outlet_flow - ET_summation
    cell_area_m2 = 100*100 #500 * 500
    cell_count = 7387 #2224  # 561   # 2224 for 500m, 561 for 1000 meters
    wshed_area_m2 = cell_count * cell_area_m2

    # calculate average % saturation
    # i.e. Effective porosity = SSM - RSM
    cell_param_array = np.genfromtxt(r'C:\Users\Prasanna\Downloads\vol balance calc\cell_param.dat', delimiter=' ')
    SSM = cell_param_array[:, 11]
    RSM = cell_param_array[:, 12]
    eff_sat = SSM - RSM
    eff_sat_avg = eff_sat.mean()

    rain_sum = sum([float(i[-1]) for i in ppt])  # /wshed_area_m2 # in mm
    eta_sum = sum([float(i[-1]) for i in eta])

    vo_0 = numeric_param['vo_t0'] / cell_area_m2 * 1000
    vs_0 = numeric_param['pvs_t0'] / 100 * .4 * cell_area_m2 * 1000
    vc_0 = 0

    vo_0 = vo[0][-1]
    vs_0 = vs[0][-1]
    vc_0 = vc[0][-1]

    del_vo = vo[-1][-1] - vo_0
    del_vc = vc[-1][-1] - vc_0
    del_vs = vs[-1][-1] - vs_0
    q_out_m3 = np.array([float(i[-1]) for i in hydrograph_series_sim]).mean() * len(eta) * 0.02832 * 24 * 3600
    q_out_mm = q_out_m3 / wshed_area_m2 * 1000

    Q_obs_m3 = np.array([float(i[-1]) for i in hydrograph_series_obs]).mean() * len(eta) * 0.02832 * 24 * 3600
    Q_obs_mm = Q_obs_m3 / wshed_area_m2 * 1000
    # print rain_sum ,del_vo ,del_vc , del_vs ,eta_sum , q_out
    # print rain_sum + del_vo + del_vc + del_vs - eta_sum - q_out

    print ( '''
    Volume balance analysis for simulation stored in HydroShare with hs_id: %s
    All the calculations are in mm:

    --initial condition --
    overland cell vol: %s
    soil cell vol    : %s
    channel cell vol  :%s

    --final condition --
    overland cell vol: %s
    soil cell vol    : %s
    channel cell vol  :%s

    -- change in state --
    overland cell vol: %s
    soil cell vol    : %s
    channel cell vol  :%s

    -- Summary --
    Rain sum        : %s
    ETa sum         : %s
    Q_sim           : %s
    Q_obs out       : %s
    -------------
    Qsim/rain coeff : %s
    Qobs/rain coeff : %s

    ''' % (hs_resource_id_created,
           vo_0, vs_0, vc_0,
           vo[-1][-1], vs[-1][-1], vc[-1][-1],
           del_vo, del_vs, del_vc,
           rain_sum, eta_sum, q_out_mm, Q_obs_mm,
           q_out_mm / rain_sum, Q_obs_mm / rain_sum
           ) )

    return return_dict

def area_from_bbox( xmin=None, ymax=None, xmax=None, ymin=None):
    # gives approximate area in mile2
    R = 3959 # in miles
    avg_lat = (ymin+ymax)/2.0

    dx = R * (xmax - xmin)*3.14/180.0
    dy = R  * math.cos(avg_lat *3.14/180.0 )  * (xmax - xmin)*3.14/180.0

    area_in_sqmiles = abs(dx*dy)

    return area_in_sqmiles

# vol_balance(result_fname='/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/sample_simulation/results.h5',
#             precip_fname='/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/sample_simulation/rainfields.h5',
#             delta_t=86400, X=100.0, cell_id=7118, A=73870000.0)


#ueb_utils.run_ueb()

def get_watershed_geojson_from_outlet(x,y, epsg='4326', output_geojson='watershed_streamstat.geojson'):
    str = "streamstatsags.cr.usgs.gov/streamstatsservices/watershed.geojson?rcode=topkapi&xlocation=%s&ylocation=%s&crs=%s&includeparameters=false&includefeatures=false&simplify=true"%(x,y,epsg)
    str = "https://streamstatsags.cr.usgs.gov/streamstatsservices/watershed.geojson?rcode=CA&xlocation=%s&ylocation=%s&crs=%s&includeparameters=false&includeflowtypes=false&includefeatures=false&simplify=true"%(x,y,epsg)

    # str = "https://streamstatsags.cr.usgs.gov/streamstatsservices/watershed.geojson?rcode=NY&xlocation=-74.524&ylocation=43.939&crs=4326&includeparameters=false&includeflowtypes=false&includefeatures=true&simplify=true"

    import requests
    response = requests.get(str)

    if response.status_code == 200:
        print ('Downloading success')
        geojson_string =  response.content
        f = open (output_geojson, 'wb')
        f.write(geojson_string)
        f.close()
    else:
        return {'success':'false'}

    return {'success':'true'}

get_watershed_geojson_from_outlet(x=-122.37113, y=41.5494)