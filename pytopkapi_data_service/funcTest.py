import numpy as np
import os, json, sys
from osgeo import gdal, ogr, osr
from gdalconst import *
def get_raster_detail(input_raster, output_json=None):
    minx = None
    miny = None
    maxx = None
    maxy = None
    total_area =  dx = ncol = nrow = bands = None

    try:
        temp_raster = 'temp.tif'
        os.system('gdalwarp -t_srs "+proj=longlat +ellps=WGS84" %s %s -overwrite' % (input_raster, temp_raster))

        ds = gdal.Open(temp_raster)
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + width * gt[4] + height * gt[5]
        maxx = gt[0] + width * gt[1] + height * gt[2]
        maxy = gt[3]

        x0, dx, fy, y0, fx, dy = ds.GetGeoTransform()
        ncol = ds.RasterXSize
        nrow = ds.RasterYSize
        bands = ds.RasterCount

        array = ds.ReadAsArray()
        cell_count = array[array==1].shape[0]
        watershed_area = cell_count*dx*dx  # area in m2

    except Exception as e:
        print (' Progres --> Error: Tiff files contents is not supported. Try another Tiff file', e)
        # return {'success': 'True', 'message':e}

    if not output_json == None:
        JSON_dict = {}
        JSON_dict['minx'] = minx
        JSON_dict['miny'] = miny
        JSON_dict['maxx'] = maxx
        JSON_dict['maxy'] = maxy

        JSON_dict['dx'] = dx
        JSON_dict['cell_size'] = dx
        JSON_dict['ncol'] = ncol
        JSON_dict['nrow'] = nrow
        JSON_dict['bands'] = bands

        # save
        with open(output_json, 'w') as newfile:
            json.dump(JSON_dict, newfile)

    print (' minx, maxx, miny, maxy', minx, maxx, miny, maxy)
    return {'success': 'True', 'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy, 'cell_size': dx, 'dx': dx,
            'ncol': ncol, 'nrow': nrow, 'bands': bands, 'watershed_area':watershed_area, 'cell_count':cell_count}
from datetime import timedelta, datetime

def subset_netCDF_to_reference_raster(input_netcdf, reference_raster, output_netcdf):
    """ this gives netcdf subset for reference_raster; to get the exact boundary of the
        reference_raster, the input and reference must have same resolution
        The coordinates of the bounding box are projected to the netcdf projection
    To Do: Boundary check-> check if the bounding box of subset raster is
               within the input_netcdf's boundary
    Boundary parameters extracted from reference_Raster
    """

    if os.path.exists(output_netcdf):
        os.remove(output_netcdf)


    data_set = gdal.Open(reference_raster, GA_ReadOnly)
    s_srs = data_set.GetProjection()
    geo_transform = data_set.GetGeoTransform()
    # use ulx uly lrx lry
    xmin = geo_transform[0]
    ymax = geo_transform[3]
    dx = geo_transform[1]
    dy = geo_transform[5]

    xmax = xmin + dx * data_set.RasterXSize
    ymin = ymax + dy* data_set.RasterYSize          # dy is -ve
    data_set = None
    data_set = gdal.Open(input_netcdf, GA_ReadOnly)
    t_srs = data_set.GetProjection()
    geo_transform = data_set.GetGeoTransform()
    dxT = geo_transform[1]
    dyT = -1*(geo_transform[5])       #dy is -ve
    data_set = None

    nwX, nwY = project_a_point_srs(xmin,ymax,s_srs,t_srs)
    neX, neY = project_a_point_srs(xmax,ymax,s_srs,t_srs)
    seX, seY = project_a_point_srs(xmax,ymin,s_srs,t_srs)
    swX, swY = project_a_point_srs(xmin, ymax,s_srs,t_srs)

    #take the bigger cell size for buffer
    if(dx > dxT):
        dxT = dx
    if(-1*dy > dyT):
        dyT = -1*dy
    #add a buffer around the boundary
    xmin = lesser(nwX,swX) - 2*dxT
    xmax = greater(seX,neX) + 2*dxT
    ymin = lesser(swY,seY) - 2*dyT
    ymax = greater(nwY,neY) + 2*dyT

    cmdString = "ncea -4 -d y,"+str(ymin)+","+str(ymax)+" -d x,"+str(xmin)+","+str(xmax)+" -O "\
                 +input_netcdf+" "+output_netcdf

    data_set = None
    return call_subprocess(cmdString, 'subset netcdf')

#This concatenates netcdf files along the time dimension
def concatenate_netCDF(input_netcdf1, input_netcdf2, output_netcdf):
    """To  Do: may need to specify output no-data value
    """
    if os.path.exists(output_netcdf):
        os.remove(output_netcdf)

    cmdString = "ncks --mk_rec_dmn time "+input_netcdf1+" tempNetCDF1.nc"
    callSubprocess(cmdString, "intermediate netcdf with record dimension")
    cmdString = "ncks --mk_rec_dmn time "+input_netcdf2+" tempNetCDF2.nc"
    subprocess_response_dict = call_subprocess(cmdString, "intermediate netcdf with record dimension")
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict
    #
    cmdString = "ncrcat -4 tempNetCDF1.nc tempNetCDF2.nc "+output_netcdf
    subprocess_response_dict = call_subprocess(cmdString, "concatenate netcdf files")

    #delete intermediate files
    os.remove('tempNetCDF1.nc')
    os.remove('tempNetCDF2.nc')
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict

    subprocess_response_dict['message'] = "concatenate of netcdf files was successful"
    return subprocess_response_dict

#This combines (stitches) (spatially adjacent) netcdf files accross the spatial/horizontal dimensions
def combineNetCDFs(input_netcdf1, input_netcdf2, output_netcdf):
    """To  Do: may need to specify output no-data value
    """
    cmdString = "gdalwarp -of GTiff -overwrite "+input_netcdf1+" "+input_netcdf2+" tempRaster.tif"  #+output_raster
    callSubprocess(cmdString, "create intermediate raster file")

    cmdString = "gdal_translate -of NetCDF tempRaster.tif "+output_netcdf
    callSubprocess(cmdString, "combine two netcdf files")

    #delete intermediate file
    os.remove('tempRaster.tif')

def project_a_point_srs(xcoord, ycoord, s_srs, t_srs):
    s_srsT = osr.SpatialReference()
    s_srsT.ImportFromWkt(s_srs)
    t_srsT = osr.SpatialReference()
    t_srsT.ImportFromWkt(t_srs)
    transform = osr.CoordinateTransformation(s_srsT, t_srsT)
    pointC = ogr.Geometry(ogr.wkbPoint)
    pointC.SetPoint_2D(0,float(xcoord), float(ycoord))
    pointC.Transform(transform)
    xproj = pointC.GetX()
    yproj = pointC.GetY()
    return xproj, yproj


def callSubprocess(cmdString, debugString):
    cmdargs = shlex.split(cmdString)
    debFile = open('debug_file.txt', 'w')
    debFile.write('Starting %s \n' % debugString)
    retValue = subprocess.call(cmdargs,stdout=debFile)
    if (retValue==0):
        debFile.write('%s Successful\n' % debugString)
        debFile.close()
    else:
        debFile.write('There was error in %s\n' % debugString)
        debFile.close()

def change_date_from_mmddyyyy_to_yyyyddmm(in_date):
    '''
    :param in_date:         accepts date of formate '01/25/2010'
    :return:                converts the date to formate: '2010-01-25'
    '''
    from datetime import datetime
    in_date_element = datetime.strptime(in_date , '%m/%d/%Y')
    out_date = "%s-%s-%s"%(in_date_element.year, in_date_element.month, in_date_element.day)
    return out_date

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

def download_daymet2(input_raster, startYear,endYear):
    import requests

    list_of_years = [startYear+item for item in range(endYear-startYear+1)]
    working_dir = os.path.split(input_raster)[0]
    print (working_dir)
    os.chdir(working_dir)

    in_raster = get_raster_detail(input_raster)
    west, east, south, north = in_raster['minx']-.05, in_raster['maxx']+.05, in_raster['miny']-.02, in_raster['maxy']+.02

    for year in list_of_years:
        for var in ['tmin', 'tmax', 'prcp', 'vp', 'srad']:
            str = 'https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1328/%s/daymet_v3_%s_%s_na.nc4?' \
                  'var=lat&var=lon&var=%s&north=%s&west=%s&east=%s&south=%s&' \
                  'disableProjSubset=on&horizStride=1&time_start=%s-01-01T12:00:00Z&' \
                  'time_end=%s-12-30T12:00:00Z&timeStride=1&accept=netcdf'%(year,var,year ,var,north, west, east, south, year, year)
            print ('TRY_%s_%s.nc'%(var, year))

            response = requests.get(str)
            if response.status_code == 200:
                print ('Downloading success')
                res =  response.content
                f = open ('TRY_%s_%s.nc'%(var, year), 'wb')
                f.write(res)
                f.close()
    return


def download_daymet(input_raster, startYear,endYear ):
    list_of_years = [startYear+item for item in range(endYear-startYear+1)]
    working_dir = os.path.split(input_raster)[0]
    print (working_dir)
    os.chdir(working_dir)

    in_raster = get_raster_detail(input_raster)
    west, east, south, north = in_raster['minx']-.05, in_raster['maxx']+.05, in_raster['miny']-.02, in_raster['maxy']+.02

    # cmd = """for year in {%s..1}; do
    #   for par in tmin tmax prcp vp srad; do
    #
    #     if [ $(( $year % 4 )) -eq 0  ]; then
    #       wget -O ${par}_${year}.nc4 "https://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/1328/${year}/daymet_v3_
    #       ${par}_${year}_na.nc4?var=lat&var=lon&var=${par}&north=%s&west=%s&east=%s&south=%s&horizStride=1
    #       &time_start=${year}-01-01T12:00:00Z&time_end=${year}-12-30T12:00:00Z&timeStride=1&accept=netcdf4"
    #     else
    #       wget -O ${par}_${year}.nc4 "https://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/1328/${year}/daymet_v3_
    #       ${par}_${year}_na.nc4?var=lat&var=lon&var=${par}&north=%s&west=%s&east=%s&south=%s&horizStride=1
    #       &time_start=${year}-01-01T12:00:00Z&time_end=${year}-12-31T12:00:00Z&timeStride=1&accept=netcdf4"
    #     fi
    #   done;
    # done""" % (string_of_years, north, west, east, south, north, west, east, south)
    # print (cmd)
    # os.system(cmd)
    import urllib
    # testfile = urllib.urlretrieve()


    for year in list_of_years:
        for var in ['tmin']: #, 'tmax', 'prcp', 'vp', 'srad']:
            str = 'https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1328/%s/daymet_v3_%s_%s_na.nc4?' \
                  'var=lat&var=lon&var=%s&north=%s&west=%s&east=%s&south=%s&' \
                  'disableProjSubset=on&horizStride=1&time_start=%s-01-01T12:00:00Z&' \
                  'time_end=%s-12-30T12:00:00Z&timeStride=1&accept=netcdf'%(year,var,year ,var,north, west, east, south, year, year)
            print ('wget ' + str)

            str = 'https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1328/2002/daymet_v3_tmin_2002_na.nc4?var=lat&var=lon&var=tmin&north=34.26778098414941&west=-117.28194628018716&east=-117.01684110006099&south=34.07082914625168&disableProjSubset=on&horizStride=1&time_start=2002-01-01T12:00:00Z&time_end=2002-12-30T12:00:00Z&timeStride=1&accept=netcdf'
            os.system('wget '+ str) #'wget -o prcp_example.nc '+


    # url = 'https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1328/%s/daymet_v3_%s_%s_na.nc4'
    # values = {'var': 'lat','var': 'lon','var': 'tmin',
    #           'north': north,
    #           'south':south,
    #           'west':west,
    #           'east':east,
    #           'time_start':'%s-01-01T12:00:00Z'%(2000),
    #           'time_end': '%s-01-01T12:00:00Z' % (2001),
    #           'disableProjSubset':'on',
    #           'horizStride':'1',
    #           'timeStride':'1',
    #           'accept':'netcdf'}
    #
    # data = urllib.parse.urlencode(values)
    # data = data.encode('ascii')  # data should be bytes
    # req = urllib.request.Request(url, data)  # NOTE: you try to read from req
    # with urllib.request.urlopen(req) as response:
    #     the_csv = response.read()
    #
    # for year in list_of_years:
    #     for var in ['tmin']: #, 'tmax', 'prcp', 'vp', 'srad']:
    #         str = 'wget https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1328/%s/daymet_v3_%s_%s_na.nc4?' \
    #               'var=lat&var=lon&var=%s&north=%s&west=%s&east=%s&south=%s&disableProjSubset=on&' \
    #               'horizStride=1&time_start=%s-01-01T12:00:00Z&' \
    #               'time_end=%s-12-30T12:00:00Z&timeStride=1&accept=netcdf'%( year, var,year, var,north, west, east, south, year, year )
    #         print (str)
    #         os.system(str)

    # for year in list_of_years:
    #     for var in ['tmin']: #, 'tmax', 'prcp', 'vp', 'srad']:
    #         str = 'wget -O %s/%s_%s.nc https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1328/%s/daymet_v3_%s_%s_na.nc4?' \
    #               'var=lat&var=lon&var=%s&north=%s&west=%s&east=%s&south=%s&disableProjSubset=on&' \
    #               'horizStride=1&time_start=%s-01-01T12:00:00Z&' \
    #               'time_end=%s-12-30T12:00:00Z&timeStride=1&accept=netcdf'%(working_dir,var, year, year, var,year, var,north, west, east, south, year, year )
    #         print (str)
    #         os.system(str)

# download_daymet2(input_raster='/home/ahmet/ciwater/usu_data_service/workspace/2b8700f2e7ab4e3ea954222cb2166fae/mask.tif',
#                 startYear=2002,endYear=2003 )

if __name__ == "__main__":
    climate_Vars = [ 'tmin', 'tmax', 'srad', 'prcp']  # , 'dayl'
    startYear = 2005
    endYear = 2006
    input_raster = '/home/ahmet/ciwater/usu_data_service/workspace/e8119dc40c1e4576a2879d473fed2b79/mask.tif'
    for var in climate_Vars:
        for year in range(startYear, endYear + 1):
            climatestaticFile1 = os.path.split(input_raster)[0]+'/'+ var + "_" + str(year) + ".nc"  #4
            climateFile1 = var + "__" + str(year) + ".nc"
            Year1sub_request = subset_netCDF_to_reference_raster(input_netcdf=climatestaticFile1,
                                                                 reference_raster=input_raster,
                                                                 output_netcdf=climateFile1)
            concatFile = "conc_" + climateFile1
            if year == startYear:
                concatFile1_url = climateFile1
            else:
                concatFile2_url = climateFile1
                concateNC_request = concatenate_netCDF(input_netcdf1=concatFile1_url,
                                                       input_netcdf2=concatFile2_url,
                                                       output_netcdf=concatFile)
                concatFile1_url = concatFile

