try:
    from osgeo import gdal, osr, ogr
except:
    import gdal, osr, ogr
from gdalconst import *
import shlex
import subprocess
import os
import numpy
import netCDF4
from scipy import interpolate
from .utils import *


def convert_netcdf_units(input_netcdf, output_netcdf, variable_name, variable_new_units=" ", multiplier_factor=1,
                         offset=0):
    """
    does unit conversion for a variable in netcdf file
    :param input_netcdf: input
    :param output_netcdf: output
    :param variable_name: name of variable of interest
    :param variable_new_units: name of the new unit after conversion
    :param multiplier_factor: self explanatory
    :param offset: additive factor
    :return:
    """
    working_dir = os.path.dirname(input_netcdf)
    temp_netcdf = os.path.join(working_dir, "temp_netcdf.nc")
    cmdString = "ncap2 -s\'"+variable_name+"=float("+str(offset) +" + "+str(multiplier_factor)+"*"+variable_name+")\' "+input_netcdf+" "+temp_netcdf

    subprocess_response_dict = call_subprocess(cmdString, 'convert netcdf units')
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict

    cmdString = "ncatted -a units,"+variable_name+",m,c,\'"+variable_new_units+"\' "+temp_netcdf+" "+output_netcdf
    subprocess_response_dict = call_subprocess(cmdString, 'rename netcdf units')
    return subprocess_response_dict


def project_and_resample_Array(input_array, srs_geotrs, srs_proj, Nxin, Nyin, reference_netcdf):  #, output_array):

    #srs_data = gdal.Open(input_raster, GA_ReadOnly)
    #srs_proj = srs_data.GetProjection() #osr.SpatialReference(wkt
    srs_data = gdal.GetDriverByName('MEM').Create('', Nxin, Nyin, 1, gdal.GDT_Float32)
    srs_data.SetGeoTransform(srs_geotrs)
    srs_data.SetProjection(srs_proj)
    srsband = srs_data.GetRasterBand(1)
    srsband.WriteArray(input_array)
    srsband.FlushCache()

    ref_data = gdal.Open(reference_netcdf, GA_ReadOnly)
    ref_proj = ref_data.GetProjection()
    ref_geotrs = ref_data.GetGeoTransform()
    Ncols = ref_data.RasterXSize
    Nrows = ref_data.RasterYSize
    ref_data = None

    out_data = gdal.GetDriverByName('MEM').Create('', Ncols, Nrows, 1, gdal.GDT_Float32)
    out_data.SetGeoTransform(ref_geotrs)
    out_data.SetProjection(ref_proj)

    gdal.ReprojectImage(srs_data,out_data,srs_proj,ref_proj, gdal.GRA_Bilinear )
    output_array = out_data.ReadAsArray()

    srs_data = None
    out_data = None
    return output_array



def project_subset_and_resample_netcdf_to_reference_netcdf(input_netcdf, reference_netcdf, variable_name, output_netcdf):
    """This re-grids a netcdf to target/reference resolution
    Input coordinates are time, y, x
    Warning: Works only if the target boundary is within the input boundary & the coordinates directions are
    the same, i.e. y increasing / decreasing """

    if os.path.exists(output_netcdf):
        os.remove(output_netcdf)

    # Read input geo information
    srs_data = gdal.Open(input_netcdf, GA_ReadOnly)
    srs_geotrs = srs_data.GetGeoTransform()
    Nxin = srs_data.RasterXSize
    Nyin = srs_data.RasterYSize
    srs_proj = srs_data.GetProjection()
    srs_data = None

    #Add dummy dimensions and variables
    # temp_netcdf = output_netcdf
    # cmdString = "ncrename -d x,x_2 -d y,y_2 -v x,x_2 -v y,y_2 -v"+variable_name+","+variable_name+"_2 "+\
    #              input_netcdf+" "+temp_netcdf
    # callSubprocess(cmdString, 'copy netcdf with rename old dimensions')

    temp_netcdf_1 = os.path.join(os.path.dirname(output_netcdf), 'temp_1.nc')
    if os.path.exists(temp_netcdf_1):
        os.remove(temp_netcdf_1)


    cmdString = "ncrename -v x,x_2 -v y,y_2 -v "+variable_name+","+variable_name+"_2 "+\
                 input_netcdf+" "+temp_netcdf_1

    subprocess_response_dict = call_subprocess(cmdString, 'copy netcdf with rename old dimensions')
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict

    temp_netcdf_2 = os.path.join(os.path.dirname(output_netcdf), 'temp_2.nc')
    if os.path.exists(temp_netcdf_2):
        os.remove(temp_netcdf_2)
    cmdString = "ncrename -d x,x_2 -d y,y_2 " + temp_netcdf_1+" "+temp_netcdf_2

    subprocess_response_dict = call_subprocess(cmdString, 'copy netcdf with rename old dimensions')
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict

    ncRef = netCDF4.Dataset(reference_netcdf,"r") # format='NETCDF4')
    xout = ncRef.variables['x'][:]
    yout = ncRef.variables['y'][:]
    ncRef.close()

    ncIn = netCDF4.Dataset(input_netcdf,"r") # format='NETCDF4')
    xin = ncIn.variables['x'][:]
    yin = ncIn.variables['y'][:]

    ncOut = netCDF4.Dataset(temp_netcdf_2,"r+", format='NETCDF4')
    ncOut.createDimension('y',len(yout))
    ncOut.createDimension('x', len(xout))
    dataType = ncIn.variables['x'].datatype
    vardataType = ncIn.variables[variable_name].datatype
    ncOut.createVariable('y',dataType,('y',))
    ncOut.createVariable('x',dataType,('x',))
    ncOut.variables['y'][:] = yout[:]
    ncOut.variables['x'][:] = xout[:]
    ncOut.createVariable(variable_name,vardataType,('time','y','x',))
    #Copy attributes
    varAtts = ncIn.variables[variable_name].ncattrs()
    attDict = dict.fromkeys(varAtts)
    for attName in varAtts:
        attDict[attName] = getattr(ncIn.variables[variable_name],attName)

    ncOut.variables[variable_name].setncatts(attDict)
    xAtts = ncIn.variables['x'].ncattrs()
    attDict = dict.fromkeys(xAtts)
    for attName in xAtts:
        attDict[attName] = getattr(ncIn.variables['x'],attName)
    ncOut.variables['x'].setncatts(attDict)
    yAtts = ncIn.variables['y'].ncattrs()
    attDict = dict.fromkeys(yAtts)
    for attName in yAtts:
        attDict[attName] = getattr(ncIn.variables['y'],attName)
    ncOut.variables['y'].setncatts(attDict)
    ncOut.close()
    #delete the old variables
    cmdString = "ncks -4 -C -x -v x_2,y_2,"+variable_name+"_2 "+\
                 temp_netcdf_2+" "+output_netcdf
    #callSubprocess(cmdString, 'delete old dimensions')
    subprocess_response_dict = call_subprocess(cmdString, 'delete old dimensions')
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict

    #re-open file to write re-gridded data
    ncOut = netCDF4.Dataset(output_netcdf,"r+") #, format='NETCDF4')
    varin = numpy.zeros((len(yin),len(xin)),dtype=vardataType)
    varout = numpy.zeros((len(yout),len(xout)),dtype=vardataType)
    timeLen = len(ncIn.dimensions['time'])
    #yLen = len(yout)
    #xLen = len(xout)
    for tk in range(timeLen):
        varin[:,:] = ncIn.variables[variable_name][tk,:,:]
        #Because gdal tif and Daymet nc y axes directions differ, here array is reversed
        #varin_rev = varin[::-1]
        varout[:,:] = project_and_resample_Array(varin, srs_geotrs, srs_proj, Nxin, Nyin, reference_netcdf)
        ncOut.variables[variable_name][tk,:,:] = varout[:,:]
    ncIn.close()
    ncOut.close()

    subprocess_response_dict['message'] = "project, sunset and resample of netcdf was successful"
    return subprocess_response_dict
    #delete temp netcdf file
    #os.remove(temp_netcdf)

def project_subset_and_resample_daymet_netcdf_to_reference_netcdf(input_netcdf, reference_netcdf, variable_name, output_netcdf):
    """This re-grids a netcdf to target/reference resolution
    Input coordinates are time, y, x
    Warning: Works only if the target boundary is within the input boundary & the coordinates directions are
    the same, i.e. y increasing / decreasing """

    # based on : https://gis.stackexchange.com/questions/168910/gdal-python-extracting-projection-info-from-hdf-file
    if os.path.exists(output_netcdf):
        os.remove(output_netcdf)

    # Read input geo information
    srs_data_ = gdal.Open(input_netcdf, GA_ReadOnly)
    srs_data = gdal.Open(srs_data_.GetSubDatasets()[2][0], GA_ReadOnly)


    srs_geotrs = srs_data.GetGeoTransform()
    Nxin = srs_data.RasterXSize
    Nyin = srs_data.RasterYSize
    srs_proj = srs_data.GetProjection()
    srs_data = None

    #Add dummy dimensions and variables
    # temp_netcdf = output_netcdf
    # cmdString = "ncrename -d x,x_2 -d y,y_2 -v x,x_2 -v y,y_2 -v"+variable_name+","+variable_name+"_2 "+\
    #              input_netcdf+" "+temp_netcdf
    # callSubprocess(cmdString, 'copy netcdf with rename old dimensions')

    temp_netcdf_1 = os.path.join(os.path.dirname(output_netcdf), 'temp_1.nc')
    if os.path.exists(temp_netcdf_1):
        os.remove(temp_netcdf_1)


    cmdString = "ncrename -v x,x_2 -v y,y_2 -v "+variable_name+","+variable_name+"_2 "+\
                 input_netcdf+" "+temp_netcdf_1

    subprocess_response_dict = call_subprocess(cmdString, 'copy netcdf with rename old dimensions')
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict

    temp_netcdf_2 = os.path.join(os.path.dirname(output_netcdf), 'temp_2.nc')
    if os.path.exists(temp_netcdf_2):
        os.remove(temp_netcdf_2)
    cmdString = "ncrename -d x,x_2 -d y,y_2 " + temp_netcdf_1+" "+temp_netcdf_2

    subprocess_response_dict = call_subprocess(cmdString, 'copy netcdf with rename old dimensions')
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict

    ncRef = netCDF4.Dataset(reference_netcdf,"r") # format='NETCDF4')
    xout = ncRef.variables['x'][:]
    yout = ncRef.variables['y'][:]
    ncRef.close()

    ncIn = netCDF4.Dataset(input_netcdf,"r") # format='NETCDF4')
    xin = ncIn.variables['x'][:]
    yin = ncIn.variables['y'][:]

    ncOut = netCDF4.Dataset(temp_netcdf_2,"r+", format='NETCDF4')
    ncOut.createDimension('y',len(yout))
    ncOut.createDimension('x', len(xout))
    dataType = ncIn.variables['x'].datatype
    vardataType = ncIn.variables[variable_name].datatype
    ncOut.createVariable('y',dataType,('y',))
    ncOut.createVariable('x',dataType,('x',))
    ncOut.variables['y'][:] = yout[:]
    ncOut.variables['x'][:] = xout[:]
    ncOut.createVariable(variable_name,vardataType,('time','y','x',))
    #Copy attributes
    varAtts = ncIn.variables[variable_name].ncattrs()
    attDict = dict.fromkeys(varAtts)
    for attName in varAtts:
        attDict[attName] = getattr(ncIn.variables[variable_name],attName)

    ncOut.variables[variable_name].setncatts(attDict)
    xAtts = ncIn.variables['x'].ncattrs()
    attDict = dict.fromkeys(xAtts)
    for attName in xAtts:
        attDict[attName] = getattr(ncIn.variables['x'],attName)
    ncOut.variables['x'].setncatts(attDict)
    yAtts = ncIn.variables['y'].ncattrs()
    attDict = dict.fromkeys(yAtts)
    for attName in yAtts:
        attDict[attName] = getattr(ncIn.variables['y'],attName)
    ncOut.variables['y'].setncatts(attDict)
    ncOut.close()
    #delete the old variables
    cmdString = "ncks -4 -C -x -v x_2,y_2,"+variable_name+"_2 "+\
                 temp_netcdf_2+" "+output_netcdf
    #callSubprocess(cmdString, 'delete old dimensions')
    subprocess_response_dict = call_subprocess(cmdString, 'delete old dimensions')
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict

    #re-open file to write re-gridded data
    ncOut = netCDF4.Dataset(output_netcdf,"r+") #, format='NETCDF4')
    varin = numpy.zeros((len(yin),len(xin)),dtype=vardataType)
    varout = numpy.zeros((len(yout),len(xout)),dtype=vardataType)
    timeLen = len(ncIn.dimensions['time'])
    #yLen = len(yout)
    #xLen = len(xout)
    for tk in range(timeLen):
        varin[:,:] = ncIn.variables[variable_name][tk,:,:]
        #Because gdal tif and Daymet nc y axes directions differ, here array is reversed
        #varin_rev = varin[::-1]
        varout[:,:] = project_and_resample_Array(varin, srs_geotrs, srs_proj, Nxin, Nyin, reference_netcdf)
        ncOut.variables[variable_name][tk,:,:] = varout[:,:]
    ncIn.close()
    ncOut.close()

    subprocess_response_dict['message'] = "project, sunset and resample of netcdf was successful"
    return subprocess_response_dict
    #delete temp netcdf file
    #os.remove(temp_netcdf)



"""char transverse_mercator long form:
{'grid_mapping_name' : "transverse_mercator" ,'longitude_of_central_meridian' : -111. ,'false_easting' : 500000. , ' false_northing' : 0. , 'latitude_of_projection_origin' : 0.  , 'scale_factor_at_central_meridian' : 0.9996,
'longitude_of_prime_meridian' : 0. , 'semi_major_axis' : 6378137. , 'inverse_flattening' : 298.257222101 ,
'spatial_ref' : "PROJCS[\"NAD83 / UTM zone 12N  \",GEOGCS[\"NAD83\",DATUM[\"North_American_Datum_1983\",SPHEROID[\"GRS 1980\",63  78137,298.2572221010002,AUTHORITY[\"EPSG\",\"7019\"]],AUTHORITY[\"EPSG\",\"6269\  "]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433],AUTHORITY[\"EPSG  \",\"4269\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin  \",0],PARAMETER[\"central_meridian\",-111],PARAMETER[\"scale_factor\",0.9996],PA  RAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\"  ,1,AUTHORITY[\"EPSG\",\"9001\"]],AUTHORITY[\"EPSG\",\"26912\"]]" ,
'GeoTransform' : "432404.019091 30 0 4662392.4  4692 0 -30 " }
"""

tmAttributes = {'grid_mapping_name' : 'transverse_mercator' ,'longitude_of_central_meridian' : -111. ,
                            'false_easting' : 500000. , 'false_northing' : 0. , 'latitude_of_projection_origin' : 0.,
                            'scale_factor_at_central_meridian' : 0.9996, 'longitude_of_prime_meridian' : 0. ,
                            'semi_major_axis' : 6378137. , 'inverse_flattening' : 298.257222101 }

def lesser(x,y):
    if (x<y):
        return x
    else:
        return y

def greater(x,y):
    if (x>y):
        return x
    else:
        return y

def project_netCDF_UTM_NAD83(input_netcdf,  output_netcdf, variable_name, utm_zone):
    """ This projection assumes the source spatial reference is known
        i.e. GDAL can read it and recognize it
        variable_name: is the variable of interest in the netCDF file for which the projection is made
    """
    tmAttributes['longitude_of_central_meridian'] = float(6*(utm_zone - 1) + 3 - 180)

    data_set = gdal.Open(input_netcdf, GA_ReadOnly)
    s_srs = data_set.GetProjection()
    data_set = None
    cmdString = "nccopy -k 3 "+input_netcdf+" "+output_netcdf
    subprocess_response_dict = call_subprocess(cmdString, 'copy netcdf')
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict

    ncData = netCDF4.Dataset(output_netcdf, "r+", format='NETCDF4')
    xArray = ncData.variables['x'][:]
    yArray = ncData.variables['y'][:]

    outArrayX = numpy.zeros(len(xArray))
    outArrayY = numpy.zeros(len(yArray))

    for i in range(len(xArray)):
        outArrayX[i], dummyY = project_a_point_UTM(xArray[i], yArray[0], s_srs, utm_zone)

    for j in range(len(yArray)):
        dummyX, outArrayY[j] = project_a_point_UTM(xArray[0], yArray[j], s_srs, utm_zone)

    ncData.variables['x'][:] = outArrayX[:]
    ncData.variables['y'][:] = outArrayY[:]
    ncData.createVariable('transverse_mercator', 'c')
    ncData.variables['transverse_mercator'].setncatts(tmAttributes)
    ncData.variables[variable_name].setncattr('grid_mapping', 'transverse_mercator')

    ncData.close()
    response_dict = {'success': 'True', 'message': 'NetCDF projection was successful'}
    return response_dict

#Logan utm12 nad83 xminymin xmaxymax 432760.510, 4612686.409, 461700.887, 4662453.522
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

def subset_Daymet_netCDF_to_reference_raster(input_netcdf, reference_raster, output_netcdf):
    """ this gives netcdf subset for reference_raster; to get the exact boundary of the
        reference_raster, the input and reference must have same resolution
        The coordinates of the bounding box are projected to the netcdf projection
    To Do: Boundary check-> check if the bounding box of subset raster is
               within the input_netcdf's boundary
    Boundary parameters extracted from reference_Raster
    """
    # based on : https://gis.stackexchange.com/questions/168910/gdal-python-extracting-projection-info-from-hdf-file
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

    data_set_ = gdal.Open(input_netcdf, GA_ReadOnly)
    data_set = gdal.Open(data_set_.GetSubDatasets()[2][0], GA_ReadOnly)
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


#this gives netcdf subset along the time dimension
def get_netCDF_subset_TimeDim(input_netcdf, output_netcdf, time_dim_name, start_time_index, end_time_index):
    #Note: the time bounds are given as index of the time array

    if os.path.exists(output_netcdf):
        os.remove(output_netcdf)

    cmdString = "ncea -4 -d "+time_dim_name+","+str(start_time_index)+","+str(end_time_index)+" "\
                 +input_netcdf+" "+output_netcdf
    return call_subprocess(cmdString, 'subset netcdf')

def reverse_netCDF_yaxis(input_netcdf, output_netcdf):    # resample='near'):
    """
    """
    response_dict = {'success': "False", 'message': "failed to reverse netcdf y-axis"}
    ref_data = gdal.Open(input_netcdf, GA_ReadOnly)
    if not ref_data:
        return response_dict

    ref_proj = ref_data.GetProjection()
    ref_geotrs = ref_data.GetGeoTransform()
    Ncols = ref_data.RasterXSize
    Nrows = ref_data.RasterYSize
    inband = ref_data.GetRasterBand(1)
   # nodata = inband.GetNoDataValue()
    array = inband.ReadAsArray()
    dType = inband.DataType

    out_data = gdal.GetDriverByName('NetCDF').Create(output_netcdf, Ncols, Nrows, 1, dType,["FORMAT=NC4"])
    if not out_data:
        return response_dict

    out_data.SetGeoTransform(ref_geotrs)
    out_data.SetProjection(ref_proj)
    outband = out_data.GetRasterBand(1)
    array_rev = array[::-1]
    #outband.SetNoDataValue(nodata)
    outband.WriteArray(array_rev)
    outband.FlushCache()
    ref_data = None
    out_data = None

    ncIn = netCDF4.Dataset(output_netcdf,"r+")
    if not ncIn:
        return response_dict

    yin = ncIn.variables['y'][:]
    yin_rev = yin[::-1]
    ncIn.variables['y'][:] = yin_rev[:]
    ncIn.close()
    response_dict['success'] = 'True'
    response_dict['message'] = 'reversing of netcdf y-axis was successful'
    return response_dict


def reverse_netCDF_yaxis_and_rename_variable(input_netcdf, output_netcdf, input_varname='Band1',
                                             output_varname='Band1'):
    """
    """
    response_dict = {'success': "False", 'message': "failed to reverse netcdf y-axis"}
    try:
        ncIn = netCDF4.Dataset(input_netcdf,"r") # format='NETCDF4')
        xin = ncIn.variables['x'][:]
        yin = ncIn.variables['y'][:]

        ncOut = netCDF4.Dataset(output_netcdf,"w", format='NETCDF4')
        ncOut.createDimension('y',len(yin))
        ncOut.createDimension('x', len(xin))

        dataType = ncIn.variables['x'].datatype
        if input_varname not in ncIn.variables:
            response_dict['message'] = 'reversing of netcdf y-axis and renaming variable failed. input variable ' \
                                       'does not exist in input netcdf file'
            return response_dict

        vardataType = ncIn.variables[input_varname].datatype

        ncOut.createVariable('y',dataType,('y',))
        ncOut.createVariable('x',dataType,('x',))
        ncOut.variables['y'][:] = yin[::-1]
        ncOut.variables['x'][:] = xin[:]
        ncOut.createVariable(output_varname,vardataType,('y','x',))

        #Copy attributes
        varAtts = ncIn.ncattrs()
        attDict = dict.fromkeys(varAtts)

        for attName in varAtts:
            attDict[attName] = getattr(ncIn,attName)

        ncOut.setncatts(attDict)
        #variable
        varAtts = ncIn.variables[input_varname].ncattrs()
        attDict = dict.fromkeys(varAtts)

        for attName in varAtts:
            attDict[attName] = getattr(ncIn.variables[input_varname],attName)
        ncOut.variables[output_varname].setncatts(attDict)

        #grid mapping var
        gridMapping = attDict['grid_mapping']
        ncOut.createVariable(gridMapping,'c',())
        varAtts = ncIn.variables[gridMapping].ncattrs()
        attDict = dict.fromkeys(varAtts)

        for attName in varAtts:
            attDict[attName] = getattr(ncIn.variables[gridMapping],attName)

        ncOut.variables[gridMapping].setncatts(attDict)
        ncOut.variables[gridMapping].setncatts(attDict)
        #dim variables
        xAtts = ncIn.variables['x'].ncattrs()
        attDict = dict.fromkeys(xAtts)

        for attName in xAtts:
            attDict[attName] = getattr(ncIn.variables['x'],attName)

        ncOut.variables['x'].setncatts(attDict)
        yAtts = ncIn.variables['y'].ncattrs()
        attDict = dict.fromkeys(yAtts)

        for attName in yAtts:
            attDict[attName] = getattr(ncIn.variables['y'],attName)
        ncOut.variables['y'].setncatts(attDict)
        array = ncIn.variables[input_varname][:]
        ncOut.variables[output_varname][:] = array[::-1]

        ncIn.close()
        ncOut.close()
    except:
        response_dict['message'] = 'reversing of netcdf y-axis and renaming variable failed'
        return response_dict

    response_dict['success'] = 'True'
    response_dict['message'] = 'reversing of netcdf y-axis was successful'
    return response_dict


def netCDF_rename_variable(input_netcdf, output_netcdf, input_varname='Band1', output_varname='Band1'):
    """
    """
    response_dict = {'success': "False", 'message': "failed to rename variable"}
    try:
        ncIn = netCDF4.Dataset(input_netcdf,"r") # format='NETCDF4')
        xin = ncIn.variables['x'][:]
        yin = ncIn.variables['y'][:]

        ncOut = netCDF4.Dataset(output_netcdf,"w", format='NETCDF4')
        ncOut.createDimension('y',len(yin))
        ncOut.createDimension('x', len(xin))

        dataType = ncIn.variables['x'].datatype
        if input_varname not in ncIn.variables:
            response_dict['message'] = 'reversing of netcdf y-axis and renaming variable failed. input variable ' \
                                       'does not exist in input netcdf file'
            return response_dict

        vardataType = ncIn.variables[input_varname].datatype

        ncOut.createVariable('y',dataType,('y',))
        ncOut.createVariable('x',dataType,('x',))
        ncOut.variables['y'][:] = yin[:]
        ncOut.variables['x'][:] = xin[:]
        ncOut.createVariable(output_varname,vardataType,('y','x',))

        #Copy attributes
        varAtts = ncIn.ncattrs()
        attDict = dict.fromkeys(varAtts)

        for attName in varAtts:
            attDict[attName] = getattr(ncIn,attName)

        ncOut.setncatts(attDict)
        #variable
        varAtts = ncIn.variables[input_varname].ncattrs()
        attDict = dict.fromkeys(varAtts)

        for attName in varAtts:
            attDict[attName] = getattr(ncIn.variables[input_varname],attName)
        ncOut.variables[output_varname].setncatts(attDict)

        #grid mapping var
        gridMapping = attDict['grid_mapping']
        ncOut.createVariable(gridMapping,'c',())
        varAtts = ncIn.variables[gridMapping].ncattrs()
        attDict = dict.fromkeys(varAtts)

        for attName in varAtts:
            attDict[attName] = getattr(ncIn.variables[gridMapping],attName)

        ncOut.variables[gridMapping].setncatts(attDict)
        ncOut.variables[gridMapping].setncatts(attDict)
        #dim variables
        xAtts = ncIn.variables['x'].ncattrs()
        attDict = dict.fromkeys(xAtts)

        for attName in xAtts:
            attDict[attName] = getattr(ncIn.variables['x'],attName)

        ncOut.variables['x'].setncatts(attDict)
        yAtts = ncIn.variables['y'].ncattrs()
        attDict = dict.fromkeys(yAtts)

        for attName in yAtts:
            attDict[attName] = getattr(ncIn.variables['y'],attName)
        ncOut.variables['y'].setncatts(attDict)
        array = ncIn.variables[input_varname][:]
        ncOut.variables[output_varname][:] = array[:]

        ncIn.close()
        ncOut.close()
    except:
        response_dict['message'] = 'renaming variable failed'
        return response_dict

    response_dict['success'] = 'True'
    response_dict['message'] = 'rename variable was successful'
    return response_dict

"""
resample_netCDF_to_referenceNetCDF('SpawnProj_2010.nc','Spawn17.nc','prcp','Res2.nc')
"""
def resample_netcdf_to_reference_netcdf(input_netcdf, reference_netcdf, variable_name, output_netcdf):
    """This re-grids a netcdf to target/reference netcdf resolution
        the extent and cell size of the output_netcdf will be that of the reference netcdf
        the input netcdf must have the same projection as that of the reference netcdf
    Note: unlike in all other functions, the reference is acutally netcdf (not raster)
    Input coordinates are time, y, x

    Warning: Works only if the target boundary is within the input boundary & the coordinates directions are
    the same, i.e. y increasing / decreasing
    (GDAL generated netcdf have y inverted)
    ToDO: Check GDAL netcdf generation
    ToDO: Check boundary
    """
    #Add dummy dimensions and variables

    #temp_netcdf = "temp"+output_netcdf
    temp_netcdf_2 = os.path.join(os.path.dirname(output_netcdf), 'temp_1.nc')
    if os.path.exists(temp_netcdf_2):
        os.remove(temp_netcdf_2)
        
    cmdString = "ncrename -v x,x_2 -v y,y_2 -v "+variable_name+","+variable_name+"_2 "+\
                 input_netcdf+" "+temp_netcdf_2

    subprocess_response_dict = call_subprocess(cmdString, 'copy netcdf with rename old dimensions')
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict

    # temp_netcdf_2 = os.path.join(os.path.dirname(output_netcdf), 'temp_2.nc')
    # cmdString = "ncrename -d x,x_2 -d y,y_2 " + temp_netcdf_1+" "+temp_netcdf_2
    #
    # subprocess_response_dict = call_subprocess(cmdString, 'copy netcdf with rename old dimensions')
    # if subprocess_response_dict['success'] == 'False':
    #     return subprocess_response_dict

    ncRef = netCDF4.Dataset(reference_netcdf,"r") # format='NETCDF4')
    xout = ncRef.variables['x'][:]
    yout = ncRef.variables['y'][:]
    ncRef.close()

    ncIn = netCDF4.Dataset(input_netcdf,"r") # format='NETCDF4')
    xin = ncIn.variables['x'][:]
    yin = ncIn.variables['y'][:]
    dx = abs(xin[1] - xin[0])       # regular grid so the dx are same over the grid
    dy = abs(yin[1] - yin[0])

    ncOut = netCDF4.Dataset(temp_netcdf_2,"r+", format='NETCDF4')
    ncOut.createDimension('y',len(yout))
    ncOut.createDimension('x', len(xout))
    dataType = ncRef.variables['x'].datatype
    vardataType = ncIn.variables[variable_name].datatype
    ncOut.createVariable('y',dataType,('y',))
    ncOut.createVariable('x',dataType,('x',))
    ncOut.variables['y'][:] = yout[:]
    ncOut.variables['x'][:] = xout[:]
    ncOut.createVariable(variable_name,vardataType,('time','y','x',))
    #Copy attributes
    varAtts = ncIn.variables[variable_name].ncattrs()
    attDict = dict.fromkeys(varAtts)
    for attName in varAtts:
        attDict[attName] = getattr(ncIn.variables[variable_name],attName)
    ncOut.variables[variable_name].setncatts(attDict)
    xAtts = ncIn.variables['x'].ncattrs()
    attDict = dict.fromkeys(xAtts)
    for attName in xAtts:
        attDict[attName] = getattr(ncIn.variables['x'],attName)
    ncOut.variables['x'].setncatts(attDict)
    yAtts = ncIn.variables['y'].ncattrs()
    attDict = dict.fromkeys(yAtts)
    for attName in yAtts:
        attDict[attName] = getattr(ncIn.variables['y'],attName)
    ncOut.variables['y'].setncatts(attDict)
    ncOut.close()
    #delete the old variables
    cmdString = "ncks -4 -C -x -v x_2,y_2,"+variable_name+"_2 "+\
                 temp_netcdf_2+" "+output_netcdf
    # cmdString = "ncks -4 -C -x -v x_2,y_2,"+variable_name+"_2 "+\
    #              output_netcdf
    subprocess_response_dict = call_subprocess(cmdString, 'delete old dimensions')
    if subprocess_response_dict['success'] == 'False':
        return subprocess_response_dict

    #re-open file to write re-gridded data
    try:
        ncOut = netCDF4.Dataset(output_netcdf,"r+") #, format='NETCDF4')
        varin = numpy.zeros((len(yin),len(xin)),dtype=vardataType)
        varout = numpy.zeros((len(yout),len(xout)),dtype=vardataType)
        timeLen = len(ncIn.dimensions['time'])
        yLen = len(yout)
        xLen = len(xout)

        for tk in range(timeLen):
            varin[:,:] = ncIn.variables[variable_name][tk,:,:]
            for yi in range(yLen):
                y1 = int(numpy.floor(abs(yout[yi]-yin[0])/dy))    # abs to make sure
                y2 = y1+1

                for xj in range(xLen):
                    x1 = int(numpy.floor(abs(xout[xj]-xin[0])/dx))
                    x2 =x1+1
                    points = [(yin[y1],xin[x1],varin[y1,x1]),(yin[y1],xin[x2],varin[y1,x2]),(yin[y2],xin[x2],varin[y2,x2]),
                              (yin[y2],xin[x1],varin[y2,x1])]
                    varout[yi,xj] = bilinear_interpolation_with_points_outside_Rectangle(yout[yi], xout[xj], points)
            ncOut.variables[variable_name][tk,:,:] = varout[:,:]
        ncIn.close()
        ncOut.close()
    except:
        subprocess_response_dict['success'] = 'False'
        subprocess_response_dict['message'] = 'error in resampling netcdf file'
        return subprocess_response_dict

    subprocess_response_dict['message'] = "resample of netcdf was successful"
    return subprocess_response_dict

    #delete temp netcdf file
    #os.remove(temp_netcdf)

#This function from: http://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,[(10, 4, 100),(20, 4, 200),(10, 6, 150),(20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        #raise ValueError
        warnString = 'warning! point ('+repr(x)+', '+repr(y)+') not within the rectangle: '+repr(x1)+' '+repr(x2)+', '+\
            repr(y1)+' '+repr(y2)
        raise ValueError(warnString)

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)



#logan boundaries for test  -111.818, 42.113, -111.457, 41.662
def get_netCDFSubset_Geographic(input_netcdf, output_netcdf, lonname, latname, lonmin, lonmax, latmin, latmax):
    #similar to get DEM subset this function gets subset of netcdf in
    #geographic coordinate system; this enables dealing with projection differences between the source netcdf file
    #and the target watershed raster
    #subsettting before projecting to the target reference system avoids dealing with large file
    #however it works only if the input_netcdf has lat/lon coordinate dimenstions
    """ Note: upper left (ul) considered origin, i.e. xmin, ymax
    parameters passed as ulx uly lrx lry (xmin, ymax, xmax, ymin)
    The arguments are in decimal degrees  is in Geographic CS
    latname and lonname: name used to refer the geographic coordinates
    """
    cmdString = "ncea -4 -d "+latname+","+str(latmin)+","+str(latmax)+" -d "+lonname+","+str(lonmin)+","+str(lonmax)+" "\
                 +input_netcdf+" "+output_netcdf
    callSubprocess(cmdString, 'subset netcdf')



#gets the subset of netcdf withing the reference raster
#values are resampled to the resolution of the reference raster
def project_and_subset_netCDF2D(input_netcdf, reference_raster, output_netcdf, resample='bilinear'):
    """
    :param input_raster:
    :param reference_raster:
    :param output_raster:
    :return:
    For images use nearest neighbor interpolation; else pass the method required

    The target extent parameters -te xmin ymin xmax ymax  may be needed to provide a
    region where the destination projection is valid
    projecting a large dataset (e.g. CONUS) into local projections (e.g. NAD 12N) may fail
    because gdal checks the validity of the projection method for the entire region
    """
    data_set = gdal.Open(input_netcdf)
    target_srs = data_set.GetProjection() #osr.SpatialReference(wkt
    #target_srs.ImportFromWkt(data_set.GetPrjectionRef())
    data_set = None
    srsprjFile = 'srsprj.prf'
    prjFilep = open(srsprjFile,'w')
    prjFilep.write(target_srs)
    prjFilep.close()

    data_set = gdal.Open(reference_raster)
    target_srs = data_set.GetProjection() #osr.SpatialReference(wkt
    geo_transform = data_set.GetGeoTransform()
    dx = geo_transform[1]
    dy = geo_transform[5]
    xmin = geo_transform[0]
    ymax = geo_transform[3]
    xmax = xmin + dx * data_set.RasterXSize
    ymin = ymax + dy* data_set.RasterYSize
    data_set = None
    tprjFile = 'destprj.prf'
    prjFilep = open(tprjFile,'w')
    prjFilep.write(target_srs)
    prjFilep.close()

    cmdString = "gdalwarp -s_srs "+srsprjFile+" -t_srs "+tprjFile+" -te "\
                +str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)+" -tr "\
                +str(dx)+" "+str(-1*dy)+" -r "+resample+" -overwrite "+input_netcdf+" tempraster.tif"        #+output_netcdf
    callSubprocess(cmdString, "create intermediate tiff file ")

    cmdString = "gdal_translate -of NetCDF tempraster.tif "+output_netcdf
    callSubprocess(cmdString, "project and clip NetCDF")
     #delete intermediate file
    os.remove('tempRaster.tif')


#This one works only for 2D; with 3D netcdf inputs it images of 2D, band1, band2,...
def project_and_subset_netCDF2D_Image(input_netcdf, reference_raster, output_netcdf):    # resample='near'):
    """
    :param input_raster:
    :param reference_raster:
    :param output_raster:
    :return:
    For images leave the default nearest neighbor interpolation; else pass the method required
    """
    srs_data = gdal.Open(input_netcdf, GA_ReadOnly)
    srs_proj = srs_data.GetProjection() #osr.SpatialReference(wkt
    srs_geotrans = srs_data.GetGeoTransform()

    ref_data = gdal.Open(reference_raster, GA_ReadOnly)
    ref_proj = ref_data.GetProjection()
    ref_geotrs = ref_data.GetGeoTransform()
    Ncols = ref_data.RasterXSize
    Nrows = ref_data.RasterYSize

    out_data = gdal.GetDriverByName('NetCDF').Create(output_netcdf, Ncols, Nrows, 1, GDT_Byte)
    out_data.SetGeoTransform(ref_geotrs)
    out_data.SetProjection(ref_proj)

    gdal.ReprojectImage(srs_data,out_data,srs_proj,ref_proj, GRA_NearestNeighbour)
    out_data = None

def project_a_point_UTM(xcoord, ycoord, s_srs, utmZone):
    s_srsT = osr.SpatialReference()
    s_srsT.ImportFromWkt(s_srs)
    t_srsT = osr.SpatialReference()
    srsString = "+proj=utm +zone=" +str(utmZone)+ " +ellps=GRS80 +datum=NAD83 +units=m "
    t_srsT.ImportFromProj4(srsString)
    #t_srsT.ImportFromWkt(t_srs)
    transform = osr.CoordinateTransformation(s_srsT, t_srsT)
    pointC = ogr.Geometry(ogr.wkbPoint)
    pointC.SetPoint_2D(0,float(xcoord), float(ycoord))
    pointC.Transform(transform)
    xproj = pointC.GetX()
    yproj = pointC.GetY()
    return xproj, yproj

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

def raster_to_netCDF(input_raster, output_netcdf):
    cmdString = "gdal_translate -of netCDF -co \"FORMAT=NC4\" "+input_raster+" "+output_netcdf
    callSubprocess(cmdString, 'raster to netcdf')

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


#This function from: http://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
#_added a line for points outside rectangle to take boundary values
def bilinear_interpolation_with_points_outside_Rectangle(x, y, points):
    '''Interpolate (x,y) from values associated with four points.
    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.
        >>> bilinear_interpolation(12, 5.5,[(10, 4, 100),(20, 4, 200),(10, 6, 150),(20, 6, 300)])
        165.0
    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        #raise ValueError
        warnString = 'warning! point ('+repr(x)+', '+repr(y)+') not within the rectangle: '+repr(x1)+' '+repr(x2)+', '+\
            repr(y1)+' '+repr(y2)
        #raise ValueError(warnString)
        """TZG added this 12.5.14 for inspection """

        #use boundary values
        if (x < x1):
            x = x1
        if(x > x2):
            x = x2
        if(y < y1 ):
            y = y1
        if(y > y2):
            y = y2

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


