__author__ = 'pkdash'

import json

from django.core.exceptions import ValidationError

from rest_framework import serializers
from usu_data_service.utils import *
from usu_data_service.servicefunctions.static_data import get_static_data_file_path


class InputRasterURLorStaticRequestValidator(serializers.Serializer):
    input_raster = serializers.CharField(required=True)

    def validate_input_raster(self, value):
        # check first if it is a valid url file path
        try:
            validate_url_file_path(value)
        except ValidationError:
            # assume this a static file name
            static_file_path = get_static_data_file_path(value)
            if static_file_path is None:
                raise serializers.ValidationError("Invalid static data file name:%s" % value)

        return value


class rastersubsetRequestValidator(InputRasterURLorStaticRequestValidator):
    xmin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    xmax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)

    input_raster = serializers.CharField(required=False)
    output_raster = serializers.CharField(required=False)

class SubsetDEMRequestValidator (InputRasterURLorStaticRequestValidator):
    xmin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    xmax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    output_raster = serializers.CharField(required=False)

class subsetprismRequestValidator (serializers.Serializer):
    xmin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    xmax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    output_raster = serializers.CharField(required=False)


class SubsetUSGSNEDDEMRequestValidator(serializers.Serializer):
    xmin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    xmax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    output_raster = serializers.CharField(required=True)


class SubsetProjectResampleRasterRequestValidator(SubsetDEMRequestValidator):
    resample = serializers.CharField(min_length=1, required=False, default='near')
    dx = serializers.IntegerField(required=True)
    dy = serializers.IntegerField(required=True)


class SubsetProjectResampleRasterEPSGRequestValidator(SubsetProjectResampleRasterRequestValidator):
    epsg_code = serializers.IntegerField(required=True)


class InputRasterRequestValidator(serializers.Serializer):
    input_raster = serializers.URLField(required=True)


class InputNetCDFURLRequestValidator(serializers.Serializer):
    input_netcdf = serializers.URLField(required=True)


class InputNetCDFURLorStaticRequestValidator(serializers.Serializer):
    input_netcdf = serializers.CharField(required=True)

    def validate_input_netcdf(self, value):
        # check first if it is a valid url file path
        try:
            validate_url_file_path(value)
        except ValidationError:
            # assume this a static file name
            static_file_path = get_static_data_file_path(value)
            if static_file_path is None:
                raise serializers.ValidationError("Invalid static data file name:%s" % value)

        return value


class DelineateWatershedAtXYRequestValidator(serializers.Serializer):
    outlet_point_x = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    outlet_point_y = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    epsg_code = serializers.IntegerField(required=True)
    stream_threshold = serializers.IntegerField(required=True)
    input_DEM_raster = serializers.URLField(required=True)
    output_raster = serializers.CharField(required=True)
    output_outlet_shapefile = serializers.CharField(required=True)


class DelineateWatershedAtShapeFileRequestValidator(serializers.Serializer):
    stream_threshold = serializers.IntegerField(required=True)
    input_DEM_raster = serializers.URLField(required=True)
    input_outlet_shapefile = serializers.URLField(required=True)
    output_raster = serializers.CharField(required=True)
    output_outlet_shapefile = serializers.CharField(required=True)

    def validate_input_outlet_shapefile(self, value):
        try:
            validate_url_file_path(value)
        except NotFound:
            raise serializers.ValidationError("Invalid input outlet shapefile:%s" % value)

        if not value.endswith('.zip'):
            raise serializers.ValidationError(
                "Invalid input outlet shapefile. Shapefile needs to be a zip file:%s" % value)

        return value

    def validate_output_outlet_shapefile(self, value):
        if not value.endswith('.shp'):
            raise serializers.ValidationError(
                "Invalid output outlet shapefile. Shapefile needs to be a .shp file:%s" % value)

        return value


class CreateOutletShapeRequestValidator(serializers.Serializer):
    outletPointX = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    outletPointY = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    output_shape_file_name = serializers.CharField(required=False)


class RasterToNetCDFVariableRequestValidator(InputRasterRequestValidator):
    output_netcdf = serializers.CharField(required=False)
    increasing_x = serializers.BooleanField(required=False)
    increasing_y = serializers.BooleanField(required=False)
    output_varname = serializers.CharField(required=False)


class RasterToNetCDFRequestValidator(InputRasterRequestValidator):
    output_netcdf = serializers.CharField(required=False)


class ComputeRasterAspectRequestValidator(InputRasterRequestValidator):
    output_raster = serializers.CharField(required=False)


class ComputeRasterSlopeRequestValidator(InputRasterRequestValidator):
    output_raster = serializers.CharField(required=False)


class ReverseNetCDFYaxisRequestValidator(InputNetCDFURLRequestValidator):
    output_netcdf = serializers.CharField(required=False)


class ConvertNetCDFUnitsRequestValidator(InputNetCDFURLRequestValidator):
    output_netcdf = serializers.CharField(required=False)
    variable_name = serializers.CharField(required=True)
    variable_new_units = serializers.CharField(required=False)
    multiplier_factor = serializers.DecimalField(required=False, default=1.0, max_digits=12, decimal_places=8)
    offset = serializers.DecimalField(required=False, default=0, max_digits=12, decimal_places=8)


class ReverseNetCDFYaxisAndRenameVariableRequestValidator(ReverseNetCDFYaxisRequestValidator):
    input_varname = serializers.CharField(required=False)
    output_varname = serializers.CharField(required=False)


class NetCDFRenameVariableRequestValidator(ReverseNetCDFYaxisAndRenameVariableRequestValidator):
    pass


class ProjectRasterRequestValidator(InputRasterRequestValidator):
    utmZone = serializers.IntegerField(required=True)
    output_raster = serializers.CharField(required=False)


class CombineRastersRequestValidator(serializers.Serializer):
    input_raster1 = serializers.URLField(required=True)
    input_raster2 = serializers.URLField(required=True)
    output_raster = serializers.CharField(required=False)


class ResampleRasterRequestValidator(InputRasterRequestValidator):
    dx = serializers.IntegerField(required=True)
    dy = serializers.IntegerField(required=True)
    # TODO: may be this should be a choice type field
    resample = serializers.CharField(min_length=1, required=False, default='near')
    output_raster = serializers.CharField(required=False)


class ProjectResampleRasterRequestValidator(ResampleRasterRequestValidator):
    pass


class ProjectResampleRasterUTMRequestValidator(ProjectResampleRasterRequestValidator):
    utm_zone = serializers.IntegerField(required=True)


class ProjectResampleRasterEPSGRequestValidator(ProjectResampleRasterRequestValidator):
    epsg_code = serializers.IntegerField(required=True)


class SubsetRasterToReferenceRequestValidator(InputRasterRequestValidator):
    reference_raster = serializers.URLField(required=True)
    output_raster = serializers.CharField(required=False)


class SubsetNetCDFToReferenceRequestValidator(InputNetCDFURLorStaticRequestValidator):
    reference_raster = serializers.URLField(required=True)
    output_netcdf = serializers.CharField(required=False)


class ProjectNetCDFRequestValidator(InputNetCDFURLRequestValidator):
    utm_zone = serializers.IntegerField(required=True)
    output_netcdf = serializers.CharField(required=False)
    variable_name = serializers.CharField(required=True)


class SubsetNetCDFByTimeDimensionRequestValidator(InputNetCDFURLorStaticRequestValidator):
    time_dim_name = serializers.CharField(required=True)
    start_time_index = serializers.IntegerField(required=True)
    end_time_index = serializers.IntegerField(required=True)
    output_netcdf = serializers.CharField(required=False)

    def validate_start_time_index(self, value):
        if value < 0:
            raise serializers.ValidationError(
                "Invalid start_time_index value:%s. It must be a positive integer." % value)
        return value

    def validate_end_time_index(self, value):
        if value < 0:
            raise serializers.ValidationError("Invalid end_time_index value:%s. It must be a positive integer." % value)
        return value

    def validate(self, data):
        """
        Check that the start_time_index is before the end_time_index.
        """
        if data['start_time_index'] > data['end_time_index']:
            raise serializers.ValidationError("start time index must be a value less than the end time index")

        return data


class ResampleNetCDFRequestValidator(InputNetCDFURLRequestValidator):
    reference_netcdf = serializers.URLField(required=True)
    output_netcdf = serializers.CharField(required=False)
    variable_name = serializers.CharField(required=True)


class ConcatenateNetCDFRequestValidator(serializers.Serializer):
    input_netcdf1 = serializers.URLField(required=True)
    input_netcdf2 = serializers.URLField(required=True)
    output_netcdf = serializers.CharField(required=False)


class ProjectSubsetResampleNetCDFRequestValidator(ResampleNetCDFRequestValidator):
    pass


class ProjectClipRasterRequestValidator(InputRasterURLorStaticRequestValidator):
    output_raster = serializers.CharField(required=False)
    reference_raster = serializers.CharField(required=True)


class GetCanopyVariablesRequestValidator(serializers.Serializer):
    in_NLCDraster = serializers.URLField(required=True)
    out_ccNetCDF = serializers.CharField(required=False)
    out_hcanNetCDF = serializers.CharField(required=False)
    out_laiNetCDF = serializers.CharField(required=False)


class GetCanopyVariableRequestValidator(serializers.Serializer):
    in_NLCDraster = serializers.URLField(required=True)
    output_netcdf = serializers.CharField(required=True)
    variable_name = serializers.CharField(required=True)

    def validate_variable_name(self, value):
        if value not in ('cc', 'hcan', 'lai'):
            raise serializers.ValidationError("Invalid canopy variable name:%s. " % value)
        return value


class ProjectShapeFileUTMRequestValidator(serializers.Serializer):
    input_shape_file = serializers.URLField(required=True)
    output_shape_file = serializers.CharField(required=False)
    utm_zone = serializers.IntegerField(required=True)


class ProjectShapeFileEPSGRequestValidator(serializers.Serializer):
    input_shape_file = serializers.URLField(required=True)
    output_shape_file = serializers.CharField(required=False)
    epsg_code = serializers.IntegerField(required=True)


class ZipMyFilesRequestValidator(serializers.Serializer):
    file_names = serializers.CharField(min_length=5, required=True)
    zip_file_name = serializers.CharField(min_length=5, required=True)

    def validate_file_names(self, value):
        file_names = value.split(',')
        if len(file_names) == 0:
            raise serializers.ValidationError("No file name provided to be zipped")

        for file_name in file_names:
            if not validate_file_name(file_name):
                raise serializers.ValidationError("%s is not a valid file name" % file_name)

        return file_names

    def validate_zip_file_name(self, value):
        if not value.endswith('.zip'):
            raise serializers.ValidationError("%s is not a valid zip file name" % value)

        if not validate_file_name(value):
            raise serializers.ValidationError("%s is not a valid file name" % value)

        return value


class HydroShareCreateResourceRequestValidator(serializers.Serializer):
    hs_username = serializers.CharField(min_length=1, required=True)
    hs_password = serializers.CharField(min_length=1, required=True)
    file_name = serializers.CharField(min_length=5, required=True)
    resource_type = serializers.CharField(min_length=5, required=True)
    title = serializers.CharField(min_length=5, max_length=200, required=False)
    abstract = serializers.CharField(min_length=5, required=False)
    keywords = serializers.CharField(required=False)
    metadata = serializers.CharField(required=False)

    def validate_keywords(self, value):
        if value:
            kws = value.split(',')
            if len(kws) == 0:
                raise serializers.ValidationError("%s must be a comma separated string" % value)
            return kws

    def validate_metadata(self, value):
        if value:
            try:
                json.loads(value)
            except Exception:
                raise serializers.ValidationError("%s must be a valid json string" % value)

        return value


class DownloadStreamflowRequestValidator(serializers.Serializer):
    USGS_gage = serializers.CharField(required=True)
    Start_Year = serializers.IntegerField(required=True)
    End_Year = serializers.IntegerField(required=True)
    output_streamflow = serializers.CharField(required=False)


class GetHydrogateResultFileRequestValidator(serializers.Serializer):
    result_file_name = serializers.CharField(min_length=10, required=True)
    save_as_file_name = serializers.CharField(min_length=5, required=True)

    def validate_result_file_name(self, value):
        if not value.endswith('.zip'):
            raise serializers.ValidationError("%s must be a file name ending with .zip" % 'result_file_name')
        return value

    def validate_save_as_file_name(self, value):
        if not value.endswith('.zip'):
            raise serializers.ValidationError("%s must be a file name ending with .zip" % 'save_as_file_name')

        return value


class helloworldvalidator(serializers.Serializer):
    a = serializers.IntegerField(required=False)
    b = serializers.IntegerField(required=False)


class DownloadClimatedataRequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)

    Start_Year = serializers.IntegerField(required=True)
    End_Year = serializers.IntegerField(required=True)
    output_rainfile = serializers.CharField(required=True)
    output_temperaturefile = serializers.CharField(required=True)
    output_cliparfile = serializers.CharField(required=True)
    output_gagefile = serializers.CharField(required=True)

    def validate_output_gagefile(self, value):
        if not value.endswith('.shp'):
            raise serializers.ValidationError(
                "Invalid output outlet shapefile. Shapefile needs to be a .shp file:%s" % value)

        return value


class DownloadSoildataRequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)
    output_f_file = serializers.CharField(required=True)
    output_dth1_file = serializers.CharField(required=True)
    output_dth2_file = serializers.CharField(required=True)
    output_k_file = serializers.CharField(required=True)
    output_psif_file = serializers.CharField(required=True)
    output_sd_file = serializers.CharField(required=True)
    output_tran_file = serializers.CharField(required=True)


class WatershedDelineationdataRequestValidator(serializers.Serializer):
    DEM_Raster = serializers.URLField(required=True)
    Outlet_shapefile = serializers.URLField(required=True)
    Src_threshold = serializers.IntegerField(required=True)
    Min_threshold = serializers.IntegerField(required=True)
    Max_threshold = serializers.IntegerField(required=True)
    Number_threshold = serializers.IntegerField(required=True)

    output_pointoutletshapefile = serializers.CharField(required=False)
    output_watershedfile = serializers.CharField(required=False)
    output_treefile = serializers.CharField(required=False)
    output_coordfile = serializers.CharField(required=False)
    output_streamnetfile = serializers.CharField(required=False)
    output_slopareafile = serializers.CharField(required=False)
    output_distancefile = serializers.CharField(required=False)

    def validate_Outlet_shapefile(self, value):
        try:
            validate_url_file_path(value)
        except NotFound:
            raise serializers.ValidationError("Invalid input outlet shapefile:%s" % value)

        if not value.endswith('.zip'):
            raise serializers.ValidationError(
                "Invalid input outlet shapefile. Shapefile needs to be a zip file:%s" % value)

        return value

    def validate_output_pointoutletshapefile(self, value):
        if not value.endswith('.shp'):
            raise serializers.ValidationError(
                "Invalid output outlet shapefile. Shapefile needs to be a .shp file:%s" % value)

        return value

    def validate_output_streamnetfile(self, value):
        if not value.endswith('.shp'):
            raise serializers.ValidationError(
                "Invalid output outlet shapefile. Shapefile needs to be a .shp file:%s" % value)

        return value


class ReachLinkdataRequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)
    DEM_Raster = serializers.URLField(required=True)
    treefile = serializers.URLField(required=True)
    coordfile = serializers.URLField(required=True)

    output_reachfile = serializers.CharField(required=True)
    output_reachareafile = serializers.CharField(required=True)
    output_nodefile = serializers.CharField(required=True)
    output_rchpropertiesfile = serializers.CharField(required=True)


class dist_wetness_distributiondataRequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)
    SaR_Raster = serializers.URLField(required=True)
    Dist_Raster = serializers.URLField(required=True)
    output_distributionfile = serializers.CharField(required=True)


class getprismrainfalldataRequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)
    output_raster = serializers.CharField(required=True)


class createrainweightdataRequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)
    Rain_gauge_shapefile = serializers.URLField(required=True)
    nodelink_file = serializers.URLField(required=True)
    annual_rainfile = serializers.URLField(required=True)
    output_rainweightfile = serializers.CharField(required=True)


class createbasinparameterdataRequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)
    DEM_Raster = serializers.URLField(required=True)
    f_raster = serializers.URLField(required=True)
    k_raster = serializers.URLField(required=True)
    dth1_raster = serializers.URLField(required=True)
    dth2_raster = serializers.URLField(required=True)
    sd_raster = serializers.URLField(required=True)
    tran_raster = serializers.URLField(required=True)
    psif_raster = serializers.URLField(required=True)
    lulc_raster = serializers.URLField(required=True)
    lutlc = serializers.URLField(required=True)
    lutkc = serializers.URLField(required=True)
    parameter_specficationfile = serializers.URLField(required=True)

    nodelinksfile = serializers.URLField(required=True)
    output_basinfile = serializers.CharField(required=True)


class createlatlonfromxydataRequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)
    output_latlonfromxyfile = serializers.CharField(required=True)


class createparmfiledataRequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)
    output_parspcfile = serializers.CharField(required=True)


class getlanduselandcoverdataRequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)
    output_LULCRaster = serializers.CharField(required=True)


# Prasanna added (for pytopkapi)

class downloadglobalDEMRequestValidator(serializers.Serializer):
    xmin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    xmax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    output_raster = serializers.CharField(required=False)

class rastersubset2RequestValidator(InputRasterURLorStaticRequestValidator):
    xmin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    xmax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymin = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    ymax = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    cell_size  = serializers.DecimalField(required=False, max_digits=12, decimal_places=8)
    input_raster = serializers.CharField(required=False)
    output_raster = serializers.CharField(required=False)

class ReclassifyRasterWithLUTdataRequestValidator(serializers.Serializer):
    delimiter = serializers.CharField(required=False)
    input_raster = serializers.URLField(required=True)
    LUT = serializers.URLField(required=False)
    output_raster = serializers.CharField(required=False)
    # reclassify_overland_or_stream = serializers.CharField(required=False)

class delineatewatershedtogetcompleterastersetValidator(serializers.Serializer):
    # outlet_point_x = serializers.DecimalField(required=False, max_digits=12, decimal_places=8)
    # outlet_point_y = serializers.DecimalField(required=False, max_digits=12, decimal_places=8)
    # epsg_code = serializers.IntegerField(required=False)
    stream_threshold = serializers.IntegerField(required=True)
    input_DEM_raster = serializers.URLField(required=True)
    input_outlet_shapefile = serializers.URLField(required=True)
    output_raster = serializers.CharField(required=True)
    output_outlet_shapefile = serializers.CharField(required=True)

    output_fill_raster = serializers.CharField(required=False)
    output_slope_raster = serializers.CharField(required=False)
    output_flow_direction_raster = serializers.CharField(required=False)
    output_contributing_area_raster = serializers.CharField(required=False)
    output_accumulated_stream_raster = serializers.CharField(required=False)
    output_stream_raster = serializers.CharField(required=False)

    output_geojson = serializers.CharField(required=False)
    output_shapefile = serializers.CharField(required=False)
    # output_mannings_n_stream_raster = serializers.CharField(required=False)



class runpytopkapi2validator(serializers.Serializer):
    user_name = serializers.CharField(required=False)
    simulation_name = serializers.CharField(required=False)
    simulation_start_date = serializers.CharField(required=False)
    simulation_end_date = serializers.CharField(required=False)
    USGS_gage = serializers.IntegerField(required=False)

    threshold = serializers.IntegerField(required=False)
    cell_size = serializers.FloatField(required=True)
    timestep = serializers.IntegerField(required=False)

    outlet_y = serializers.FloatField(required=True)
    outlet_x = serializers.FloatField(required=True)
    box_topY = serializers.FloatField(required=True)
    box_bottomY =serializers.FloatField(required=True)
    box_rightX = serializers.FloatField(required=True)
    box_leftX = serializers.FloatField(required=True)
    timeseries_source = serializers.CharField(required=False)
    model_engine = serializers.CharField(required=False)

    output_zip = serializers.CharField(required=False)

class runpytopkapi3validator(serializers.Serializer):
    user_name = serializers.CharField(required=False)
    simulation_name = serializers.CharField(required=False)
    simulation_start_date = serializers.CharField(required=False)
    simulation_end_date = serializers.CharField(required=False)
    USGS_gage = serializers.IntegerField(required=False)

    threshold = serializers.IntegerField(required=False)
    cell_size = serializers.CharField(required=False)
    timestep = serializers.IntegerField(required=False)

    mask_fname = serializers.URLField(required=True)
    # channel_manning_fname = serializers.URLField(required=False)
    overland_manning_fname= serializers.URLField(required=False)
    hillslope_fname =serializers.URLField(required=False)
    dem_fname = serializers.URLField(required=False)
    channel_network_fname= serializers.URLField(required=False)
    flowdir_fname = serializers.URLField(required=False)
    pore_size_dist_fname=serializers.URLField(required=False)
    bubbling_pressure_fname=serializers.URLField(required=False)
    resid_moisture_content_fname = serializers.URLField(required=False)
    sat_moisture_content_fname =serializers.URLField(required=False)
    conductivity_fname =serializers.URLField(required=False)
    soil_depth_fname =serializers.URLField(required=False)

    # rain_fname =serializers.URLField(required=False)
    # et_fname= serializers.URLField(required=False)
    # runoff_fname = serializers.URLField(required=False)
    # outlet_x = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # outlet_y = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # box_topY = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # box_bottomY = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # box_rightX = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # box_leftX = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # timeseries_source = serializers.CharField(required=False)
    # model_engine = serializers.CharField(required=False)

    output_txt = serializers.CharField(required=False)

class runpytopkapi4validator(serializers.Serializer):
    user_name = serializers.CharField(required=False)
    simulation_name = serializers.CharField(required=False)
    simulation_start_date = serializers.CharField(required=False)
    simulation_end_date = serializers.CharField(required=False)
    USGS_gage = serializers.IntegerField(required=False)

    threshold = serializers.IntegerField(required=False)   #IntegerField
    cell_size = serializers.CharField(required=False)
    timestep = serializers.IntegerField(required=False)

    mask_fname = serializers.URLField(required=True)
    # channel_manning_fname = serializers.URLField(required=False)
    overland_manning_fname= serializers.URLField(required=False)
    hillslope_fname =serializers.URLField(required=False)
    dem_fname = serializers.URLField(required=False)
    channel_network_fname= serializers.URLField(required=False)
    flowdir_fname = serializers.URLField(required=False)
    pore_size_dist_fname=serializers.URLField(required=False)
    bubbling_pressure_fname=serializers.URLField(required=False)
    resid_moisture_content_fname = serializers.URLField(required=False)
    sat_moisture_content_fname =serializers.URLField(required=False)
    conductivity_fname =serializers.URLField(required=False)
    soil_depth_fname =serializers.URLField(required=False)

    # rain_fname =serializers.URLField(required=False)
    # et_fname= serializers.URLField(required=False)
    # runoff_fname = serializers.URLField(required=False)
    # outlet_x = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # outlet_y = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # box_topY = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # box_bottomY = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # box_rightX = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # box_leftX = serializers.DecimalField(required=True, max_digits=12, decimal_places=8)
    # timeseries_source = serializers.CharField(required=False)
    # model_engine = serializers.CharField(required=False)

    output_hs_rs_id_txt = serializers.CharField(required=False)
    output_q_sim_txt = serializers.CharField(required=False)


class runpytopkapi5validator(serializers.Serializer):
    user_name = serializers.CharField(required=False)
    simulation_name = serializers.CharField(required=False)
    simulation_start_date = serializers.CharField(required=False)
    simulation_end_date = serializers.CharField(required=False)
    USGS_gage = serializers.CharField(required=False)

    threshold = serializers.IntegerField(required=False)   #IntegerField
    cell_size = serializers.CharField(required=False)
    timestep = serializers.IntegerField(required=False)

    mask_fname = serializers.URLField(required=True)
    # channel_manning_fname = serializers.URLField(required=False)
    overland_manning_fname= serializers.URLField(required=False)
    hillslope_fname =serializers.URLField(required=False)
    dem_fname = serializers.URLField(required=False)
    channel_network_fname= serializers.URLField(required=False)
    flowdir_fname = serializers.URLField(required=False)
    pore_size_dist_fname=serializers.URLField(required=False)
    bubbling_pressure_fname=serializers.URLField(required=False)
    resid_moisture_content_fname = serializers.URLField(required=False)
    sat_moisture_content_fname =serializers.URLField(required=False)
    conductivity_fname =serializers.URLField(required=False)
    soil_depth_fname =serializers.URLField(required=False)

    rain_fname =serializers.URLField(required=False)
    et_fname= serializers.URLField(required=False)

    output_response_txt = serializers.CharField(required=False)

class runpytopkapi6validator(serializers.Serializer):
    user_name = serializers.CharField(required=False)
    simulation_name = serializers.CharField(required=False)
    simulation_start_date = serializers.CharField(required=False)
    simulation_end_date = serializers.CharField(required=False)
    USGS_gage = serializers.CharField(required=False)

    threshold = serializers.IntegerField(required=False)   #IntegerField
    cell_size = serializers.CharField(required=False)
    timestep = serializers.IntegerField(required=False)
    timeseries_source = serializers.CharField(required=False)

    mask_fname = serializers.URLField(required=True)
    # channel_manning_fname = serializers.URLField(required=False)
    overland_manning_fname= serializers.URLField(required=False)
    hillslope_fname =serializers.URLField(required=False)
    dem_fname = serializers.URLField(required=False)
    channel_network_fname= serializers.URLField(required=False)
    flowdir_fname = serializers.URLField(required=False)
    pore_size_dist_fname=serializers.URLField(required=False)
    bubbling_pressure_fname=serializers.URLField(required=False)
    resid_moisture_content_fname = serializers.URLField(required=False)
    sat_moisture_content_fname =serializers.URLField(required=False)
    conductivity_fname =serializers.URLField(required=False)
    soil_depth_fname =serializers.URLField(required=False)

    rain_fname =serializers.URLField(required=False)
    et_fname= serializers.URLField(required=False)

    output_response_txt = serializers.CharField(required=False)

class runpytopkapi7validator(serializers.Serializer):
    user_name = serializers.CharField(required=False)
    simulation_name = serializers.CharField(required=False)
    simulation_start_date = serializers.CharField(required=False)
    simulation_end_date = serializers.CharField(required=False)
    USGS_gage = serializers.CharField(required=False)

    threshold = serializers.CharField(required=False)
    cell_size = serializers.CharField(required=False)
    timestep = serializers.IntegerField(required=False)
    timeseries_source = serializers.CharField(required=False)

    mask_fname = serializers.URLField(required=True)
    # channel_manning_fname = serializers.URLField(required=False)
    overland_manning_fname= serializers.URLField(required=False)
    hillslope_fname =serializers.URLField(required=False)
    dem_fname = serializers.URLField(required=False)
    channel_network_fname= serializers.URLField(required=False)
    flowdir_fname = serializers.URLField(required=False)
    pore_size_dist_fname=serializers.URLField(required=False)
    bubbling_pressure_fname=serializers.URLField(required=False)
    resid_moisture_content_fname = serializers.URLField(required=False)
    sat_moisture_content_fname =serializers.URLField(required=False)
    conductivity_fname =serializers.URLField(required=False)
    soil_depth_fname =serializers.URLField(required=False)

    rain_fname =serializers.URLField(required=False)
    et_fname= serializers.URLField(required=False)

    #intials
    init_soil_percentsat= serializers.FloatField(required=False)
    init_overland_vol = serializers.FloatField(required=False)
    init_channel_flow =  serializers.FloatField(required=False)

    output_response_txt = serializers.CharField(required=False)

class downloadsoildataforpytopkapiRequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)
    output_f_file = serializers.CharField(required=True)
    output_dth1_file = serializers.CharField(required=True)
    output_dth2_file = serializers.CharField(required=True)
    output_k_file = serializers.CharField(required=True)
    output_psif_file = serializers.CharField(required=True)
    output_sd_file = serializers.CharField(required=True)
    output_tran_file = serializers.CharField(required=True)

    output_bubbling_pressure_file = serializers.CharField(required=True)
    output_pore_size_distribution_file = serializers.CharField(required=True)
    output_residual_soil_moisture_file = serializers.CharField(required=True)
    output_saturated_soil_moisture_file = serializers.CharField(required=True)
    output_ksat_rawls_file = serializers.CharField(required=True)

class downloadsoildataforpytopkapi2RequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)
    output_f_file = serializers.CharField(required=True)
    output_dth1_file = serializers.CharField(required=True)
    output_dth2_file = serializers.CharField(required=True)
    output_k_file = serializers.CharField(required=True)
    output_psif_file = serializers.CharField(required=True)
    output_sd_file = serializers.CharField(required=True)
    output_tran_file = serializers.CharField(required=True)

    output_bubbling_pressure_file = serializers.CharField(required=True)
    output_pore_size_distribution_file = serializers.CharField(required=True)
    output_residual_soil_moisture_file = serializers.CharField(required=True)
    output_saturated_soil_moisture_file = serializers.CharField(required=True)
    output_ksat_rawls_file = serializers.CharField(required=True)
    output_ksat_hz_file = serializers.CharField(required=True)

class downloadsoildataforpytopkapi3RequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)

    output_dth1_file = serializers.CharField(required=True)
    output_dth2_file = serializers.CharField(required=True)
    output_psif_file = serializers.CharField(required=True)
    output_sd_file = serializers.CharField(required=True)
    output_bubbling_pressure_file = serializers.CharField(required=True)
    output_pore_size_distribution_file = serializers.CharField(required=True)
    output_residual_soil_moisture_file = serializers.CharField(required=True)
    output_saturated_soil_moisture_file = serializers.CharField(required=True)
    output_ksat_rawls_file = serializers.CharField(required=True)
    output_ksat_hz_file = serializers.CharField(required=True)


class downloadsoildataforpytopkapi4RequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)

    output_dth1_file = serializers.CharField(required=False)
    output_dth2_file = serializers.CharField(required=False)
    output_psif_file = serializers.CharField(required=False)
    output_sd_file = serializers.CharField(required=False)
    output_bubbling_pressure_file = serializers.CharField(required=False)
    output_pore_size_distribution_file = serializers.CharField(required=False)
    output_residual_soil_moisture_file = serializers.CharField(required=False)
    output_saturated_soil_moisture_file = serializers.CharField(required=False)


    output_ksat_LUT_file = serializers.CharField(required=False)
    output_ksat_ssurgo_wtd_file = serializers.CharField(required=False)
    output_ksat_ssurgo_min_file = serializers.CharField(required=False)
    output_hydrogrp_file = serializers.CharField(required=False)



class downloadsoildataforpytopkapi5RequestValidator(serializers.Serializer):
    Watershed_Raster = serializers.URLField(required=True)

    output_dth1_file = serializers.CharField(required=False)
    output_dth2_file = serializers.CharField(required=False)
    output_psif_file = serializers.CharField(required=False)
    output_sd_file = serializers.CharField(required=False)
    output_bubbling_pressure_file = serializers.CharField(required=False)
    output_pore_size_distribution_file = serializers.CharField(required=False)
    output_residual_soil_moisture_file = serializers.CharField(required=False)
    output_saturated_soil_moisture_file = serializers.CharField(required=False)


    output_ksat_LUT_file = serializers.CharField(required=False)
    output_ksat_ssurgo_wtd_file = serializers.CharField(required=False)
    output_ksat_ssurgo_min_file = serializers.CharField(required=False)
    output_hydrogrp_file = serializers.CharField(required=False)
    output_df   = serializers.CharField(required=False)



class loadpytopkapivalidator(serializers.Serializer):
    hs_res_id = serializers.CharField(required=True)

    # output_hs_rs_id_txt = serializers.CharField(required=False)
    # output_q_sim_txt= serializers.CharField(required=False)
    output_response_txt = serializers.CharField(required=False)

class modifypytopkapivalidator(serializers.Serializer):
    fac_l = serializers.FloatField(required=False)
    fac_ks = serializers.FloatField(required=False)
    fac_n_o = serializers.FloatField(required=False)
    fac_n_c = serializers.FloatField(required=False)
    fac_th_s = serializers.FloatField(required=False)
    pvs_t0 = serializers.FloatField(required=False)
    vo_t0 = serializers.FloatField(required=False)
    qc_t0 = serializers.FloatField(required=False)
    kc = serializers.FloatField(required=False)

    hs_res_id = serializers.CharField(required=True)

    # output_hs_rs_id_txt = serializers.CharField(required=False)
    # output_q_sim_txt= serializers.CharField(required=False)
    output_response_txt = serializers.CharField(required=False)


class abstractclimatedataRequestValidator(serializers.Serializer):
    startDate = serializers.CharField(required=True)
    endDate = serializers.CharField(required=True)
    cell_size = serializers.CharField(required=False)

    input_raster = serializers.URLField(required=True)
    output_vp_fname = serializers.CharField(required=False)
    output_tmin_fname = serializers.CharField(required=False)
    output_tmax_fname = serializers.CharField(required=False)
    output_srad_fname = serializers.CharField(required=False)
    output_prcp_fname = serializers.CharField(required=False)
    # output_dayl_fname = serializers.CharField(required=False)


class calculatereferenceetfromdaymetRequestValidator(serializers.Serializer):
    windspeed = serializers.FloatField(required=False)

    dem_nc = serializers.URLField(required=True)
    srad_nc = serializers.URLField(required=True)
    tmax_nc = serializers.URLField(required=True)
    tmin_nc = serializers.URLField(required=True)
    vp_nc = serializers.URLField(required=False)

    out_et_nc = serializers.CharField(required=False)


class calculaterainETfromdaymetRequestValidator(serializers.Serializer):
    startDate = serializers.CharField(required=False)
    endDate = serializers.CharField(required=False)
    cell_size = serializers.FloatField(required=False)

    output_et_reference_fname = serializers.CharField(required=False)
    output_rain_fname = serializers.CharField(required=False)

    input_raster = serializers.URLField(required=True)
    input_dem= serializers.URLField(required=True)


class downloadgeospatialandforcingfilesRequestValidator(serializers.Serializer):
    inputs_dictionary_json = serializers.URLField(required=True)
    download_request = serializers.CharField(required=False)

    output_zipfile = serializers.CharField(required=False)
    output_response_txt= serializers.CharField(required=False)



class changetimestepofforcingnetcdfRequestValidator(serializers.Serializer):
    time_interval_in_hr = serializers.CharField(required=False)

    output_et_reference_fname = serializers.CharField(required=True)
    output_rain_fname = serializers.CharField(required=True)

    input_netcdf = serializers.URLField(required=True)



class bboxfromtiffRequestValidator(serializers.Serializer):
    output_json = serializers.CharField(required=False)
    input_raster = serializers.URLField(required=True)


class bboxfromshpRequestValidator(serializers.Serializer):
    output_json = serializers.CharField(required=False)
    input_shp = serializers.URLField(required=True)

class outletxyfromshpRequestValidator(serializers.Serializer):
    output_json = serializers.CharField(required=False)
    input_shp = serializers.URLField(required=True)

class runuebRequestValidator(serializers.Serializer):
    watershedName = serializers.CharField(required=True)
    topY = serializers.FloatField(required=True)
    bottomY = serializers.FloatField(required=True)
    leftX = serializers.FloatField(required=True)
    rightX = serializers.FloatField(required=True)
    cell_size = serializers.FloatField(required=True)
    startDateTime = serializers.CharField(required=True)
    endDateTime = serializers.CharField(required=True)




# support for individual hydroshare
class runpytopkapi8validator(serializers.Serializer):
    user_name = serializers.CharField(required=False)
    simulation_name = serializers.CharField(required=False)
    simulation_start_date = serializers.CharField(required=False)
    simulation_end_date = serializers.CharField(required=False)
    USGS_gage = serializers.CharField(required=False)

    threshold = serializers.IntegerField(required=False)   #IntegerField
    cell_size = serializers.CharField(required=False)
    timestep = serializers.IntegerField(required=False)
    timeseries_source = serializers.CharField(required=False)

    mask_fname = serializers.URLField(required=True)
    # channel_manning_fname = serializers.URLField(required=False)
    overland_manning_fname= serializers.URLField(required=False)
    hillslope_fname =serializers.URLField(required=False)
    dem_fname = serializers.URLField(required=False)
    channel_network_fname= serializers.URLField(required=False)
    flowdir_fname = serializers.URLField(required=False)
    pore_size_dist_fname=serializers.URLField(required=False)
    bubbling_pressure_fname=serializers.URLField(required=False)
    resid_moisture_content_fname = serializers.URLField(required=False)
    sat_moisture_content_fname =serializers.URLField(required=False)
    conductivity_fname =serializers.URLField(required=False)
    soil_depth_fname =serializers.URLField(required=False)

    rain_fname =serializers.URLField(required=False)
    et_fname= serializers.URLField(required=False)

    #intials
    init_soil_percentsat = serializers.FloatField(required=False)
    init_overland_vol = serializers.FloatField(required=False)
    init_channel_flow = serializers.FloatField(required=False)

    output_response_txt = serializers.CharField(required=False)

    hs_username = serializers.CharField(required=False)
    hs_client_id = serializers.CharField(required=False)
    hs_client_secret = serializers.CharField(required=False)
    token = serializers.CharField(required=False)


class downloadgeospatialandforcingfiles2RequestValidator(serializers.Serializer):
    inputs_dictionary_json = serializers.URLField(required=True)
    download_request = serializers.CharField(required=False)

    output_zipfile = serializers.CharField(required=False)
    output_response_txt= serializers.CharField(required=False)

    hs_username = serializers.CharField(required=False)
    hs_client_id = serializers.CharField(required=False)
    hs_client_secret = serializers.CharField(required=False)
    token = serializers.CharField(required=False)


class loadpytopkapi2validator(serializers.Serializer):
    hs_res_id = serializers.CharField(required=True)

    # output_hs_rs_id_txt = serializers.CharField(required=False)
    # output_q_sim_txt= serializers.CharField(required=False)
    output_response_txt = serializers.CharField(required=False)

    hs_username = serializers.CharField(required=False)
    hs_client_id = serializers.CharField(required=False)
    hs_client_secret = serializers.CharField(required=False)
    token = serializers.CharField(required=False)

class modifypytopkapi2validator(serializers.Serializer):
    fac_l = serializers.FloatField(required=False)
    fac_ks = serializers.FloatField(required=False)
    fac_n_o = serializers.FloatField(required=False)
    fac_n_c = serializers.FloatField(required=False)
    fac_th_s = serializers.FloatField(required=False)
    pvs_t0 = serializers.FloatField(required=False)
    vo_t0 = serializers.FloatField(required=False)
    qc_t0 = serializers.FloatField(required=False)
    kc = serializers.FloatField(required=False)

    hs_res_id = serializers.CharField(required=True)

    # output_hs_rs_id_txt = serializers.CharField(required=False)
    # output_q_sim_txt= serializers.CharField(required=False)
    output_response_txt = serializers.CharField(required=False)

    hs_username = serializers.CharField(required=False)
    hs_client_id = serializers.CharField(required=False)
    hs_client_secret = serializers.CharField(required=False)
    token = serializers.CharField(required=False)



