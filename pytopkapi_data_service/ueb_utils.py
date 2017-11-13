import json, os, shutil, tempfile, zipfile, subprocess
from datetime import datetime
import requests
import zipfile
import time
# from servicefunctions_pytopkapi import pull_from_hydroshare
# import xmltodict
# from .tian_hydrods import HydroDS
from .tian_hydrods import HydroDS as HDS

hydrods_name_pr= 'pdahal'
hydrods_password_pr = 'pDahal2016'

hs_name = 'topkapi_app' #'prasanna_310'
hs_password = 'topkapi12!@' #'Hydrology12!@'
epsg_code = ''
# EPSG_List = []

validation = {}
validation['is_valid'] = True

def validate_model_input_form(request):

    validation = {
        'is_valid': True,
        'result': {}
    }

    # check the bounding box value
    north_lat = request.POST['north_lat']
    south_lat = request.POST['south_lat']
    west_lon = request.POST['west_lon']
    east_lon = request.POST['east_lon']

    try:
        north_lat = round(float(north_lat), 4)
        south_lat = round(float(south_lat), 4)
        west_lon = round(float(west_lon), 4)
        east_lon = round(float(east_lon), 4)
        box_type_valid = True

    except Exception:
        validation['is_valid'] = False
        validation['result']['box_title'] = 'Please enter number values for bounding box.'
        box_type_valid = False

    if box_type_valid:
        error_info = []

        if north_lat <= 24.9493 or north_lat >= 49.5904:
            error_info.append('The North Latitude should be in the US continent.')

        if south_lat <= 24.9493 or south_lat >= 49.5904:
            error_info.append('The South Latitude should be in the US continent.')

        if south_lat >= north_lat:
            error_info.append('The South Latitude should be smaller than the North Latitude.')

        if west_lon <= -125.0011 or west_lon >= -66.9326:
            error_info.append('The West Longitude should be in the US continent.')

        if east_lon <= -125.0011 or east_lon >= -66.9326:
            error_info.append('The East Longitude should be in the US continent.')

        if west_lon >= east_lon:
            error_info.append('The West Longitude should be smaller than the East Longitude.')

        if error_info:
            validation['is_valid'] = False
            validation['result']['box_title'] = ' '.join(error_info)


    # check the outlet point value
    outlet_x = request.POST['outlet_x']
    outlet_y = request.POST['outlet_y']

    try:
        outlet_x = float(outlet_x)
        outlet_y = float(outlet_y)
        point_type_valid = True

    except Exception:
        validation['is_valid'] = False
        validation['result']['point_title'] = 'Please enter number values for outlet point.'
        point_type_valid = False

    if point_type_valid:
        error_info = []
        if not (outlet_x >= west_lon and outlet_x <= east_lon):
            error_info.append('The outlet point longitude should be in the bounding box.')

        if not (outlet_y >= south_lat and outlet_y <= north_lat):
            error_info.append('The outlet point latitude should be in the bounding box.')

        if error_info:
            validation['is_valid'] = False
            validation['result']['point_title'] = ' '.join(error_info)


   # check stream threshold
    stream_threshold = request.POST['stream_threshold']
    try:
        stream_threshold = int(stream_threshold)
        thresh_type_valid = True
    except:
        validation['is_valid'] = False
        validation['result']['threshold_title'] = 'The stream threshold should be an integer or as default value 1000.'


    # # check epsg
    # epsg_code = request.POST['epsg_code']
    # if epsg_code not in [item[1] for item in EPSG_List]:
    #     validation['is_valid'] = False
    #     validation['result']['epsg_title'] = 'Please provide the valide epsg code from the dropdown list.'
    # else:
    #     epsg_code = int(epsg_code)

    # check the date
    start_time_str = request.POST['start_time']
    end_time_str = request.POST['end_time']

    try:
        start_time_obj = datetime.strptime(start_time_str, '%Y/%M/%d')
        end_time_obj = datetime.strptime(end_time_str, '%Y/%M/%d')
        time_type_valid = True
    except:
        validation['is_valid'] = False
        validation['result']['time_title'] = 'Please provide time information.'
        time_type_valid = False

    if time_type_valid:
        # TODO check the supported time period for simulation based on the data source (2009-2015, 2005-2015)
        start_limit_obj = datetime.strptime('2010/01/01', '%Y/%M/%d')
        end_limit_obj = datetime.strptime('2011/12/31', '%Y/%M/%d')

        if start_time_obj > end_time_obj:
            validation['is_valid'] = False
            validation['result']['time_title'] = 'The end time should be equal as or later than the start time.'
        if not(start_time_obj >= start_limit_obj and end_time_obj <= end_limit_obj):
            validation['is_valid'] = False
            validation['result']['time_tile'] = 'The start and end time should be a date between {} and {}.'.\
                format(start_limit_obj.strftime('%Y/%M/%d'), end_limit_obj.strftime('%Y/%M/%d'))


    # check x, y
    x_size = request.POST['x_size']
    y_size = request.POST['y_size']

    try:
        x_size = int(x_size)
        y_size = int(y_size)
        if x_size < 1 or y_size < 1:
            validation['is_valid'] = False
            validation['result']['model_cell_title'] = 'The cell size for reprojection should not be smaller than 1 meter.'
    except:
        validation['is_valid'] = False
        validation['result']['proj_cell_title'] = 'The cell size for reprojection should be number input.'


    # check dx,dy
    dx_size = request.POST['dx_size']
    dy_size = request.POST['dy_size']

    try:
        dx_size = int(dx_size)
        dy_size = int(dy_size)
        if dx_size < 1 or dy_size < 1:
            validation['is_valid'] = False
            validation['result']['model_cell_title'] = 'The cell size for model simulation should not be smaller than 1 meter.'
    except:
        validation['is_valid'] = False
        validation['result']['model_cell_title'] = 'The cell size for model simulation should be number input.'


    # check HS res name and keywords
    res_title = request.POST['res_title']
    res_keywords = request.POST['res_keywords']

    if len(res_title) < 5:
        validation['is_valid'] = False
        validation['result']['res_title'] = 'The resource title should include at least 5 characters.'


    # create job parameter if input is valid
    if validation['is_valid']:
        # TODO: pass the hydroshare token, client-id, client-secret not the user name and password
       validation['result'] = {
            'hs_name': hs_name,
            'hs_password': hs_password,
            'hydrods_name_pr': hydrods_name_pr,
            'hydrods_password_pr': hydrods_password_pr,
            'north_lat': north_lat,
            'south_lat': south_lat,
            'west_lon': west_lon,
            'east_lon': east_lon,
            'outlet_x': outlet_x,
            'outlet_y': outlet_y,
            'watershed_name': 'UEB_model_',
            'stream_threshold': stream_threshold,
            'epsg_code': epsg_code,
            'start_time': start_time_str,
            'end_time': end_time_str,
            'x_size': x_size,
            'y_size': y_size,
            'dx_size': dx_size,
            'dy_size': dy_size,
            'res_title': res_title,
            'res_keywords': res_keywords
        }

    return validation

def hydrods_model_input_service_single_call(hs_client_id, hs_client_secret, token, hydrods_name_pr, hydrods_password_pr,
                                            hs_name, hs_password,
                                 topY, bottomY, leftX, rightX,
                                lat_outlet, lon_outlet, streamThreshold, watershedName,
                                epsgCode, startDateTime, endDateTime, dx, dy, dxRes, dyRes,
                                usic, wsic, tic, wcic, ts_last,
                                res_title, res_keywords,
                                 **kwargs):

    service_response = {
        'status': 'Error',
        'result': 'Failed to make the HydroDS request.'
    }

    try:
        url = 'http://129.123.41.218:20199/api/dataservice/createuebinput'  # TODO: change to production server link
        auth = (hydrods_name_pr, hydrods_password_pr)  # TODO: change to production account info
        payload = {
            'hs_username': hs_name,
            'hs_password': hs_password,
            'hs_client_id': hs_client_id,
            'hs_client_secret': hs_client_secret,
            'token': token,
            'hydrods_name_pr': hydrods_name_pr,
            'hydrods_password_pr': hydrods_password_pr,
            'topY': topY,
            'bottomY': bottomY,
            'leftX': leftX,
            'rightX': rightX,
            # 'lat_outlet': lat_outlet,
            # 'lon_outlet': lon_outlet,
            'streamThreshold': streamThreshold,
            'watershedName': watershedName,
            'epsgCode': epsgCode,
            'startDateTime': startDateTime,
            'endDateTime': endDateTime,
            'dx': dx,
            'dy': dy,
            'dxRes': dxRes,
            'dyRes': dyRes,
            'usic': usic,
            'wsic': wsic,
            'tic': tic,
            'wcic': wcic,
            'ts_last': ts_last,
            'res_title': res_title,
            'res_keywords': json.dumps(res_keywords.split(',')),
        }

        if lat_outlet and lon_outlet:
            payload['lat_outlet'] = lat_outlet
            payload['lon_outlet'] = lon_outlet

        print ('2: About to request service call: createuebinput')
        response = requests.get(url, params=payload, auth=auth)
        response_dict = json.loads(response.text)

        if response.status_code == 200:
            if response_dict['error']:
                service_response['result'] = format(response_dict['error'])
            elif response_dict['data']['info']:
                service_response['status'] = 'Success'
                service_response['result'] = response_dict['data']['info']
        else:
            service_response['result'] = 'Failed to run HydroDS web service for model inputs preparation.'
    except Exception:
        pass

    return service_response


## utils for running the model service
def submit_model_run_job_single_call(res_id, hs_username, hs_password):
    model_run_job = {
        'status': 'Error',
        'result': 'Failed to make the HydroDS web service call.'
    }

    try:

        # url = 'http://hydro-ds.uwrl.usu.edu/api/dataservice/runuebmodel'
        url = 'http://129.123.41.218:20199/api/dataservice/runuebmodel'
        auth = (hydrods_name_pr, hydrods_password_pr)
        payload = {'resource_id': res_id,
                   # 'hs_client_id': OAuthHS['client_id'],
                   # 'hs_client_secret': OAuthHS['client_secret'],
                   # 'token': json.dumps(OAuthHS['token']),
                   # 'hs_username': OAuthHS['user_name'],
                   'hs_username':hs_username,
                   'hs_password':hs_password
                   }
        response = requests.get(url, params=payload, auth=auth)
        response_dict = json.loads(response.text)

        if response.status_code == 200:
            if response_dict['error']:
                model_run_job['result'] = response_dict['error']
            elif response_dict['data']['info']:
                model_run_job['status'] = 'Success'
                model_run_job['result'] = response_dict['data']['info']
        else:
            model_run_job['result'] = 'Failed to run HydroDS web service for model execution.'

    except Exception as e:
        model_run_job = {
            'status': 'Error',
            'result': 'Failed to make the HydroDS web service call.'
        }

    # print ('response_dict', response_dict)

    return model_run_job



def get_job_status_list(id):
    error = None
    try:
        url = 'http://129.123.41.218:20199/api/dataservice/job/check_job_status'
        auth = (hydrods_name_pr, hydrods_password_pr)
        payload = {
            'id':id#'extra_data': 'HydroShare: ' + hs_username  #:TODO (Prasanna) Only give job id
        }

        response = requests.get(url, params=payload,auth=auth)



        if response.status_code == 200:
            result = json.loads(response.text)
            for job in result['data'][:]:
                start_time = datetime.strptime(job['start_time'][:10], '%Y-%m-%d')
                timedelta = datetime.now() - start_time
                if timedelta.days >= 30:
                    result['data'].remove(job)
            job_list = result['data']
            input_check_status = 'success'
        else:
            job_list = []
            input_check_status = 'error'

    except Exception as e:
        error = e
        job_list = []
        input_check_status = 'error'

    # print ('result: ', result)


    return job_list,input_check_status


def pull_from_hydroshare(hs_resource_id=None, output_folder=None, hs_usr_name='prasanna_310',
                         hs_password='Hydrology12!@'):
    """
    :param hs_resource_id:      hydroshare resource id for public dataset(??), that contains a single shapefile
    :return: hs_to_shp:         {'outshp_path': path of shapefile (point or polygon) based on hs_resource_id, 'error':}
    """
    from hs_restclient import HydroShare, HydroShareAuthBasic
    hs_resource_id = str(hs_resource_id)
    # :TODO this password and user name should be User's
    auth = HydroShareAuthBasic(username=hs_usr_name, password=hs_password)
    hs = HydroShare(auth=auth)

    contains_pytopkapi_file = False
    pytopkapi_files = {}
    data_folder = os.path.join(output_folder, hs_resource_id, hs_resource_id, 'data', 'contents')

    resource = hs.getResource(pid=hs_resource_id, destination=output_folder, unzip=True)

    files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    shp = [os.path.join(data_folder, f) for f in files if f.endswith('.shp')]
    tiff = [os.path.join(data_folder, f) for f in files if f.endswith('.tif')]
    zip_files =  [os.path.join(data_folder, f) for f in files if f.endswith('.zip')]

    ini = [f for f in files if f.endswith('.ini')]
    dat = [f for f in files if f.endswith('.dat')]
    h5 = [f for f in files if f.endswith('.h5')]

    if ('TOPKAPI.ini' in ini) and ('cell_param.dat' and 'global_param.dat' in dat) and (
        'rainfields.h5' and 'ET.h5' in h5):
        contains_pytopkapi_file = True
        pytopkapi_files = files

    return_dict = {'hs_res_id': resource, 'files': files, 'shp': shp, 'tiffs': tiff,
                   'contains_pytopkapi_file': contains_pytopkapi_file, 'pytopkapi_files': pytopkapi_files, 'zip_files':zip_files}

    return return_dict


def run_ueb( watershedName, topY, bottomY, leftX, rightX,
                streamThreshold,   startDateTime, endDateTime, cell_size, lat_outlet=None, lon_outlet=None,
                hs_name=hs_name, hs_password=hs_password,hs_client_id=None, hs_client_secret=None, token=None, epsgCode=5072, dx=30, dy=30,
                usic=0, wsic=0, tic=0, wcic=0, ts_last=0, hydrods_name=None, hydrods_password=None,
                 res_keywords='ueb, pytopkapi, melt+snowmelt' , output_rain_and_melt='SWIT.nc'):

    check_interval = 60 # in seconds
    job_id =  None
    new_job_id =  None # 217
    input_check_status = False
    run_check_status = False
    dxRes = dyRes = cell_size
    streamThreshold =int(streamThreshold * 1000000 / (cell_size * cell_size))  # because input streamthresohold is in km2
    res_title = 'UEB result for PyTOPKAPI app for %s watershed'%watershedName
    # hs_res_id  = '120e102e1ce74737afb07fdfdb846d8e' # None
    
    if hydrods_name == None:
        hydrods_name = hydrods_name_pr
        hydrods_password = hydrods_password_pr
    try:
        startDate = datetime.strptime(startDateTime, '%Y/%m/%d')
        endDate = datetime.strptime(endDateTime, '%Y/%m/%d')

        startDateTime = '%s/%s/%s'%( startDate.year, startDate.month, startDate.day )
        endDateTime =  '%s/%s/%s'%( endDate.year, endDate.month, endDate.day )
    except:
        pass

    # step1: submit inputs, get job_id
    print ('1: About to check service call')
    ueb_inputs_response = hydrods_model_input_service_single_call( hydrods_name_pr=hydrods_name_pr, hydrods_password_pr=hydrods_password_pr,
                                               hs_name=hs_name, hs_password=hs_password, hs_client_id=None, hs_client_secret=None, token=None,
                                                                   topY=topY, bottomY=bottomY, leftX=leftX,
                                                                   rightX=rightX,
                                                                   lat_outlet=lat_outlet, lon_outlet=lon_outlet,
                                                                   streamThreshold=streamThreshold,
                                                                   watershedName=watershedName,
                                                                   epsgCode=epsgCode, startDateTime=startDateTime,
                                                                   endDateTime=endDateTime,
                                                                   dx=dx, dy=dy, dxRes=dxRes, dyRes=dyRes,
                                                                   usic=usic, wsic=wsic, tic=tic, wcic=wcic,
                                                                   ts_last=ts_last,
                                                                   res_title=res_title, res_keywords=res_keywords)

    # print ('Response of ueb_inputs_response : ',ueb_inputs_response)
    reps = 0
    while job_id == None:
        try:
            reps = reps + 1
            if 'result' in ueb_inputs_response:
                print ('Jone id is ',job_id)
                job_id =   ueb_inputs_response['result'].split(" ")[-1]
                print ('Success I: Model-input request submitted. JOB ID: ', job_id)
                job_id = int(job_id)

        except Exception as e:
            job_id = None
            print ('error',e)
        if reps > 200:
            return {'success': 'False', 'Message':'UEB input preparation took Waited too long, hence process aborted'}
        time.sleep(check_interval)



    # step2: Check if the  model-inputs is successful, wait until input_check_status = 'success'
    reps = 0
    while input_check_status == False: # != 'success':# False:
        try:
            reps = reps + 1
            job_list, input_check_status = get_job_status_list(id=job_id)             # keep checking the job id until success, or some error msg returned
            print ('Checking if model-input preparation with job id: %s  is completed is completed... Returned Job check status is: %s' % (job_id, input_check_status))

            if input_check_status == 'success':
                job_description = job_list[0]['job_description']                    # e.g. 'create ueb model input'
                id = job_list[0]['id']
                hs_res_id = job_list[0]['message'].split('/')[-1]
                print ('Success II: Model-inputs prepared succesfully. i.e. %s Successful with Job id: %s, and the model-input files are saved at %s' % (job_description, id, hs_res_id))
            else:
                time.sleep(check_interval)
        except Exception as e:
            print ('Error', e)
            time.sleep(check_interval)

        if reps > 200:
            return {'success': 'False', 'Message':'UEB input preparation took Waited too long, hence process aborted'}
    # :TODO, if  Failed, stop the process, throw error msg






    # step3: now that the model input is created, submit the job, and get new job_id
    model_run_response = submit_model_run_job_single_call(res_id=hs_res_id, hs_username=hs_name, hs_password=hs_password)
    if model_run_response['status'] == 'Success'  or model_run_response['status'] == 'success' :
        new_job_id = model_run_response['result'].split(" ")[-1]
        print ('Inputs submitted for execution. New Job ID is ', new_job_id)



    # step4: Check if the model ran succesfully. If yes, execute a function to donwload & extract zip to return SWIT.nc
    reps = 0
    while run_check_status == False: # != 'success' or run_check_status != True or run_check_status != 'Success': # False:

        try:
            reps = reps + 1
            # keep checking the job id until success, or some error msg returned
            job_list, run_check_status = get_job_status_list(id=new_job_id)
            print ('Checking if model-run with job id: %s is completed... Returned job check status is : %s' % (new_job_id, job_list[0]['is_success']))


            if  job_list[0]['is_success'] ==True:  #job_list[0]['is_success'] == 'success' or job_list[0]['is_success'] == 'Success' or
                job_description = job_list[0]['job_description']

                print ('Success III: Yay,  Successful in %s. We now have results'%(job_description))
                run_check_status =  job_list[0]['is_success']
            else:
                time.sleep(check_interval)
                run_check_status = False

        except Exception as e:
            print ('Error', e)
            time.sleep(check_interval)
            run_check_status = False
        if reps > 200:
            return {'success': 'False', 'Message':'UEB input preparation took Waited too long, hence process aborted'}

    output_folder = '.'
    files = pull_from_hydroshare(hs_resource_id=hs_res_id, output_folder=output_folder, hs_usr_name=hs_name, hs_password=hs_password)
    zip_files = files['zip_files']
    for file in zip_files:
        if file.endswith('output_package.zip'):
            print ('This is the file: ', file )


            zip_ref = zipfile.ZipFile(file, 'r')
            zip_ref.extractall(output_folder)
            zip_ref.close()

            print ('SWIT.nc location : ', output_folder+'/SWIT.nc')
            ##output_rain_and_melt
            try:
                shutil.copy(output_folder+'/SWIT.nc', output_rain_and_melt)
            except:
                pass

            return {'success':'True'}


