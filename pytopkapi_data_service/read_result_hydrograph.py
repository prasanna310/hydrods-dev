'''
The purpose of this script is to convert FDR tiff's into connectivity rasters
'''

import argparse, os, sys
import h5py, numpy
from datetime import datetime, timedelta


def read_hydrograph_from_results(results=None,outlet_id=None, rain_h5=None, simulation_start_date=None,timestep=24,
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


    # # default results location
    # if results == None:
    #     results = os.path.join(output_folder, "results.h5")
    #
    # # output path
    # if output_qsim == None:
    #     output_qsim = os.path.join(output_folder, "Q_sim.txt")

    f =  h5py.File(results, 'r')

    # channel flow
    ndar_Qc_out = f['Channel/Qc_out'][:]
    ar_Qsim = ndar_Qc_out[:,int(outlet_id)]

    # ET actual out
    ET_out_ar = f['ET_out'][:]
    eta_ar = numpy.average(ET_out_ar, axis=1)

    # overland water vol
    Vo = f['Overland']['V_o'][:]
    vo_ar = numpy.average(Vo, axis=1)

    # soil water vol
    V_s = f['Soil']['V_s'][:]
    vs_ar = numpy.average(V_s, axis=1)

    print ('WARNING: Total nan values in simulated discharge (converted to 0)',numpy.count_nonzero(~numpy.isnan(ar_Qsim)) )
    ar_Qsim[numpy.isnan(ar_Qsim)] = 0  # line added

    f.close()


    # create an array with first line as
    # 2011  01  05  0   0   15

    s = datetime.strptime(simulation_start_date, "%m/%d/%Y")
    timestep = timedelta(hours=int(timestep))
    final_array = []
    for i in range(len(ar_Qsim)-1):  # for some reason, simulated values are one more than the observed.. :TODO, fix this
        one_timestep = [s.year, s.month, s.day, s.hour, s.minute, ar_Qsim[i]/ 0.028316846592 ] # with the multiplier, output is now in cfs
        final_array.append(one_timestep)
        s = s + timestep


    # replace nans with 0
    # final_array[numpy.isnan(final_array)] = 0
    # final_array = numpy.nan_to_num(final_array)
    # print (final_array)

    # numpy.savetxt(output_qsim, X=final_array, fmt='%2d %2d %2d %2d %2d %7.3f',delimiter=',')  # was 5.1f
    numpy.savetxt(output_qsim, X=final_array, fmt='%2d,%2d,%2d,%2d,%2d,%7.3f', delimiter=',')  # was 5.1f

    print ('Progress --> Hydrograph results Read')
    return


def read_values_from_results(results=None, outlet_id=None, rain_h5=None, simulation_start_date=None, timestep=24,
                                 output_qsim=None, input_q_obs=None,cell_size = None ):
    '''
    The output file contains array of the format:
            YYYY  MM  DD  hh  mm  q_simulated ..
            Both q_simulated and q_observed are in cfs
    :param results:
    :param outlet_id:
    :param simulation_start_date:
    :param timestep:
    :param output_qsim:
    :param input_q_obs:
    :return:
    creates output file output_qsim  e.g. "q_sim_cfs.txt" which contains data in format YYYY MM DD hh mm qsim eta vo vs vc ppt,
    '''

    # default results location
    if results == None:
        results = "results.h5"
    # output path
    if output_qsim == None:
        output_qsim =  "q_sim_cfs.txt"
    if rain_h5 is None:
        rain_h5 = os.path.join(os.path.split(results)[0], 'rainfields.h5')

    # read results h5 file
    f = h5py.File(results, 'r')

    # cell size
    if cell_size is None:
        from osgeo import gdal
        dset = gdal.Open(os.path.join(os.path.split(results)[0], 'mask.tif'))
        x0, dx, fy, y0, fx, dy = dset.GetGeoTransform()
        cell_size = dx

    # channel flow
    ndar_Qc_out = f['Channel/Qc_out'][:]
    ar_Qsim = ndar_Qc_out[:, int(outlet_id)]

    # total number of col
    total_cell_no = int(ndar_Qc_out.shape[1])

    # ET actual out
    ET_out_ar = f['ET_out'][:]
    eta_ar = numpy.average(ET_out_ar, axis=1)           # in mm/day

    # overland water vol
    Vo = f['Overland']['V_o'][:]
    vo_ar = numpy.average(Vo, axis=1) /(cell_size**2)*1000    # in mm/day

    # soil water vol
    V_s = f['Soil']['V_s'][:]
    vs_ar = numpy.average(V_s, axis=1)/(cell_size**2)*1000    # in mm/day

    # channel water vol
    V_c = f['Channel']['V_c'][:]
    vc_ar = numpy.average(V_c, axis=1)/(cell_size**2)*1000   # in mm/day

    print ('WARNING: Total nan simulated discharge (converted to 0)', numpy.count_nonzero(numpy.isnan(ar_Qsim)))
    ar_Qsim[numpy.isnan(ar_Qsim)] = 0  # line added

    f.close()

    # read rainfall h5 file
    rain_f = h5py.File(rain_h5, 'r')
    ppt = rain_f['sample_event']['rainfall'][:]
    ppt_1d = numpy.average(ppt, axis=1)           # in mm/day
    rain_f.close()

    # create an array with first line as
    # 2011  01  05  0   0   15

    s = datetime.strptime(simulation_start_date, "%m/%d/%Y")
    timestep = timedelta(hours=int(timestep))
    final_array = []
    for i in range(len(ar_Qsim) - 1):  # for some reason, simulated values are one more than the observed.. :TODO, fix this
        # YYYY  MM  DD  hh  mm  q_simulated  eta vo vs vc ppt
        one_timestep = [s.year, s.month, s.day, s.hour, s.minute, ar_Qsim[i] / 0.028316846592,eta_ar[i], vo_ar[i], vs_ar[i], vc_ar[i], ppt_1d[i]  ]  # with the multiplier, output is now in cfs
        final_array.append(one_timestep)
        s = s + timestep

    # replace nans with 0
    # final_array[numpy.isnan(final_array)] = 0
    # final_array = numpy.nan_to_num(final_array)
    # print (final_array)

    # numpy.savetxt(output_qsim, X=final_array, fmt='%2d %2d %2d %2d %2d %7.3f',delimiter=',')  # was 5.1f
    numpy.savetxt(output_qsim, X=final_array, fmt='%2d,%2d,%2d,%2d,%2d,%7.3f,%7.3f,%7.3f,%7.3f,%7.3f,%7.3f', delimiter=',')  # was 5.1f

    print ('Progress --> Hydrograph results Read')
    return




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_results', required=True, help="input results.h5 file")
    parser.add_argument('-oid', '--outlet_id', required=True, help="Outlet ID of the cell whose discharge is desired")
    parser.add_argument('-d', '--start_date', required=True, help="Simulation start date")
    parser.add_argument('-t', '--timestep', required=False, help="Timestep for the simulation carried out")
    parser.add_argument('-iq', '--q_obs', required=False, help="Path to input q observed file (txt)")
    parser.add_argument('-oq', '--q_sim', required=False, help="Path to output q simulated file (txt) will be saved")

    args = parser.parse_args()

    read_values_from_results(results=args.input_results,
                                 outlet_id=args.outlet_id,
                                  simulation_start_date=args.start_date,
                                  timestep=args.timestep,
                                 output_qsim=args.q_sim,
                                input_q_obs = args.q_obs
                                  )





