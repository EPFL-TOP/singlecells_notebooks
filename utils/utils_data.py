import json, os
import copy


import numpy as np




#__________________________________________________________________
def convert_pos_intensities_to_df(analysis_data, mode='mean'):
    intensities = {}
    times = {}
    channels = []
    for exp in analysis_data:
        for well in analysis_data[exp]:
            for pos in analysis_data[exp][well]:
                for cell in analysis_data[exp][well][pos]['cells']:
                    if len(analysis_data[exp][well][pos][cell]['ROI']["intensity_mean"])>0:
                        for ch in analysis_data[exp][well][pos][cell]['ROI']["intensity_mean"][0]:
                            intensities[ch]={}
                            times[ch]={}
                            channels.append(ch)
                        break
                    break
                break
            break
        break

    for exp in analysis_data:
        for well in analysis_data[exp]:
            for pos in analysis_data[exp][well]:
                for cell in analysis_data[exp][well][pos]['cells']:
                    name = '{}_{}'.format(pos.replace('.nd2',''), cell.replace('cell',''))
                    for ch in channels:
                        intensities[ch][name]=[]
                        times[ch][name]=[t/(60000.) for t in analysis_data[exp][well][pos][cell]['time']]
                        if mode=='mean':
                            data_int  = analysis_data[exp][well][pos][cell]['ROI']["intensity_mean"]
                            for d in data_int:
                                intensities[ch][name].append(d[ch])

    return intensities, times
#d = {'col1': [1, 2], 'col2': [3, 4]}

#df = pd.DataFrame(data=d)

#__________________________________________________________________
def get_intensities_time(analysis_data):
    to_ret={}
    for exp in analysis_data:
        for well in analysis_data[exp]:
            for pos in analysis_data[exp][well]:
                for cell in analysis_data[exp][well][pos]['cells']:
                    name = '{}_{}'.format(pos.replace('.nd2',''), cell.replace('cell',''))
                    to_ret[name]={'time':[], 'channels':[]}
                    if len(analysis_data[exp][well][pos][cell]['ROI']["intensity_mean"])>0:
                        for ch in analysis_data[exp][well][pos][cell]['ROI']["intensity_mean"][0]:
                            to_ret[name]['channels'].append(ch)
                            to_ret[name][ch]={'mean':[], 'max':[], 'std':[]}
                    
                    for tf in range(len(analysis_data[exp][well][pos][cell]['time'])):
                        to_ret[name]['time'].append(analysis_data[exp][well][pos][cell]['time'][tf]/60000.)
                        for ch in to_ret[name]['channels']:
                            to_ret[name][ch]['mean'].append(analysis_data[exp][well][pos][cell]['ROI']["intensity_mean"][tf][ch])
                            to_ret[name][ch]['max'].append(analysis_data[exp][well][pos][cell]['ROI']["intensity_max"][tf][ch])
                            to_ret[name][ch]['std'].append(analysis_data[exp][well][pos][cell]['ROI']["intensity_std"][tf][ch])

    return to_ret


#__________________________________________________________________
def get_peaks_tod(analysis_data):
    to_ret={}
    for exp in analysis_data:
        for well in analysis_data[exp]:
            for pos in analysis_data[exp][well]:
                for cell in analysis_data[exp][well][pos]['cells']:
                    name = '{}_{}'.format(pos.replace('.nd2',''), cell.replace('cell',''))
                    to_ret[name]={'start_oscillation_time':analysis_data[exp][well][pos][cell]['start_oscillation_time'], 
                                  'end_oscillation_time':analysis_data[exp][well][pos][cell]['end_oscillation_time'],
                                  'time_of_death':analysis_data[exp][well][pos][cell]['time_of_death'],
                                  'peaks':{'min_time':analysis_data[exp][well][pos][cell]['peaks']['min_time'],
                                           'max_time':analysis_data[exp][well][pos][cell]['peaks']['max_time'],
                                           'min_frame':analysis_data[exp][well][pos][cell]['peaks']['min_frame'],
                                           'max_frame':analysis_data[exp][well][pos][cell]['peaks']['max_frame']},
                                           'min_int':analysis_data[exp][well][pos][cell]['peaks']['min_int'],
                                           'max_int':analysis_data[exp][well][pos][cell]['peaks']['max_int']}


    return to_ret


#__________________________________________________________________
def get_last_peaks_time(peaks_tod):
    last_peak_list=[]
    for pos in peaks_tod:

        if len(peaks_tod[pos]['peaks']['max_time'])>0 and len(peaks_tod[pos]['peaks']['min_time'])>0:
            if peaks_tod[pos]['peaks']['max_time'][-1]>peaks_tod[pos]['peaks']['min_time'][-1]:
                last_peak_list.append(peaks_tod[pos]['peaks']['max_time'][-1])
            else:
                last_peak_list.append(peaks_tod[pos]['peaks']['min_time'][-1])
        if len(peaks_tod[pos]['peaks']['min_time'])==0:       
            last_peak_list.append(peaks_tod[pos]['peaks']['max_time'][-1])

    return last_peak_list

#__________________________________________________________________
def get_peaks_stuff_time(name, peaks_tod):
    to_ret=[]
    for pos in peaks_tod:
        to_ret.append(peaks_tod[pos][name])
    return to_ret

