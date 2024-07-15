import json, os
import copy
import numpy as np

#__________________________________________________________________
def merge_inputs(inputFiles, inputDir):
    analysis_data = {}

    for f in inputFiles:
        fname = os.path.join(inputDir, f)
        file = open(fname)
        data = json.load(file)
        for exp in data:
            try:
                wells = analysis_data[exp]
                for well in wells:
                    try:
                        positions = analysis_data[exp][well]
                        for position in positions:
                            try:
                                data_pos = analysis_data[exp][well][position]
                            except KeyError:
                                analysis_data[exp][well][position] = data[exp][well][position]
                    except KeyError:
                        analysis_data[exp][well] = data[exp][well]
            except KeyError:
                analysis_data[exp]=data[exp]
    return analysis_data

#__________________________________________________________________
def select_data(analysis_data, selected_positions):
    selected_data = {}
    for exp in analysis_data:
        for selexp in selected_positions:
            if selexp in exp:
                selected_data[exp]={}
                for well in analysis_data[exp]:
                    for selwell in selected_positions[selexp]:
                        if selwell in well:
                            selected_data[exp][well]={}
                            for pos in analysis_data[exp][well]:
                                for selpos in selected_positions[selexp][selwell]:
                                    if selpos in pos:
                                        selected_data[exp][well][pos]=analysis_data[exp][well][pos]
    return remove_empty_stuff(selected_data)


#__________________________________________________________________
def select_main_flags(analysis_data, flag):
    to_ret = copy.deepcopy(analysis_data)
    for exp in analysis_data:
        for well in analysis_data[exp]:
            for pos in analysis_data[exp][well]:
                try:
                    if analysis_data[exp][well][pos][flag[0]]!=flag[1]:
                        del to_ret[exp][well][pos]
                except KeyError:
                    print('this main flag: -->{}<-- does not exist'.format(flag[0]))

    return remove_empty_stuff(to_ret)



#__________________________________________________________________
def remove_empty_stuff(analysis_data):
    to_ret = copy.deepcopy(analysis_data)
    for exp in analysis_data:
        for well in analysis_data[exp]:
            for pos in analysis_data[exp][well]:
                if len(analysis_data[exp][well][pos]['cells'])==0: 
                    del to_ret[exp][well][pos]

    to_ret2 = copy.deepcopy(to_ret)
    for exp in to_ret:
        for well in to_ret[exp]:
            if len(to_ret[exp][well])==0: del to_ret2[exp][well]

    to_ret3 = copy.deepcopy(to_ret2)
    for exp in to_ret2:
        if len(to_ret2[exp])==0: del to_ret3[exp]

    return to_ret3


#__________________________________________________________________
def select_cell_flags(analysis_data, flag, mode='all'):
    to_ret = copy.deepcopy(analysis_data)
    for exp in analysis_data:
        for well in analysis_data[exp]:
            for pos in analysis_data[exp][well]:
                for cell in analysis_data[exp][well][pos]['cells']:
                    try:
                        found=False
                        for status in analysis_data[exp][well][pos][cell][flag[0]]:
                            if status!=flag[1] and mode=='all':
                                del to_ret[exp][well][pos][cell]
                                to_ret[exp][well][pos]['cells'].remove(cell)
                                break
                            if status==flag[1] and mode=='alo':
                                found=True
                        if found==False and mode=='alo':
                            del to_ret[exp][well][pos][cell]
                            to_ret[exp][well][pos]['cells'].remove(cell)

                    except KeyError:
                        print('this cell flag: -->{}<-- does not exist'.format(flag[0]))

    return remove_empty_stuff(to_ret)


#__________________________________________________________________
def select_cell_main_features(analysis_data, flag, mode='gt', useneg=False):
    to_ret = copy.deepcopy(analysis_data)
    for exp in analysis_data:
        for well in analysis_data[exp]:
            for pos in analysis_data[exp][well]:
                for cell in analysis_data[exp][well][pos]['cells']:
                    if useneg==False and analysis_data[exp][well][pos][cell][flag[0]]<0:
                        continue

                    if mode == 'gt' and analysis_data[exp][well][pos][cell][flag[0]]<flag[1]:
                        del to_ret[exp][well][pos][cell]
                        to_ret[exp][well][pos]['cells'].remove(cell)
                        break
                    if mode == 'lt' and analysis_data[exp][well][pos][cell][flag[0]]>flag[1]:
                        del to_ret[exp][well][pos][cell]
                        to_ret[exp][well][pos]['cells'].remove(cell)
                        break
                    if mode == 'eq' and analysis_data[exp][well][pos][cell][flag[0]]!=flag[1]:
                        del to_ret[exp][well][pos][cell]
                        to_ret[exp][well][pos]['cells'].remove(cell)
                        break

    return remove_empty_stuff(to_ret)


