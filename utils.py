import json, os

#Function to merge inputs
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
    return selected_data