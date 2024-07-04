import json, os
import copy

from bokeh.io import output_notebook, show, push_notebook
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, CustomJS, Select, Span
from bokeh.plotting import figure, curdoc
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.server.server import Server
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
def select_cell_main_features(analysis_data, flag, mode='gt'):
    to_ret = copy.deepcopy(analysis_data)
    for exp in analysis_data:
        for well in analysis_data[exp]:
            for pos in analysis_data[exp][well]:
                for cell in analysis_data[exp][well][pos]['cells']:
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

#__________________________________________________________________
intensities={}
peaks={}
last_peaks_time=[]
start_osc_time=[]
end_osc_time=[]
time_of_death=[]

#__________________________________________________________________
global valid_signals
colors=['navy', 'red', 'green']
def modify_doc(doc):
    
    selected_plots_source       = ColumnDataSource(data=dict(selected_plots=[]))
    dropdown_intensity_type     = Select(value='mean', title='intensity', options=['mean','max','std'])
    dropdown_normalisation_type = Select(value='no norm', title='intensity', options=['no norm','normalised','same min'])

    last_peak_time_fig = figure(width=300, height=300, title='last peak time')
    bins = np.linspace(np.min(last_peaks_time), np.max(last_peaks_time), 20)    
    hist, edges = np.histogram(last_peaks_time, density=False, bins=bins)
    last_peak_time_fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.5, fill_color="skyblue", line_color="white", legend_label="last peak")

    osc_time_fig = figure(width=300, height=300, title='start osc time')
    bins = np.linspace(np.min(start_osc_time), np.max(end_osc_time), 20)    
    hist, edges = np.histogram(start_osc_time, density=False, bins=bins)
    osc_time_fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.5, fill_color="skyblue", line_color="white", legend_label="start osc")
    hist, edges = np.histogram(end_osc_time, density=False, bins=bins)
    osc_time_fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.5, fill_color="green", line_color="white", legend_label="end osc")

    time_of_death_fig = figure(width=300, height=300, title='time of death')
    bins = np.linspace(np.min(time_of_death), np.max(time_of_death), 20)    
    hist, edges = np.histogram(time_of_death, density=False, bins=bins)
    time_of_death_fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.5, fill_color="black", line_color="white", legend_label="time of death")


    def select_intensity_type(attr, old, new):
        new_layout = create_plots_layout()
        doc.clear()  # Clear the current document
        doc.add_root(column(row(dropdown_intensity_type,dropdown_normalisation_type), 
                            row(last_peak_time_fig,osc_time_fig, time_of_death_fig), 
                            new_layout, print_button, rerender_button))
    dropdown_intensity_type.on_change('value', select_intensity_type)

    def select_normalisation_type(attr, old, new):
        new_layout = create_plots_layout()
        doc.clear()  # Clear the current document
        doc.add_root(column(row(dropdown_intensity_type,dropdown_normalisation_type), 
                            row(last_peak_time_fig,osc_time_fig, time_of_death_fig), 
                            new_layout, print_button, rerender_button))
    dropdown_normalisation_type.on_change('value', select_normalisation_type)
    


    # Function to get selected plots in Python
    def get_selected_plots():
        return selected_plots_source.data['selected_plots']

    # Function to create plots and buttons layout
    def create_plots_layout():
        plots = []
        buttons = []

        for col in intensities:
            if col not in get_selected_plots():
                #source = ColumnDataSource(data={col: df_cleaned[col], 'x': df_cleaned_time[col]})
                time=intensities[col]['time']
                p = figure(width=300, height=200, title=col)

                color_idx=0
                oscillation_data_x  = [t for t in time if t>=peaks[col]['start_oscillation_time'] and t<=peaks[col]['end_oscillation_time']]
                oscillation_data_frame = [t for t in range(len(time)) if time[t]>=peaks[col]['start_oscillation_time'] and time[t]<=peaks[col]['end_oscillation_time']]
                tod_data_x  = [t for t in time if t>=peaks[col]['time_of_death']]
                tod_data_frame = [t for t in range(len(time)) if time[t]>=peaks[col]['time_of_death']]


                for ch in intensities[col]['channels']:
                    if "BF" in ch: continue
                    intensity=intensities[col][ch][dropdown_intensity_type.value]
                    norm_fact=1
                    if dropdown_normalisation_type.value == 'normalised':
                        norm_fact = np.sum(intensity)
                    intensity=[i/norm_fact for i in intensity]
                    p.line(time, intensity, line_width=2, color=colors[color_idx], alpha=0.8, legend_label=ch)

                    if color_idx==0:
                        oscillation_data_y1 = [intensity[t] for t in oscillation_data_frame]
                        oscillation_data_y2 = [0 for i in range(len(oscillation_data_frame))]
                        p.varea(x=oscillation_data_x, y1=oscillation_data_y1, y2=oscillation_data_y2, fill_alpha=0.10, fill_color='blue')
                        tod_data_y1 = [intensity[t] for t in tod_data_frame]
                        tod_data_y2 = [0 for i in range(len(tod_data_frame))]
                        p.varea(x=tod_data_x, y1=tod_data_y1, y2=tod_data_y2, fill_alpha=0.10, fill_color='black')
                        p.scatter(peaks[col]['peaks']['max_time'], [intensity[t] for t in peaks[col]['peaks']['max_frame']], fill_alpha=0.5, fill_color="red", size=7, line_color='red')
                        p.scatter(peaks[col]['peaks']['min_time'], [intensity[t] for t in peaks[col]['peaks']['min_frame']], fill_alpha=0.5, fill_color="green", size=7, line_color='red')

                    color_idx+=1
                button = Button(label=col, width=60, button_type="success")

                p.legend.location = "bottom_right"

                #p.legend.title = "Channels"
                p.legend.border_line_width = 0
                #p.legend.border_line_color = "navy"
                p.legend.border_line_alpha = 0.0
                #p.legend.title_text_font_size = '10pt'
                p.legend.background_fill_alpha = 0.0
                p.legend.label_text_font_size = "5pt"
                # Increasing the glyph height
                p.legend.glyph_height = 2                
                # increasing the glyph width
                p.legend.glyph_width = 2
                # Increasing the glyph's label height
                p.legend.label_height = 1
                # Increasing the glyph's label height
                p.legend.label_width = 1

                def create_button_callback(plot, column_name, btn):
                    def callback():
                        selected_plots = selected_plots_source.data['selected_plots']
                        if column_name in selected_plots:
                            plot.background_fill_color = 'white'
                            selected_plots.remove(column_name)
                            btn.button_type = 'success'
                        else:
                            plot.background_fill_color = 'rgba(255, 0, 0, 0.1)'
                            selected_plots.append(column_name)
                            btn.button_type = 'danger'
                        selected_plots_source.data = {'selected_plots': selected_plots}  # Update the data source
                        global valid_signals
                        #valid_signals = df_cleaned.drop(columns=selected_plots_source.data['selected_plots']).columns
                        # push_notebook()  # Ensure updates are reflected in the notebook
                    return callback

                button.on_click(create_button_callback(p, col, button))

                plots.append(p)
                buttons.append(button)

        # Organize layout
        plot_rows = []
        for i in range(0, len(plots), 5):
            plot_row = plots[i:i+5]
            button_row = buttons[i:i+5]
            plot_rows.append(row(*plot_row, column(*button_row)))

        layout = column(*plot_rows)
        return layout

    # Create the initial layout
    layout = create_plots_layout()

    # Button to print excluded plots
    print_button = Button(label="Print list of excluded plots", width=200, button_type="primary")
    def print_selected_plots():
        print(get_selected_plots())
        # push_notebook()  # Ensure notebook updates
    print_button.on_click(print_selected_plots)

    # Button to rerender without selected plots
    rerender_button = Button(label="Exclude selected plots", width=200, button_type="warning")
    def rerender_plots():
        new_layout = create_plots_layout()
        doc.clear()  # Clear the current document
        doc.add_root(column(row(dropdown_intensity_type, dropdown_normalisation_type),
                            row(last_peak_time_fig,osc_time_fig, time_of_death_fig), 
                            new_layout, print_button, rerender_button))
        # push_notebook()
    rerender_button.on_click(rerender_plots)

    # Add the layout and buttons to the current document
    doc.add_root(column(row(dropdown_intensity_type, dropdown_normalisation_type),
                        row(last_peak_time_fig,osc_time_fig, time_of_death_fig), 
                        layout, print_button, rerender_button))
    #doc.add_root(column(print_button, rerender_button))

