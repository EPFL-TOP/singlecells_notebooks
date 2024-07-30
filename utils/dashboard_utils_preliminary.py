from __future__ import annotations

import pandas as pd
import numpy as np
from bokeh.plotting import figure, curdoc, show, output_file, output_notebook
from bokeh.models import Button, ColumnDataSource, Slider, Select, Span, CustomJS, CheckboxGroup, ResetTool, RadioGroup, ButtonGroup, TapTool, BoxZoomTool, Span, HoverTool, SaveTool
import bokeh.layouts
import bokeh.palettes
from bokeh.layouts import gridplot
import nd2
from pathlib import Path, WindowsPath
from skimage.io import imread
import scipy

# Turn off warnings
import logging # isort:skip
log = logging.getLogger(__name__)

from bokeh.util.warnings import BokehUserWarning 
import warnings 
warnings.simplefilter(action='ignore', category=BokehUserWarning)

def normalize_df(dataframe):
    """ 0-1 normalization of the entire dataframe

    Parameters:
        dataframe: pandas.DataFrame
            Input dataframe
    
    Returns:
        normalized_dataframe: pandas.DataFrame
            Pandas dataframe containing the 0-1 normalization for each column
    
    """
    normalized_dataframe = dataframe.copy(deep=True)
    for column in normalized_dataframe.columns.values:
        normalized_dataframe[column] = (normalized_dataframe[column] - normalized_dataframe[column].min()) / (normalized_dataframe[column].max() - normalized_dataframe[column].min())  
    return normalized_dataframe


def format_dataframe(path_to_traces, normalize_df):
    """ Format the input excel data for peak registration in the dashboard

    Parameters:
        path_to_traces: str
            A string pointing the path to the traces
        normalize_df: func
            A function to normalzie the dataset
    
    Returns:
        all_signals_norm: pandas.DataFrame
            Pandas dataframe containing the 0-1 normalization for each column
        tod: pandas.DataFrame
            Pandas dataframe containig the ToD of each cell, 0-index corrected
        min_max_values: dict
            A dictionary with key-values pairs being the keys the name of the cells and
            the values a list with the first element being the minimum intensity for that
            particular trace and the second element the maximum intensity for that trace 
    
    """    
    traces = pd.read_excel(Path(path_to_traces).as_posix(), skiprows=[1]).iloc[:,1:]
    traces.columns = traces.columns.astype(str)
    looping_p = [position for position in traces.columns if ('.' not in position) and ('Unnamed:' not in position)]
    heidi_p = [position for position in traces.columns if ('.' in position) and ('Unnamed:' not in position)]
    tod = pd.read_excel(Path(path_to_traces).as_posix(), nrows=1).iloc[:,1:]
    tod.columns = tod.columns.astype(str)
    tod = tod[looping_p]
    tod.columns = tod.columns.astype(str)
    tod = tod-1
    all_signals_norm = normalize_df(traces[looping_p])
    min_max_values = {str(cell):[traces[cell].min(), traces[cell].max()] for cell in looping_p}
    if heidi_p is not None:
        return all_signals_norm, tod, min_max_values, traces[looping_p], traces[heidi_p]
    return all_signals_norm, tod, min_max_values, traces[looping_p], None

def get_abs(signal, peak_heights, min_max_values):
    """ Revert 0-1 normalization

    Parameters:
        signal: str
            The name of the cell signal, as used in the traces dataframe    
        peak_heights: list
            List containing the normalized peak intensities
        min_max_values: dictionary
            A dictionary with key-values pairs being the keys the name of the cells and
            the values a list with the first element being the minimum intensity for that
            particular trace and the second element the maximum intensity for that trace
        
    Returns:
        abs_heights: list
            A list of the absolute intensities of each peak
        
    """

    min_value = min_max_values[signal][0]
    max_value = min_max_values[signal][1]
    abs_heights = []
    for height in peak_heights:
        abs_height = height * (max_value - min_value) + min_value
        abs_heights.append(abs_height)
    
    return abs_heights

def dashboard_registration(all_signals_norm, tod, min_max_values, exp_name, interval, notebook_url):

    # App to record peaks
    # Implement manual addition of peaks - removal still to be added
    # Implement sliders for beginning of oscillations, end of oscillatiosn
    # Implement a button to change the validity of the signal
    # Turn off warnings

#     from __future__ import annotations

#     import logging # isort:skip
#     log = logging.getLogger(__name__)

#     from bokeh.util.warnings import BokehUserWarning 
#     import warnings 
#     warnings.simplefilter(action='ignore', category=BokehUserWarning)

    # Validity of a given signal
    global validity_pablo
    validity_pablo = True

    # SInitial values
    signals_df = all_signals_norm.copy(deep=True)
    # signals_df = df_total.copy(deep=True)
    t = np.linspace(0,len(signals_df)-1,len(signals_df))
    t = [int(i) for i in t]
    selected_signal = signals_df.columns.values[0]
    signal = signals_df[selected_signal]
    tod_list = [float(tod[selected_signal])]*len(signals_df)

    peaks=[]
    heights=[]
    troughs=[]
    trough_heights = []

    # signal=signal/(np.max(signal))

    # Initial parameters for find_peaks
    initial_prominence = 0.5
    initial_height = 0.00
    threshold=np.zeros(len(t))

    # Initial parameters for timepoint sliders
    initial_tf = 0
    v_line = np.linspace(0,len(signals_df)-1, len(signals_df))
    # tf = np.ones(len(v_line))
    bo = np.zeros(len(v_line))
    eo = np.zeros(len(v_line))

    # Data dataframe to store data
    global peak_data_dashboard
    peak_data_dashboard = pd.DataFrame()


    # Find Peaks functions
    # def find_peaks_with_params(signal, prominence, height):
    #     peaks, properties = scipy.signal.find_peaks(signal, prominence=prominence, height=height)
    #     return peaks, properties['peak_heights']


    def find_peaks_with_params(signal, prominence, height):
        peaks, properties = scipy.signal.find_peaks(signal, prominence=prominence, height=height)
        properties = signal.iloc[peaks]
        return peaks, properties

    def find_troughs_with_params(signal, prominence):
        troughs, properties = scipy.signal.find_peaks(-signal, prominence=prominence)
        properties = signal.iloc[troughs]
        return troughs, properties

    # Create ColumnDataSource
    source1 = ColumnDataSource(data=dict(t=t, signal=signal))                    # Signal and time domain
    source2 = ColumnDataSource(data=dict(peaks=[], heights=[]))                  # Peaks and peak heights
    source3 = ColumnDataSource(data=dict(t=t, threshold=threshold))              # Time domain and height threshold
    source4 = ColumnDataSource(data=dict(troughs=troughs, trough_heights=trough_heights))              # Trophs and heights
    source5 = ColumnDataSource(data=dict(tod_list=tod_list, 
                                         v_line=v_line))
    source7 = ColumnDataSource(data=dict(bo=bo, v_line=v_line))                  # Beginning of oscillations
    source8 = ColumnDataSource(data=dict(eo=eo, v_line=v_line))                  # End of oscillations


    # Create Bokeh figure to display signal and peaks
    plot = figure(title='normalized  '+exp_name+'_pos_'+signals_df.columns.values[0], x_range=(np.min(t)-0.3, np.max(t)+0.3), y_range=(np.min(signal)-0.025, np.max(signal)+0.025), width=1000, height=800, tools=[TapTool(), BoxZoomTool(), ResetTool()])
    plot.circle('t', 'signal', source=source1, line_width=2, size=6, line_color='blue', legend_label='Signal', nonselection_alpha=1.0)
    plot.line('t', 'signal', source=source1, line_width=2, line_color='blue', legend_label='Signal', nonselection_alpha=1.0)
    plot.circle('peaks', 'heights', source=source2, size=8, color='red', legend_label='Peaks', line_width=2, line_color='black', nonselection_alpha=1.0)
    plot.circle('troughs', 'trough_heights', source=source4, size=8, color='green', legend_label='Trophs', line_width=2, line_color='black', nonselection_alpha=1.0)
    plot.line('t', 'threshold', source=source3, line_color='green', line_dash='dashed', legend_label='Height threshold')
    plot.line('tod_list', 'v_line', source=source5, line_color='red', line_dash='solid', line_width=10, legend_label='ToD', alpha=0.6)
    plot.line('bo', 'v_line', source=source7,line_color='orange', line_dash='dashed', line_width=3, legend_label='Beginning of oscillations')
    plot.line('eo', 'v_line', source=source8,line_color='blue', line_dash='dashed', line_width=3, legend_label='End of oscillations')


    # Callback function for filtering signal - not implemented for now

    # Callback function for dropdown menu
    def update_signal(attr, old, new):
        selected_signal = signal_select.value
        signal = signals_df[selected_signal]

        # Update data source
        source1.data = dict(t=t, signal=signal)

        # Update peaks and heights based on new signal
        prominence_value = prominence_slider.value
        height_value = height_slider.value
        peaks, peak_heights = find_peaks_with_params(signal, prominence_value, height_value)
        troughs, trough_heights = find_troughs_with_params(signal, prominence_value)
        source2.data = dict(peaks=peaks, heights=peak_heights)
        source4.data = dict(troughs=troughs, trough_heights=trough_heights)    

        # Update ToD line
        #index = bo_slider.value
        index = tod[selected_signal].to_numpy().item()
        tod_list = np.ones(len(v_line)) * index
        source5.data = dict(tod_list=tod_list, v_line=v_line)  

        # Update plot title
        plot.title.text = 'normalized ' + exp_name+'_pos_' + selected_signal



    # Callback function for sliders using on_change
    def update_peaks(attr, old, new):
        # Retrieve values from sliders
        prominence_value = prominence_slider.value
        height_value = height_slider.value
        selected_signal = signal_select.value
        signal = signals_df[selected_signal]
        bo = bo_slider.value
        eo = eo_slider.value
        # Update peaks and heights based on new prominence value
        # First use eo as end limit for find peaks, then use boolean array to remove any peak before bo and finally update threshold
        # Caution here: the behaviour of find peaks is different for beginning and end filtering of peaks
        if eo < len(signal): 
            peaks, peak_heights = find_peaks_with_params(signal[:eo+1], prominence_value, height_value)
            troughs, trough_heights = find_troughs_with_params(signal[:eo+1], prominence_value)
        else: 
            peaks, peak_heights = find_peaks_with_params(signal, prominence_value, height_value)
            troughs, trough_heights = find_troughs_with_params(signal, prominence_value)
        bo_mask = peaks >= bo
        # eo_mask = peaks <= eo
        peaks = peaks[bo_mask]
        peak_heights = peak_heights[bo_mask]
        bo_mask = troughs >= bo
        troughs = troughs[bo_mask]
        trough_heights = trough_heights[bo_mask]
        threshold = np.ones(len(t))*height_value
        # Update data source
        source2.data = dict(peaks=peaks, heights=peak_heights)
        source3.data = dict(t=t, threshold=threshold)
        source4.data = dict(troughs=troughs, trough_heights=trough_heights)

    # Callback function for vertical slider

    # Callback for beginning of oscillations
    def update_bo(attr, old, new):
        index = bo_slider.value
        bo = np.ones(len(v_line)) * index
        source7.data = dict(bo=bo, v_line=v_line)

    # Callback for end of oscillations
    def update_eo(attr, old, new):
        index = eo_slider.value
        eo = np.ones(len(v_line)) * index
        source8.data = dict(eo=eo, v_line=v_line)

    # Save data function
    def save_data():
        selected_signal = signal_select.value
        signal = signals_df[selected_signal]
        prominence_value = prominence_slider.value
        height_value = height_slider.value
        bo = bo_slider.value
        eo = eo_slider.value
        tod_value = int(tod[selected_signal].values)
        # peaks, peak_heights = find_peaks_with_params(signal, prominence_value, height_value)
        peaks, peak_heights = source2.data['peaks'], source2.data['heights']
        troughs, trough_heights = source4.data['troughs'], source4.data['trough_heights']
        abs_heights = get_abs(signal=selected_signal, peak_heights=source2.data['heights'], min_max_values=min_max_values)
        abs_trough_heights = get_abs(signal=selected_signal, peak_heights=source4.data['trough_heights'], min_max_values=min_max_values)

        critical_points = np.concatenate([peaks, troughs])
        critical_heights = np.concatenate([peak_heights, trough_heights])
        abs_critical_heights = np.concatenate([abs_heights, abs_trough_heights])
        is_peak = np.concatenate([np.ones(len(peaks)), np.zeros(len(troughs))]).astype(bool)
        is_trough = np.concatenate([np.zeros(len(peaks)), np.ones(len(troughs))]).astype(bool)

        global peak_data_dashboard
        try:
            if peak_data_dashboard['Signal'].isin([selected_signal]).any(): # Remove the stored-data if the signal is re-done
                peak_data_dashboard = peak_data_dashboard[~peak_data_dashboard['Signal'].isin([selected_signal])]

        except KeyError:
            print('First signal')

        if validity_pablo == True:
            data = {
                'Exp': [exp_name] * len(critical_points),
                'Signal': [exp_name + '_pos_' + selected_signal] * len(critical_points),
                'Critical Point pos': critical_points,
                'Critical Point heights': critical_heights,
                'Absolute Critical Point heights': abs_critical_heights,
                'Threshold': [height_value] * len(critical_points),
                'Critical Point proms': [prominence_value] * len(critical_points),
                'Beginning of oscillation': [bo] * len(critical_points),
                'End of oscillation': [eo] * len(critical_points),
                'ToD': [tod_value] * len(critical_points),
                'Sample rate': [interval] * len(critical_points),
                'Is peak': is_peak,
                'Is trough': is_trough,
                }
        else:
            data = {
                'Exp': [exp_name],
                'Signal': [exp_name + '_pos_' + selected_signal],
                'Critical Point pos': [np.nan],
                'Critical Point heights': [np.nan],
                'Absolute Critical Point heights': [np.nan],
                'Threshold': [np.nan],
                'Critical Point proms': [np.nan],
                'Beginning of oscillation': [np.nan],
                'End of oscillation': [np.nan],
                'ToD': [tod_value],
                'Sample rate': [interval],
                'Is peak': False,
                'Is trough': False,
            }
        df = pd.DataFrame(data)
        df = df.sort_values(by='Critical Point pos')
        peak_data_dashboard=pd.concat([peak_data_dashboard, df], ignore_index=True)


        # Move to the next signal in the dropdown menu
        signal_index = list(signals_df.keys()).index(selected_signal)
        try:
            # next_index = (signal_index + 1) % len(signals_df)
            next_index = signal_index + 1
            next_signal = list(signals_df.keys())[next_index]
            signal_select.value = next_signal
        except IndexError:
            print('{selected_signal} is the last signal of the dataset'.format(selected_signal=selected_signal))




    # def select_tap_callback():
    #     return """
    #     const indices = cb_data.source.selected.indices;

    #     if (indices.length > 0) {
    #         const index = indices[0];
    #         other_source.data = {'index': [index]};
    #         other_source.change.emit();  
    #     }
    #     """
    # def remove_peak(attr, old, new):
    #     try:
    #         # peaks = source2.data['peaks']
    #         # peak_heights = source2.data['heights']
    #         selected_index = int(new['index'][0])
    #         selected_peak = source1.data['t'][selected_index]
    #         selected_height = source1.data['signal'][selected_index]
    #         # add peak if peak is not in the previous peak list
    #         print('**************')
    #         print(selected_index)
    #         print(selected_peak)
    #         print(selected_height)
    #         print(source2.data['peaks'])
    #         print('-------------------')
    #         peak_list=[int(peak) for peak in source2.data['peaks']]
    #         height_list=[int(peak) for peak in source2.data['heights']] 

    #         print(peak_list)
    #         if selected_peak not in peak_list:
    #             new_peaks = {'peaks': peak_list.append(selected_peak), 'heights': height_list.append(selected_height)}
    #             source2.data = new_peaks
    #         else:
    #             temp_index = source2.data['peaks'].tolist().index(selected_index)
    #             temp_height = source2.data['heights'].tolist().remove(source2.data['heights'][temp_index])
    #             temp_peaks = source2.data['peaks'].tolist().remove(selected_index)
    #             print(temp_index)
    #             print(temp_height)
    #             print(temp_peaks)
    #             if temp_height == None and temp_peaks==None: 
    #                 temp_height=[]
    #                 temp_peaks=[]
    #             new_peaks = {'peaks':temp_peaks, 'heights': temp_height}
    #             source2.data = new_peaks

    #     except IndexError:
    #         pass


    # def tap_point(attr, old, new):
    #     try:
    #         print(peaks)
    #         print(source2.data['peaks'])
    #         # peak_heights = source2.data['heights']
    #         selected_index = source1.selected.indices[0]
    #         selected_peak = source1.data['t'][selected_index]
    #         selected_height = source1.data['signal'][selected_index]
    #         # add peak if peak is not in the previous peak list
    #         if selected_peak not in peaks:
    #             new_peaks = {'peaks': np.append(source2.data['peaks'], selected_peak), 'heights': np.append(source2.data['heights'], selected_height)}
    #             source2.data = new_peaks
    #         elif selected_index in peaks:
    #             global position_selected_height
    #             global position_selected_peak
    #             position_selected_height = np.where(source2.data['heights'] == selected_height)[0][0]
    #             position_selected_peak = np.where(source2.data['peaks'] == selected_peak)[0][0]
    #             new_peaks = {'peaks': np.delete(source2.data['peaks'], position_selected_peak), 'heights': np.delete(source2.data['heights'], position_selected_height)}
    #             source2.data = new_peaks

    #     except IndexError:
    #         pass

    # def tap_point(attr, old, new):
    #     try:

    #         peaks = source2.data['peaks']
    #         peak_heights = source2.data['heights']
    #         selected_index = source1.selected.indices[0]
    #         selected_peak = source1.data['t'][selected_index]
    #         selected_height = source1.data['signal'][selected_index]
    #         # add peak if peak is not in the previous peak list
    #         if selected_peak not in peaks:
    #             new_peaks = {'peaks': np.append(source2.data['peaks'], selected_peak), 'heights': np.append(source2.data['heights'], selected_height)}
    #             source2.data = new_peaks
    #         elif selected_index in peaks:
    #             global position_selected_height
    #             global position_selected_peak
    #             position_selected_height = np.where(source2.data['heights'] == selected_height)[0][0]
    #             position_selected_peak = np.where(source2.data['peaks'] == selected_peak)[0][0]
    #             new_peaks = {'peaks': np.delete(source2.data['peaks'], position_selected_peak), 'heights': np.delete(source2.data['heights'], position_selected_height)}
    #             source2.data = new_peaks

    #     except IndexError:
    #         pass

    def tap_point(attr, old, new):
        try:
            peaks = list(source2.data['peaks'])
            peak_heights = list(source2.data['heights'])
            troughs = list(source4.data['troughs'])
            trough_heights = list(source4.data['trough_heights'])
            selected_index = source1.selected.indices[0]
            selected_point = source1.data['t'][selected_index]
            selected_height = source1.data['signal'][selected_index]

            if selected_point in peaks:
                # Remove the peak if selected point is a peak
                index = peaks.index(selected_point)
                peaks.pop(index)
                peak_heights.pop(index)
            elif selected_point in troughs:
                # Remove the trough if selected point is a trough
                index = troughs.index(selected_point)
                troughs.pop(index)
                trough_heights.pop(index)
            else:
                # Add the new peak and its height
                peaks.append(selected_point)
                peak_heights.append(selected_height)

            # Update sources
            source2.data = {'peaks': peaks, 'heights': peak_heights}
            source4.data = {'troughs': troughs, 'trough_heights': trough_heights}

        except IndexError:
            pass  # Handle case where no point is selected

    def change_validity(new):
        global validity_pablo
        validity_pablo = not validity_pablo

    # Create sliders with on_change callback
    prominence_slider = Slider(title='Prominence', value=initial_prominence, start=0.0, end=1.0, step=0.01)
    prominence_slider.on_change('value', update_peaks)

    height_slider = Slider(title='Height', value=initial_height, start=0.00, end=1.0, step=0.01)
    height_slider.on_change('value', update_peaks)

    bo_slider = Slider(title='Beginning of oscillations', value=initial_tf, start=0, end=(len(signal)-1), step=1)
    bo_slider.on_change('value', update_bo)
    bo_slider.on_change('value', update_peaks)

    eo_slider = Slider(title='End of oscillations', value=initial_tf, start=0, end=(len(signal)-1), step=1)
    eo_slider.on_change('value', update_eo)
    eo_slider.on_change('value', update_peaks)


    # Create dropdown menu
    signal_select = Select(title='Select Signal:', value=selected_signal, options=list(signals_df.keys()))
    signal_select.on_change('value', update_signal)

    # Create a button to save data
    save_button = Button(label="Save Data", button_type="success")
    save_button.on_click(save_data)

    # Create a radiobutton to change validity of data
    validity_button = RadioGroup(labels=['Valid', 'Not Valid'], active=0)
    validity_button.on_click(change_validity)


    # Add or remove peaks by tapping - Removing still to be done
    source1.selected.on_change('indices', tap_point)

    # tap_tool = bokeh.models.TapTool(callback=bokeh.models.CustomJS(args=dict(other_source=source9),code=select_tap_callback()))



    # plot.add_tools(tap_tool)
    # source9.on_change('data', remove_peak)



    slider_layout = bokeh.layouts.column(
        bokeh.layouts.Spacer(height=30),
        prominence_slider,
        bokeh.layouts.Spacer(height=15),
        height_slider,
        bo_slider,
        eo_slider,
    )

    #Dropdown and save button
    dropdown_layout = bokeh.layouts.column(
        bokeh.layouts.Spacer(height=30),
        signal_select,
        save_button,
        validity_button
    )

    # Set up layout
    norm_layout = bokeh.layouts.row(
        plot,
        bokeh.layouts.Spacer(width=15),
        slider_layout,
        dropdown_layout,

    )
    # Add layout to the current document
    def norm_app(doc):
        doc.add_root(norm_layout)
    
    bokeh.io.show(norm_app, notebook_url)

    
    
def visualize_traces(traces, exp_name=None):
    palette=bokeh.palettes.d3['Category10'][4]
    plots_per_row = 5      
    plots = []
    traces_l = traces[0]
    traces_h = traces[1]
    if traces_h is None:
        for col in traces_l.columns:
            source = ColumnDataSource(data=dict(index=traces_l.index, values=traces_l[col]))
            p = figure(title=f'{exp_name}'+'_'+col, plot_width=275, plot_height=275)
            p.line('index', 'values', source=source, line_width=2)
            plots.append(p)

        grid = gridplot(plots, ncols=plots_per_row, merge_tools=False)
    else:
        traces_h.columns = pd.Index([column.split('.')[0]  for column in traces_h.columns])
        for col in traces_l.columns:
            source_l = ColumnDataSource(data=dict(index=traces_l.index, values=traces_l[col]))
            source_h = ColumnDataSource(data=dict(index=traces_h.index, values=traces_h[col]))
            p = figure(title=f'{exp_name}'+'_'+col, plot_width=275, plot_height=275)
            p.line('index', 'values', source=source_l, line_width=2, color=palette[0])
            p.line('index', 'values', source=source_h, line_width=2, color=palette[-1])

            plots.append(p)
        grid = gridplot(plots, ncols=plots_per_row, merge_tools=False)

    # Show the results in the notebook
    show(grid)

def visualize_traces_wpoints(traces, registered_peaks, exp_name):
    palette=bokeh.palettes.d3['Category10'][4]
    plots_per_row = 5
    plots = []
    valid_signals = list(registered_peaks.dropna()['Signal'].unique())
    traces_df = traces.copy(deep=True)
    traces_df.columns = pd.Index([exp_name+'_pos_'+str(column) for column in traces_df.columns])
    valid_traces = traces_df[valid_signals]
    
    for col in valid_traces.columns:
        # Traces
        source = ColumnDataSource(data=dict(index=valid_traces.index, values=valid_traces[col]))

        # Critical points
        peak_df = registered_peaks.loc[(registered_peaks['Signal'] == col) & (registered_peaks['Is peak'] == True),:]
        trough_df = registered_peaks.loc[(registered_peaks['Signal'] == col) & (registered_peaks['Is trough'] == True),:]

        source2 = ColumnDataSource(data=dict(peak_pos=peak_df['Critical Point pos'], peak_heights=peak_df['Absolute Critical Point heights'])) # Peaks
        source3 = ColumnDataSource(data=dict(trough_pos=trough_df['Critical Point pos'], trough_heights=trough_df['Absolute Critical Point heights'])) #Troughs

        p = figure(title=col, plot_width=275, plot_height=275)

        # Render
        p.line('index', 'values', source=source, line_width=2, color=palette[0])
        p.circle('peak_pos', 'peak_heights', source=source2, color=palette[-1], size=5)
        p.circle('trough_pos', 'trough_heights', source=source3, color=palette[-2], size=5)

        # Append plots
        plots.append(p)

    # Arrange the plots in a grid
    grid = gridplot(plots, ncols=plots_per_row, merge_tools=False)

    # Show the results in the notebook
    show(grid)

# if __name__ == '__main__':
#     notebook_url = 'localhost:8890'    # Web socket allocated for the Jupyter server (see the web browser url bar)
#     path_to_traces_looping = './wscepfl0090example_intensity_w_multiple.xlsx'    # Path to the traces and ToD data
#     path_to_traces_heidi = ''
#     exp_name = 'wscepfl0090'    # Experiment name
#     output_folder = './peak_analysis'    # Path to output folder
#     interval = 10    # Sample rate
    
#     output_folder_path = Path(output_folder)
#     peaks_output_path = Path('./' + output_folder_path.as_posix() + '/peaks_2' + exp_name + '.csv')
    
#     all_signals_norm, tod, min_max_values = format_dataframe(path_to_traces_looping, normalize_df=normalize_df)
    
    
#     dashboard_registration(all_signals_norm=all_signals_norm, tod=tod, min_max_values=min_max_values, exp_name=name, interval=interval, notebook_url=notebook_url)