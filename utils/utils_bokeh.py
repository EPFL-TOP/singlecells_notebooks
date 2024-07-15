import json, os

from bokeh.io import output_notebook, show, push_notebook
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, CustomJS, Select, Span, Whisker
from bokeh.plotting import figure, curdoc
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.server.server import Server
from bokeh.transform import jitter
import numpy as np




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
    time_of_death_pos = [t for t in time_of_death if t>0.]
    bins = np.linspace(np.min(time_of_death_pos), np.max(time_of_death), 20)    
    hist, edges = np.histogram(time_of_death_pos, density=False, bins=bins)
    time_of_death_fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.5, fill_color="black", line_color="white", legend_label="time of death")


    osc_cycle_fig = figure(width=300, height=300, title='oscillation cycle')
    cycle=[]
    time=[]
    source_osc_period      = ColumnDataSource(data=dict(cycle=[], time=[]))
    source_osc_period_err  = ColumnDataSource(data=dict(base=[], upper=[], lower=[]))
    source_osc_period_line = ColumnDataSource(data=dict(x=[], y=[]))


    for p in peaks:
        max_time=[]
#peaks[col]['peaks']['max_frame']]
        try:
            max_time = peaks[p]['peaks']["max_time"]
        except KeyError:
            continue
        if len(max_time)<2: break
        for i in range(len(max_time)-1):
            cycle.append(i+1)
            time.append(max_time[i+1]-max_time[i])
    source_osc_period.data=dict(cycle=cycle, time=time)
    

    classes = list(set(cycle))
    tmp_dict={}
    for cl in classes:
        tmp_dict[cl]=[]
    for i in range(len(cycle)):
        tmp_dict[cycle[i]].append(time[i])
    upper=[]
    lower=[]
    mean=[]
    for c in range(1, len(classes)+1):
        array = np.array(tmp_dict[c])
        upper.append(np.mean(array)+np.std(array)/2)
        lower.append(np.mean(array)-np.std(array)/2)
        mean.append(np.mean(array))
    source_osc_period_err.data=dict(base=classes, upper=upper, lower=lower)
    source_osc_period_line.data=dict(x=classes, y=mean)

    whisker = Whisker(base='base',upper='upper', lower='lower', source=source_osc_period_err, level="annotation", line_width=2)
    #whisker.upper_head.size=20
    #whisker.lower_head.size=20
    osc_cycle_fig.add_layout(whisker)
    osc_cycle_fig.scatter(x=jitter('cycle', width=0.25, range=osc_cycle_fig.x_range), y='time', source=source_osc_period, size=8)
    osc_cycle_fig.line('x', 'y', source=source_osc_period_line, line_color='black')




    def select_intensity_type(attr, old, new):
        new_layout = create_plots_layout()
        doc.clear()  # Clear the current document
        doc.add_root(column(row(dropdown_intensity_type,dropdown_normalisation_type), 
                            row(last_peak_time_fig,osc_time_fig, time_of_death_fig, osc_cycle_fig), 
                            new_layout, print_button, rerender_button))
    dropdown_intensity_type.on_change('value', select_intensity_type)

    def select_normalisation_type(attr, old, new):
        new_layout = create_plots_layout()
        doc.clear()  # Clear the current document
        doc.add_root(column(row(dropdown_intensity_type,dropdown_normalisation_type), 
                            row(last_peak_time_fig,osc_time_fig, time_of_death_fig, osc_cycle_fig), 
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
                tod_data_x  = [t for t in time if t>=peaks[col]['time_of_death'] and peaks[col]['time_of_death']>0.]
                tod_data_frame = [t for t in range(len(time)) if time[t]>=peaks[col]['time_of_death'] and peaks[col]['time_of_death']>0.]


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
                            row(last_peak_time_fig,osc_time_fig, time_of_death_fig, osc_cycle_fig), 
                            new_layout, print_button, rerender_button))
        # push_notebook()
    rerender_button.on_click(rerender_plots)

    # Add the layout and buttons to the current document
    doc.add_root(column(row(dropdown_intensity_type, dropdown_normalisation_type),
                        row(last_peak_time_fig,osc_time_fig, time_of_death_fig, osc_cycle_fig), 
                        layout, print_button, rerender_button))
    #doc.add_root(column(print_button, rerender_button))

