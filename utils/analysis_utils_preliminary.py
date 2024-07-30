import numpy as np
import pandas as pd

# from scipy.stats import iqr

import re

import pathlib
from pathlib import Path, WindowsPath


from bokeh.io import export_svgs
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.models import Circle, ColumnDataSource, HoverTool, Whisker, FixedTicker
from bokeh.transform import factor_cmap, jitter
from bokeh.palettes import d3

import matplotlib.pyplot as plt

TOOLTIPS = [
    ("Signal", "@Signal"),
]


def first_quartile(series):
    first_q = series.quantile(0.25)
    return first_q
def third_quartile(series):
    third_q = series.quantile(0.75)
    return third_q

def iqr(series):
    first_q = series.quantile(0.25)
    third_q = series.quantile(0.75)
    return abs(third_q-first_q)

# N cycles plotting
def number_cycles_histogram(valid_signals, crit_point='peak'):
    if crit_point=='peak':
        peaks_df = valid_signals.loc[valid_signals['Is peak'] == True,:]
    elif crit_point=='trough':
        peaks_df = valid_signals.loc[valid_signals['Is trough'] == True,:]
    else:
        print('No signal')
    collapsed_df = peaks_df.groupby('Signal').agg(
        Peak_pos_count=('Critical Point pos', 'nunique'),
        Beginning_of_oscillation=('Beginning of oscillation', 'first'),
        End_of_oscillation=('End of oscillation', 'first'),
        Condition=('Condition', 'first'),
        Experiment_name=('Exp', 'first')
    ).reset_index()
    
    # Cycle number from peaks plotting
    grouped = collapsed_df.groupby('Condition')

    # Categorical palette
    n_cond = len(collapsed_df['Condition'].unique())
    if n_cond < 3:
        palette = d3['Category10'][3]  # Index  = # categories
    else:
        palette = d3['Category10'][n_cond] # Up to 10

    # Create figures for each condition
    plots = []
    conditions = collapsed_df['Condition'].unique()
    exp_name = collapsed_df['Experiment_name'].unique()

    for i, (name, group) in enumerate(grouped):

        hist, edges = np.histogram(group['Peak_pos_count'], bins=range(1, max(collapsed_df['Peak_pos_count']) + 2))
        source = ColumnDataSource(data={'top': hist, 'left': edges[:-1], 'right': edges[1:], 'Condition': [name] * len(hist)})

        num_signals = len(group)

        # Title
        if name == 'Control':
            p = figure(title=f"Exp:{exp_name.item()} - Condition: {name} - {crit_point}s", y_axis_label='Hits', tools='pan,wheel_zoom,box_zoom,reset,save')
        else:
            p = figure(title=f"Exp:{exp_name.item()} - Condition: {name} - {crit_point}s", y_axis_label='Hits', tools='pan,wheel_zoom,box_zoom,reset,save')

        p.quad(top='top', bottom=0, left='left', right='right', source=source, fill_color=palette[i], line_color='black', alpha=0.5)

        # Stats

        collapsed_mean = collapsed_df.groupby('Condition')['Peak_pos_count'].agg('mean').reset_index()
        collapsed_median = collapsed_df.groupby('Condition')['Peak_pos_count'].agg('median').reset_index()
        collapsed_min = collapsed_df.groupby('Condition')['Peak_pos_count'].agg('min').reset_index()
        collapsed_max = collapsed_df.groupby('Condition')['Peak_pos_count'].agg('max').reset_index()
        collapsed_std = collapsed_df.groupby('Condition')['Peak_pos_count'].agg(np.std, ddof=1).reset_index()




        mean = collapsed_mean.loc[collapsed_mean['Condition']==name, 'Peak_pos_count'].to_numpy().item()
        median = collapsed_median.loc[collapsed_median['Condition']==name, 'Peak_pos_count'].to_numpy().item()
        min_peak = collapsed_min.loc[collapsed_min['Condition']==name, 'Peak_pos_count'].to_numpy().item()
        max_peak = collapsed_max.loc[collapsed_max['Condition']==name, 'Peak_pos_count'].to_numpy().item()
        std = collapsed_std.loc[collapsed_max['Condition']==name, 'Peak_pos_count'].to_numpy().item()



        p.circle([], [], legend_label=f"Total signals: {num_signals}", fill_color=palette[i])
        p.circle([], [], legend_label=f"Mean: {mean:.2f}", fill_color=palette[i])
        p.circle([], [], legend_label=f"Std: {std:.2f}", fill_color=palette[i])
        p.circle([], [], legend_label=f"Median: {median:.2f}", fill_color=palette[i])
        p.circle([], [], legend_label=f"Min cyc number: {min_peak:.0f}", fill_color=palette[i])
        p.circle([], [], legend_label=f"Max cyc number: {max_peak:.0f}", fill_color=palette[i])



        p.legend.title = 'Statistics'
        p.legend.location = 'top_left'
        p.legend.label_text_font_size = '8pt'
        p.xaxis.axis_label = 'Number of cycles'
        p.xaxis.axis_label_text_font_size='12pt'
        p.xaxis.ticker = FixedTicker(ticks=list(collapsed_df['Peak_pos_count'].unique()))


        # p.output_backend = "svg"
        # export_svgs(p, filename=f"./images/{name}_n_oscillations.svg")
        plots.append(p)


    # Layout arrangement
    layout = column(*plots)

    # Show plot
    show(layout)
    
# Last peak plotting

def last_peak_histogram(valid_signals):
    
    peaks_df = valid_signals.loc[valid_signals['Is peak'] == True,:]
    interval = float(valid_signals['Sample rate'].max())
    
    # Last peak time:
    collapsed_df = peaks_df.groupby('Signal').agg(
        Last_Peak_Time=('Critical Point pos', 'max'),
        Beginning_of_oscillation=('Beginning of oscillation', 'first'),
        End_of_oscillation=('End of oscillation', 'first'),
        Condition=('Condition', 'first'),
        Experiment_name=('Exp', 'first')
    ).reset_index()
   
    grouped = collapsed_df.groupby('Condition')

    # Categorical palette
    n_cond = len(collapsed_df['Condition'].unique())
    if n_cond < 3:
        palette = d3['Category10'][3]  # Index  = # categories
    else:
        palette = d3['Category10'][n_cond] # Up to 10

   # Create figures for each condition
    plots = []
    conditions = collapsed_df['Condition'].unique()
    exp_name = collapsed_df['Experiment_name'].unique()

    for i, (name, group) in enumerate(grouped):

        # hist, edges = np.histogram(group['Last_Peak_Time'], bins=range(int(min(collapsed_df['Last_Peak_Time']) -1), int(max(collapsed_df['Last_Peak_Time']) + 2)))
        hist, edges = np.histogram(group['Last_Peak_Time'] * interval, bins='fd')

        source = ColumnDataSource(data={'top': hist, 'left': edges[:-1], 'right': edges[1:], 'Condition': [name] * len(hist)})

        num_signals = len(group)

        # Assign name
        if name == 'Control':
            p = figure(title=f"Exp:{exp_name.item()} - Condition: {name}", x_axis_label='Last peak time', y_axis_label='Hits', tools='pan,wheel_zoom,box_zoom,reset,save')
        else:
            p = figure(title=f"Exp:{exp_name.item()} - Condition: {name}", x_axis_label='Last peak time', y_axis_label='Hits', tools='pan,wheel_zoom,box_zoom,reset,save')

        p.quad(top='top', bottom=0, left='left', right='right', source=source, fill_color=palette[i], line_color='black', alpha=0.5)

        # Stats

        collapsed_mean = collapsed_df.groupby('Condition')['Last_Peak_Time'].agg('mean').reset_index()
        collapsed_median = collapsed_df.groupby('Condition')['Last_Peak_Time'].agg('median').reset_index()
        collapsed_min = collapsed_df.groupby('Condition')['Last_Peak_Time'].agg('min').reset_index()
        collapsed_max = collapsed_df.groupby('Condition')['Last_Peak_Time'].agg('max').reset_index()
        collapsed_std = collapsed_df.groupby('Condition')['Last_Peak_Time'].agg(np.std, ddof=1).reset_index()




        mean = collapsed_mean.loc[collapsed_mean['Condition']==name, 'Last_Peak_Time'].to_numpy().item() * interval
        median = collapsed_median.loc[collapsed_median['Condition']==name, 'Last_Peak_Time'].to_numpy().item() * interval
        min_peak = collapsed_min.loc[collapsed_min['Condition']==name, 'Last_Peak_Time'].to_numpy().item() * interval
        max_peak = collapsed_max.loc[collapsed_max['Condition']==name, 'Last_Peak_Time'].to_numpy().item() * interval
        std = collapsed_std.loc[collapsed_max['Condition']==name, 'Last_Peak_Time'].to_numpy().item() * interval



        p.circle([], [], legend_label=f"Total signals: {num_signals}", fill_color=palette[i])
        p.circle([], [], legend_label=f"Mean: {mean:.2f}", fill_color=palette[i])
        p.circle([], [], legend_label=f"Std: {std:.2f}", fill_color=palette[i])
        p.circle([], [], legend_label=f"Median: {median:.2f}", fill_color=palette[i])
        p.circle([], [], legend_label=f"Earliest last peak: {min_peak:.0f}", fill_color=palette[i])
        p.circle([], [], legend_label=f"Latest last peak: {max_peak:.0f}", fill_color=palette[i])



        p.legend.title = 'Statistics'
        p.legend.location = 'top_right'
        p.legend.label_text_font_size = '8pt'
        p.xaxis.axis_label = 'Last peak time (min)'
        p.xaxis.axis_label_text_font_size='12pt'
        # p.output_backend = "svg"
        # export_svgs(p, filename=f"./images/{name}_n_oscillations.svg")
        plots.append(p)


    # Layout arrangement
    layout = column(*plots)

    # Show plot
    show(layout)
    

    
def iqr(serie):
    first_q = serie.quantile(0.25)
    third_q = serie.quantile(0.75)
    return abs(third_q-first_q)

def period_vs_cycn(valid_signals, crit_point='peak', alignment='first-peak', min_crit_p=2, first_quartile=first_quartile, third_quartile=third_quartile, iqr=iqr):
    # Compute periods by succ diff of crit_points
    if crit_point=='peak':
        peaks_df = valid_signals.loc[valid_signals['Is peak'] == True,:]
    elif crit_point=='trough':
        peaks_df = valid_signals.loc[valid_signals['Is trough'] == True,:]
    else:
        print('No signal')
        
    # Filter out signals with less than min_crit_p critical points
    mask = peaks_df['Signal'].value_counts() >= min_crit_p
    mask = peaks_df['Signal'].map(mask)
    peaks_df = peaks_df.loc[mask,:]
    
    # Compute periods by succ. diff
    peaks_periods = peaks_df.copy(deep=True)
    interval = peaks_periods['Sample rate'].max()
    peaks_periods['Periods'] = peaks_periods.groupby('Signal')['Critical Point pos'].transform(lambda x: np.diff(x * interval, prepend=np.nan))

    
    # Align periods with cycle number
    if alignment == 'first-peak':
        peaks_periods['Cycle number'] = peaks_periods.groupby('Signal')['Critical Point pos'].rank(method='first').astype(int)
    elif alignment == 'last-peak':
        peaks_periods['Cycle number'] = peaks_periods.groupby('Signal')['Critical Point pos'].rank(method='first', ascending=False).astype(int)
        peaks_periods['Cycle number'] = - (peaks_periods['Cycle number'] - 1 )
    else:
        print('Bad alignment')
        return
    
    grouped = peaks_periods.groupby(['Condition', 'Cycle number'])['Periods'].agg(['mean', iqr, 'median', first_quartile, third_quartile]).reset_index()
    grouped['upper'] = grouped['third_quartile'] + 1.5 * grouped['iqr'] 
    grouped['lower'] = grouped['first_quartile'] - 1.5 * grouped['iqr']
    
    exp_name = peaks_periods['Exp'].unique()

# Create the Bokeh figure
    p = figure(height=400, x_axis_label='Cycle number', 
               x_range=(peaks_periods['Cycle number'].min() - 0.5, peaks_periods['Cycle number'].max() + 0.5), 
               y_range=(min(peaks_periods['Periods'].min(), grouped['lower'].min()) - 10,
                        max(peaks_periods['Periods'].max(), grouped['upper'].max()) + 10), 
               title=f"Exp: {exp_name.item()} - Periods vs Cycle number - {alignment.split('-')[0]}-{crit_point} aligned - 1.5*IQR",
               y_axis_label='Period (min)')

    # Different colors for different conditions
    # Categorical palette
    n_cond = len(peaks_periods['Condition'].unique())
    if n_cond < 3:
        palette = d3['Category10'][3]  # Index  = # categories
    else:
        palette = d3['Category10'][n_cond] # Up to 10

    # Plot data points and error bars for each condition
    for i, (condition, group) in enumerate(grouped.groupby('Condition')):
        source = ColumnDataSource(group)
        p.circle(x='Cycle number', y='mean', size=10, color=palette[i], legend_label=condition, source=source)
        p.dash(x='Cycle number', y='median', size = 50, fill_alpha=1.0, line_width=4, color=palette[i], legend_label=condition, source=source)
        p.line(x='Cycle number', y='mean', color=palette[i], legend_label=condition, source=source)

        error=Whisker(source=source, base='Cycle number', upper='upper', lower='lower', level='annotation', line_width=2, line_color=palette[i])
        error.upper_head.size=20
        error.lower_head.size=20
        p.add_layout(error)
        sub_df = peaks_periods.loc[peaks_periods['Condition']==condition, :]
        sub_df['Cycle number'] = sub_df['Cycle number'].astype(float)
        source2 = ColumnDataSource(sub_df)
        circles = Circle(x=jitter("Cycle number", 0.1), y = "Periods", fill_alpha = 0.3,
            size=8, line_color="white", fill_color = palette[i])
        circles_r = p.add_glyph(source_or_glyph=source2, glyph=circles)
        hover =HoverTool(renderers=[circles_r],
                        tooltips=[('Signal', '@Signal')])
        p.add_tools(hover)
    
    p.xaxis.ticker = FixedTicker(ticks=list(peaks_periods['Cycle number'].unique()))
    p.legend.title = 'Condition'
    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'
    
    
    print(grouped)
    show(p)
    

def intensity_vs_cyclen(valid_signals, crit_point='peak', alignment='first-peak', min_crit_p=2, first_quartile=first_quartile, third_quartile=third_quartile, iqr=iqr):
    # Compute periods by succ diff of crit_points
    if crit_point=='peak':
        peaks_df = valid_signals.loc[valid_signals['Is peak'] == True,:]
    elif crit_point=='trough':
        peaks_df = valid_signals.loc[valid_signals['Is trough'] == True,:]
    else:
        print('No signal')
        
    # Filter out signals with less than min_crit_p critical points
    mask = peaks_df['Signal'].value_counts() >= min_crit_p
    mask = peaks_df['Signal'].map(mask)
    peaks_df = peaks_df.loc[mask,:]
    
    # Compute periods by succ. diff
    peaks_periods = peaks_df.copy(deep=True)
    interval = peaks_periods['Sample rate'].max()
    peaks_periods['Periods'] = peaks_periods.groupby('Signal')['Critical Point pos'].transform(lambda x: np.diff(x * interval, prepend=np.nan))

    
    # Align periods with cycle number
    if alignment == 'first-peak':
        peaks_periods['Cycle number'] = peaks_periods.groupby('Signal')['Critical Point pos'].rank(method='first').astype(int)
    elif alignment == 'last-peak':
        peaks_periods['Cycle number'] = peaks_periods.groupby('Signal')['Critical Point pos'].rank(method='first', ascending=False).astype(int)
        peaks_periods['Cycle number'] = - (peaks_periods['Cycle number'] - 1 )
    else:
        print('Bad alignment')
        return
    
    grouped = peaks_periods.groupby(['Condition', 'Cycle number'])['Absolute Critical Point heights'].agg(['mean', iqr, 'median', first_quartile, third_quartile]).reset_index()
    grouped['upper'] = grouped['third_quartile'] + 1.5 * grouped['iqr'] 
    grouped['lower'] = grouped['first_quartile'] - 1.5 * grouped['iqr']
    
    exp_name = peaks_periods['Exp'].unique()

# Create the Bokeh figure
    p = figure(height=400, x_axis_label='Cycle number', 
               x_range=(peaks_periods['Cycle number'].min() - 0.5, peaks_periods['Cycle number'].max() + 0.5), 
               y_range=(min(peaks_periods['Absolute Critical Point heights'].min(), grouped['lower'].min()) - 50,
                        max(peaks_periods['Absolute Critical Point heights'].max(), grouped['upper'].max()) + 50), 
               title=f"Exp: {exp_name.item()} - Intensity vs Cycle number - {alignment.split('-')[0]}-{crit_point} aligned - 1.5*IQR",
               y_axis_label='Intensity')

    # Different colors for different conditions
    # Categorical palette
    n_cond = len(peaks_periods['Condition'].unique())
    if n_cond < 3:
        palette = d3['Category10'][3]  # Index  = # categories
    else:
        palette = d3['Category10'][n_cond] # Up to 10

    # Plot data points and error bars for each condition
    for i, (condition, group) in enumerate(grouped.groupby('Condition')):
        source = ColumnDataSource(group)
        p.circle(x='Cycle number', y='mean', size=10, color=palette[i], legend_label=condition, source=source)
        p.dash(x='Cycle number', y='median', size = 50, fill_alpha=1.0, line_width=4, color=palette[i], legend_label=condition, source=source)
        p.line(x='Cycle number', y='mean', color=palette[i], legend_label=condition, source=source)

        error=Whisker(source=source, base='Cycle number', upper='upper', lower='lower', level='annotation', line_width=2, line_color=palette[i])
        error.upper_head.size=20
        error.lower_head.size=20
        p.add_layout(error)
        sub_df = peaks_periods.loc[peaks_periods['Condition']==condition, :]
        sub_df['Cycle number'] = sub_df['Cycle number'].astype(float)
        source2 = ColumnDataSource(sub_df)
        circles = Circle(x=jitter("Cycle number", 0.1), y = "Absolute Critical Point heights", fill_alpha = 0.3,
            size=8, line_color="white", fill_color = palette[i])
        circles_r = p.add_glyph(source_or_glyph=source2, glyph=circles)
        hover =HoverTool(renderers=[circles_r],
                        tooltips=[('Signal', '@Signal')])
        p.add_tools(hover)
    
    p.xaxis.ticker = FixedTicker(ticks=list(peaks_periods['Cycle number'].unique()))
    p.legend.title = 'Condition'
    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'
    
    
    print(grouped)
    show(p)
    
def succesive_intensity_r(valid_signals, crit_point='peak', alignment='first-peak', min_crit_p=2, ref='n', first_quartile=first_quartile, third_quartile=third_quartile, iqr=iqr):
    # Compute periods by succ diff of crit_points
    if crit_point=='peak':
        peaks_df = valid_signals.loc[valid_signals['Is peak'] == True,:]
    elif crit_point=='trough':
        peaks_df = valid_signals.loc[valid_signals['Is trough'] == True,:]
    else:
        print('No signal')
        
    # Filter out signals with less than min_crit_p critical points
    mask = peaks_df['Signal'].value_counts() >= min_crit_p
    mask = peaks_df['Signal'].map(mask)
    peaks_df = peaks_df.loc[mask,:]
    
    # Compute periods by succ. diff
    peaks_periods = peaks_df.copy(deep=True)
    interval = peaks_periods['Sample rate'].max()
    peaks_periods['Periods'] = peaks_periods.groupby('Signal')['Critical Point pos'].transform(lambda x: np.diff(x * interval, prepend=np.nan))
    
    # Get magnitude successive ratios based on the reference
    if ref == 'n':
        peaks_periods['Intensity_ratios'] =  peaks_periods.groupby('Signal')['Absolute Critical Point heights'].transform(lambda x: x/x.shift(1))
    elif ref =='n+1':
        peaks_periods['Intensity_ratios'] =  peaks_periods.groupby('Signal')['Absolute Critical Point heights'].transform(lambda x: x.shift(-1)/x)
    else:
        print('Bad position reference')
        return

    
    # Align periods with cycle number
    if alignment == 'first-peak':
        peaks_periods['Cycle number'] = peaks_periods.groupby('Signal')['Critical Point pos'].rank(method='first').astype(int)
    elif alignment == 'last-peak':
        peaks_periods['Cycle number'] = peaks_periods.groupby('Signal')['Critical Point pos'].rank(method='first', ascending=False).astype(int)
        peaks_periods['Cycle number'] = - (peaks_periods['Cycle number'] - 1 )
    else:
        print('Bad alignment')
        return
    
    grouped = peaks_periods.groupby(['Condition', 'Cycle number'])['Intensity_ratios'].agg(['mean', iqr, 'median', first_quartile, third_quartile]).reset_index()
    grouped['upper'] = grouped['third_quartile'] + 1.5 * grouped['iqr'] 
    grouped['lower'] = grouped['first_quartile'] - 1.5 * grouped['iqr']
    
    exp_name = peaks_periods['Exp'].unique()
    
    # y-axis label
    if ref=='n':
        y_axis_label = r'$$\frac{I_{n}}{I_{n-1}}$$'
    elif ref=='n+1':
        y_axis_label = r'$$\frac{I_{n+1}}{I_{n}}$$'
    
    # Create the Bokeh figure
    p = figure(height=400, x_axis_label='Cycle number', 
               x_range=(peaks_periods['Cycle number'].min() - 0.5, peaks_periods['Cycle number'].max() + 0.5), 
               y_range=(min(peaks_periods['Intensity_ratios'].min(), grouped['lower'].min()) - 1,
                        max(peaks_periods['Intensity_ratios'].max(), grouped['upper'].max()) + 1), 
               title=f"Exp: {exp_name.item()} - Successive intensity ratio vs Cycle number - {alignment.split('-')[0]}-{crit_point} aligned - 1.5*IQR",
               y_axis_label=y_axis_label)

    # Different colors for different conditions
    # Categorical palette
    n_cond = len(peaks_periods['Condition'].unique())
    if n_cond < 3:
        palette = d3['Category10'][3]  # Index  = # categories
    else:
        palette = d3['Category10'][n_cond] # Up to 10

    # Plot data points and error bars for each condition
    for i, (condition, group) in enumerate(grouped.groupby('Condition')):
        source = ColumnDataSource(group)
        p.circle(x='Cycle number', y='mean', size=10, color=palette[i], legend_label=condition, source=source)
        p.dash(x='Cycle number', y='median', size = 50, fill_alpha=1.0, line_width=4, color=palette[i], legend_label=condition, source=source)
        p.line(x='Cycle number', y='mean', color=palette[i], legend_label=condition, source=source)

        error=Whisker(source=source, base='Cycle number', upper='upper', lower='lower', level='annotation', line_width=2, line_color=palette[i])
        error.upper_head.size=20
        error.lower_head.size=20
        p.add_layout(error)
        sub_df = peaks_periods.loc[peaks_periods['Condition']==condition, :]
        sub_df['Cycle number'] = sub_df['Cycle number'].astype(float)
        source2 = ColumnDataSource(sub_df)
        circles = Circle(x=jitter("Cycle number", 0.1), y = "Intensity_ratios", fill_alpha = 0.3,
            size=8, line_color="white", fill_color = palette[i])
        circles_r = p.add_glyph(source_or_glyph=source2, glyph=circles)
        hover =HoverTool(renderers=[circles_r],
                        tooltips=[('Signal', '@Signal')])
        p.add_tools(hover)
    
    p.xaxis.ticker = FixedTicker(ticks=list(peaks_periods['Cycle number'].unique()))
    p.legend.title = 'Condition'
    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'

    
    print(grouped)
    show(p)
    
def successive_period_r(valid_signals, crit_point='peak', alignment='first-peak', min_crit_p=3, ref='n', first_quartile=first_quartile, third_quartile=third_quartile, iqr=iqr):
    
    # Peaks or troughs
    if crit_point=='peak':
        peaks_df = valid_signals.loc[valid_signals['Is peak'] == True,:]
    elif crit_point=='trough':
        peaks_df = valid_signals.loc[valid_signals['Is trough'] == True,:]
    else:
        print('No signal')
        
    # Filter out signals with less than min_crit_p critical points
    mask = peaks_df['Signal'].value_counts() >= min_crit_p
    mask = peaks_df['Signal'].map(mask)
    peaks_df = peaks_df.loc[mask,:]
    
    # Compute periods by succ. diff
    peaks_periods = peaks_df.copy(deep=True)
    interval = peaks_periods['Sample rate'].max()
    peaks_periods['Periods'] = peaks_periods.groupby('Signal')['Critical Point pos'].transform(lambda x: np.diff(x * interval, prepend=np.nan))
    
    # Ratios based on reference
    if ref == 'n':
        peaks_periods['Period_ratios'] =  peaks_periods.groupby('Signal')['Periods'].transform(lambda x: x/x.shift(1))
    elif ref =='n+1':
        peaks_periods['Period_ratios'] =  peaks_periods.groupby('Signal')['Periods'].transform(lambda x: x.shift(-1)/x)
    else:
        print('Bad position reference')
        return

    
    # Align periods with cycle number
    if alignment == 'first-peak':
        peaks_periods['Cycle number'] = peaks_periods.groupby('Signal')['Critical Point pos'].rank(method='first').astype(int)
    elif alignment == 'last-peak':
        peaks_periods['Cycle number'] = peaks_periods.groupby('Signal')['Critical Point pos'].rank(method='first', ascending=False).astype(int)
        peaks_periods['Cycle number'] = - (peaks_periods['Cycle number'] - 1 )
    else:
        print('Bad alignment')
        return
    
    
    # Values for whiskers
    grouped = peaks_periods.groupby(['Condition', 'Cycle number'])['Period_ratios'].agg(['mean', iqr, 'median', first_quartile, third_quartile]).reset_index()
    grouped['upper'] = grouped['third_quartile'] + 1.5 * grouped['iqr'] 
    grouped['lower'] = grouped['first_quartile'] - 1.5 * grouped['iqr']
    
    exp_name = peaks_periods['Exp'].unique()
    # y-axis label
    
    if ref=='n':
        y_axis_label = r'$$\frac{P_{n}}{P_{n-1}}$$'
    elif ref=='n+1':
        y_axis_label = r'$$\frac{P_{n+1}}{P_{n}}$$'
    
    # Create the Bokeh figure canvas
    p = figure(height=400, x_axis_label='Cycle number', 
               x_range=(peaks_periods['Cycle number'].min() - 0.5, peaks_periods['Cycle number'].max() + 0.5), 
               y_range=(min(peaks_periods['Period_ratios'].min(), grouped['lower'].min()) - 1,
                        max(peaks_periods['Period_ratios'].max(), grouped['upper'].max()) + 1), 
               title=f"Exp: {exp_name.item()} - Successive period ratio vs Cycle number - {alignment.split('-')[0]}-{crit_point} aligned - 1.5*IQR",
               y_axis_label=y_axis_label)

    # Different colors for different conditions
    # Categorical palette
    n_cond = len(peaks_periods['Condition'].unique())
    if n_cond < 3:
        palette = d3['Category10'][3]  # Index  = # categories
    else:
        palette = d3['Category10'][n_cond] # Up to 10

    # Plot data points and error bars for each condition
    for i, (condition, group) in enumerate(grouped.groupby('Condition')):
        source = ColumnDataSource(group)
        p.circle(x='Cycle number', y='mean', size=10, color=palette[i], legend_label=condition, source=source)
        p.dash(x='Cycle number', y='median', size = 50, fill_alpha=1.0, line_width=4, color=palette[i], legend_label=condition, source=source)
        p.line(x='Cycle number', y='mean', color=palette[i], legend_label=condition, source=source)

        error=Whisker(source=source, base='Cycle number', upper='upper', lower='lower', level='annotation', line_width=2, line_color=palette[i])
        error.upper_head.size=20
        error.lower_head.size=20
        p.add_layout(error)
        sub_df = peaks_periods.loc[peaks_periods['Condition']==condition, :]
        sub_df['Cycle number'] = sub_df['Cycle number'].astype(float)
        source2 = ColumnDataSource(sub_df)
        circles = Circle(x=jitter("Cycle number", 0.1), y = "Period_ratios", fill_alpha = 0.3,
            size=8, line_color="white", fill_color = palette[i])
        circles_r = p.add_glyph(source_or_glyph=source2, glyph=circles)
        hover =HoverTool(renderers=[circles_r],
                        tooltips=[('Signal', '@Signal')])
        p.add_tools(hover)
    
    p.xaxis.ticker = FixedTicker(ticks=list(peaks_periods['Cycle number'].unique()))
    p.legend.title = 'Condition'
    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'
    
    print(grouped)
    show(p)
    

def successive_period_sc(valid_signals, crit_point='peak', alignment='first-peak', min_crit_p=3, ref='n', first_quartile=first_quartile, third_quartile=third_quartile, iqr=iqr):
# Peaks or troughs
    if crit_point=='peak':
        peaks_df = valid_signals.loc[valid_signals['Is peak'] == True,:]
    elif crit_point=='trough':
        peaks_df = valid_signals.loc[valid_signals['Is trough'] == True,:]
    else:
        print('No signal')
        
    # Filter out signals with less than min_crit_p critical points
    mask = peaks_df['Signal'].value_counts() >= min_crit_p
    mask = peaks_df['Signal'].map(mask)
    peaks_df = peaks_df.loc[mask,:]
    
    # Compute periods by succ. diff
    peaks_periods = peaks_df.copy(deep=True)
    interval = peaks_periods['Sample rate'].max()
    peaks_periods['Periods'] = peaks_periods.groupby('Signal')['Critical Point pos'].transform(lambda x: np.diff(x * interval, prepend=np.nan))
    
    # Ratios based on reference
    if ref == 'n':
        peaks_periods['Period_shifted'] =  peaks_periods.groupby('Signal')['Periods'].transform(lambda x: x.shift(1))
    elif ref =='n+1':
        peaks_periods['Period_shifted'] =  peaks_periods.groupby('Signal')['Periods'].transform(lambda x: x.shift(-1))
    else:
        print('Bad position reference')
        return

    
    # Align periods with cycle number
    if alignment == 'first-peak':
        peaks_periods['Cycle number'] = peaks_periods.groupby('Signal')['Critical Point pos'].rank(method='first').astype(int)
    elif alignment == 'last-peak':
        peaks_periods['Cycle number'] = peaks_periods.groupby('Signal')['Critical Point pos'].rank(method='first', ascending=False).astype(int)
        peaks_periods['Cycle number'] = - (peaks_periods['Cycle number'] - 1 )
    else:
        print('Bad alignment')
        return
    
    exp_name = peaks_periods['Exp'].unique()
    # y-axis label
    
    if ref=='n':
        y_axis_label = r'$$P_{n}$$'
        x_axis_label = r'$$P_{n-1}$$'
        diagonal_legend = 'Pn = Pn-1'
        fig_title = r'Pn vs Pn-1'
    elif ref=='n+1':
        y_axis_label = r'$$P_{n+1}$$'
        x_axis_label = r'$$P_{n}$$'
        diagonal_legend = 'Pn+1 = Pn'
        fig_title = 'Pn+1 vs Pn'


    
    # Create the Bokeh figure canvas
    p = figure(height=400, x_axis_label=x_axis_label, 
               x_range=(peaks_periods['Periods'].min() - 5, peaks_periods['Periods'].max() + 5), 
               y_range=(peaks_periods['Periods'].min() - 5, peaks_periods['Periods'].max() + 5),
               title=f"Exp: {exp_name.item()} - {fig_title} - {alignment.split('-')[0]}-{crit_point} aligned - 1.5*IQR",
               y_axis_label=y_axis_label)

    # Different colors for different conditions
    # Categorical palette
    n_cond = len(peaks_periods['Condition'].unique())
    if n_cond < 3:
        palette = d3['Category10'][3]  # Index  = # categories
    else:
        palette = d3['Category10'][n_cond] # Up to 10
    
    # Diagonal line: Pk+1 = Pk
    # P1 = [x1,y1]; P2 = [x2,p2]
    p.line([peaks_periods['Periods'].min(), peaks_periods['Periods'].max()],
           [peaks_periods['Periods'].min(), peaks_periods['Periods'].max()], line_width=2,
           line_dash='dashed', legend_label=diagonal_legend)
    
    # Plot scatter plot for Pk+1 vs Pk
    for i, condition in enumerate(peaks_periods['Condition'].unique()):
        sub_df = peaks_periods.loc[peaks_periods['Condition']==condition, :]
        source2 = ColumnDataSource(sub_df[['Signal', 'Periods', 'Period_shifted']].dropna())
        if ref == 'n':
            circles = Circle(x='Period_shifted', y = 'Periods', fill_alpha = 0.3,
                size=8, line_color="white", fill_color = palette[i])
        elif ref == 'n+1':
            circles = Circle(x='Periods', y = 'Period_shifted', fill_alpha = 0.3,
                size=8, line_color="white", fill_color = palette[i])
        circles_r = p.add_glyph(source_or_glyph=source2, glyph=circles)
        hover =HoverTool(renderers=[circles_r],
                        tooltips=[('Signal', '@Signal')])
        p.add_tools(hover)
    
    p.legend.title = 'Condition'
    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'
    # p.yaxis.axis_label_text_font_size = "20pt"    
    show(p)
    
def intensity_period():
    pass