import os
import csv
import numpy  as np
import nd2
import nd2reader as nd2reader

from pathlib import Path
from skimage.filters import threshold_triangle, gaussian
from skimage.morphology import binary_opening, disk, binary_closing, white_tophat
from skimage.measure import label, regionprops, find_contours
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import torch.nn.functional as F2
import torch.nn as nn
from torchvision.models import ResNet18_Weights  # Import the appropriate weights enum
from torchvision import models


from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot, row, column
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Patch
from bokeh.models import CheckboxGroup, Div, Button, Tabs, TabPanel
from bokeh.plotting import figure, curdoc
from bokeh.server.server import Server
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.transform import linear_cmap
from bokeh.palettes import Greys256  # Grayscale palette

import nest_asyncio

import logging
from aicspylibczi import CziFile

#logging.basicConfig(level=logging.DEBUG)

nest_asyncio.apply()
data={}
time_data={}
source_file = None   # path of the file currently loaded into `data` (used for CSV export)

model_detect = None

class ToTensorNormalize:
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        else:
            image = F.pil_to_tensor(image).float()
        
        image = (image - image.min()) / (image.max() - image.min())
        return image


def load_model_detect(model_path, num_classes):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the checkpoint
    if device==torch.device('cpu'):
        checkpoint = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    #when no checkpoint
    #model.load_state_dict(torch.load(model_path, weights_only=True))

    model.to(device)
    model.eval()
    return model



def get_timelaps(file):
    current_file=os.path.join(file)
    time_lapse_path = Path(current_file)
    f = nd2.ND2File(time_lapse_path.as_posix())
    exp_period = f.experiment[0].parameters.durationMs/(f.experiment[0].count-1)
    f.close()

    stack = nd2reader.reader.ND2Reader(time_lapse_path.as_posix())
    metadata = stack.metadata
    num_frames = metadata['num_frames']
    num_pos = len(metadata["fields_of_view"])

    if num_pos*num_frames != len(metadata["z_coordinates"]):
        print('ERROR DIFFERENT NUMBER OF frames')

    timesteps = stack.timesteps.tolist()

    time_data['exp_period']=exp_period

    for pos in range(num_pos):
        time_data[pos]=[timesteps[num_pos*frame+pos] for frame in range(num_frames)]


def preprocess_image_pytorch(image_array):
    transform = ToTensorNormalize()
    image = transform(image_array)
    return image.unsqueeze(0)  # Add batch dimension


def process_czi_image(image, low_crop, high_crop, model_detect, n=-9999, show=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image_prepro = preprocess_image_pytorch(image).to(device)
    with torch.no_grad():
        predictions = model_detect(image_prepro)
        if show:
            print(predictions)
            fig, ax =plt.subplots()
            ax.imshow(image, cmap="gray")
        for idx, box in enumerate(predictions[0]['boxes']):
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            if float(predictions[0]['scores'][idx].cpu().numpy())<0.8:continue
            if (x_max-x_min)*(y_max-y_min)<150:continue
            if show:
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='white', facecolor='none')
                ax.add_patch(rect)
        if show:
            plt.show()

def process_czi(file, low_crop, high_crop, model_detect, seg_chan=2, n=-9999, verbose=False):
    global source_file
    source_file = file
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    czi  = CziFile(file)
    dims = czi.get_dims_shape()
    n_cells = len(dims)
    if verbose: print('N cells ',n_cells)
    if n_cells==1:
        n_cells=dims[0]['S'][1]
    for s in range(n_cells):
        #print(dims[s])
        n_time=dims[0]["T"][1]
        n_ch=dims[0]["C"][1]
        if verbose: print('processing scene ',s, '  ntime=',n_time, '  nchannels=',n_ch)
        img_t0, dim_t0 = czi.read_image(S=s,T=0,C=seg_chan)
        img_t0 = img_t0.squeeze()
        n_kept = 0
        image_prepro = preprocess_image_pytorch(img_t0).to(device)
        with torch.no_grad():
            predictions = model_detect(image_prepro)
            if verbose: print(predictions)
            for idx, box in enumerate(predictions[0]['boxes']):
                x_min, y_min, x_max, y_max = box.cpu().numpy()
                if float(predictions[0]['scores'][idx].cpu().numpy())<0.8:continue
                if (x_max-x_min)*(y_max-y_min)<150:continue
                data['pos{}_cell{}'.format(s,idx)]={}
                center = (x_min+(x_max-x_min)/2.,y_min+(y_max-y_min)/2.) 

                image=img_t0[int(y_min*low_crop):int(y_max*high_crop), int(x_min*low_crop):int(x_max*high_crop)]
                max_value = np.max(image)
                min_value = np.min(image)
                intensity_normalized = (image - min_value)/(max_value-min_value)*255
                intensity_normalized = intensity_normalized.astype(np.uint8)
                data['pos{}_cell{}'.format(s,idx)]['img']=intensity_normalized
                n_kept += 1


                intensities={}
                time=[]


                for ch in range(n_ch):
                    if ch==seg_chan:
                        time=[t for t in range(n_time)]
                        time=np.array(time)
                        continue
                    else:
                        intensities[ch]=[]
                        arr_t, dim_t=czi.read_image(S=s,C=ch, core=10)
                        image_t = arr_t.squeeze()
                        if verbose: print('----------------',image_t.shape)
                        for t in range(n_time):
                            intensities[ch].append(image_t[t][int(y_min*low_crop):int(y_max*high_crop), int(x_min*low_crop):int(x_max*high_crop)].max())

                for ch in intensities:
                    intensities[ch]=np.array(intensities[ch])
                    max_value = np.max(intensities[ch])
                    min_value = np.min(intensities[ch])
                    intensity_normalized = (intensities[ch] - min_value)/(max_value-min_value)
                    intensities[ch]=intensity_normalized
                    if ch==1:intensities[ch]=intensity_normalized+1
                data['pos{}_cell{}'.format(s,idx)]['time']=time
                data['pos{}_cell{}'.format(s,idx)]['intensities']=intensities
        print('[czi] scene {}: {} cells'.format(s, n_kept))
        #if s==5:break
    del czi

          



def process(file, low_crop, high_crop, model_detect, n=-9999, max_factor=1.5, verbose=False):
    global source_file
    source_file = file
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    get_timelaps(file)

    current_file=os.path.join(file)
    time_lapse_path = Path(current_file)
    if verbose: print('time_lapse_path = ',time_lapse_path)
    time_lapse = nd2.imread(time_lapse_path.as_posix())
    time_lapse = time_lapse.transpose(1,0,2,3,4)

    if verbose: print(time_lapse.shape)#(81=t, 110=pos, 3, 512, 512)

    for pos_id, pos in enumerate(time_lapse):
        if n>0 and n==pos_id:break
        pos = pos.transpose(1,0,2,3)
        BF_images = pos[0]

        n_kept = 0
        image_prepro = preprocess_image_pytorch(BF_images[0]).to(device)
        with torch.no_grad():
            predictions = model_detect(image_prepro)
            if verbose: print(predictions)
            for idx, box in enumerate(predictions[0]['boxes']):
                x_min, y_min, x_max, y_max = box.cpu().numpy()
                if float(predictions[0]['scores'][idx].cpu().numpy())<0.8:continue
                if (x_max-x_min)*(y_max-y_min)<150:continue
                data['pos{}_cell{}'.format(pos_id,idx)]={}

                center = (x_min+(x_max-x_min)/2.,y_min+(y_max-y_min)/2.)
                target_size = (100, 100)
                #cropped_image = BF_images[0][int(center[1]-target_size[1]/2):int(center[1]+target_size[1]/2), int(center[0]-target_size[0]/2):int(center[0]+target_size[0]/2)]
                image=BF_images[0][int(y_min*low_crop):int(y_max*high_crop), int(x_min*low_crop):int(x_max*high_crop)]
                #image = cropped_image
                max_value = np.max(image)
                min_value = np.min(image)
                intensity_normalized = (image - min_value)/(max_value-min_value)*255
                intensity_normalized = intensity_normalized.astype(np.uint8)
                data['pos{}_cell{}'.format(pos_id,idx)]['img']=intensity_normalized
                n_kept += 1

                #rect = patches.Rectangle((x_min*0.85, y_min*0.85), x_max*1.15 - x_min*0.85, y_max*1.15 - y_min*0.85, linewidth=1, edgecolor='white', facecolor='none')
                #ax.add_patch(rect)

                #cropped_img = BF_images[0][int(y_min*0.85):int(y_max*1.15), int(x_min*0.85):int(x_max*1.15)]
                #ax.imshow(cropped_img, cmap='gray')
                #plt.show()


                intensities={}
                time=[]
                for ch_id, ch_img in enumerate(pos):
                    if ch_id == 0:
                        #time=[i for i in range(len(ch_img))]
                        time=[t/60000. for t in time_data[pos_id]]
                        time=np.array(time)
                        continue
                    intensities[ch_id]=[]
                    for img in ch_img:
                        max_int = img[int(y_min*low_crop):int(y_max*high_crop), int(x_min*low_crop):int(x_max*high_crop)].max()

                        if max_int/65536>0.8:
                            intensities[ch_id].append(intensities[ch_id][-1])
                        elif max_int>intensities[ch_id][-1]*max_factor: 
                            intensities[ch_id].append(intensities[ch_id][-1])
                        else:
                            intensities[ch_id].append(max_int)

                for ch in intensities:
                    intensities[ch]=np.array(intensities[ch])
                    max_value = np.max(intensities[ch])
                    min_value = np.min(intensities[ch])
                    intensity_normalized = (intensities[ch] - min_value)/(max_value-min_value)
                    intensities[ch]=intensity_normalized
                    if ch==2:intensities[ch]=intensity_normalized+1
                data['pos{}_cell{}'.format(pos_id,idx)]['time']=time
                data['pos{}_cell{}'.format(pos_id,idx)]['intensities']=intensities
        print('[nd2] pos {}: {} cells'.format(pos_id, n_kept))
    del time_lapse

def build_cells_dashboard(top_row=None, max_cells_per_row=6, base_fig_size=180):
    """Build the interactive cell-preview dashboard from the global `data` dict.

    - cells are grouped per position inside a bordered, colour-coded card;
    - positions are packed several per row (a position is never split across
      rows) up to ``max_cells_per_row`` cells per row;
    - every position has a "keep" checkbox; kept positions are listed in a
      separate tab and can be dumped to CSV (one row per position, keep flag);
    - "Zoom +/-" buttons rescale every image / intensity plot on the page.

    Returns a Bokeh layout. ``top_row`` is placed above the cell grid (used by
    the nd2 dashboard for the timing-deviation plots).
    """
    color_mapper = LinearColorMapper(palette=Greys256, low=0, high=255)
    line_colors = ['blue', 'black', 'green', 'red', 'purple']

    def _pos_cell(key):
        # keys look like 'pos{pos_id}_cell{idx}'
        pos_part, cell_part = key.split('_')
        return int(pos_part[3:]), int(cell_part[4:])

    # ---- group the detected cells by their position ----------------------
    cells_by_pos = {}
    for key in data:
        if 'img' not in data[key]:
            continue
        pos_id, _ = _pos_cell(key)
        cells_by_pos.setdefault(pos_id, []).append(key)
    ordered_positions = sorted(cells_by_pos)

    # alternating colours so neighbouring positions are visually distinct
    band_colors = ["#1f77b4", "#ff7f0e"]   # strong colour for the header band
    tint_colors = ["#eaf2fb", "#fff3e6"]   # matching light figure background

    # ---- shared state ----------------------------------------------------
    selected_positions = set()
    all_figs = []                       # every figure, so the zoom buttons can resize them
    fig_size = {'v': base_fig_size}
    selected_div = Div(text="<b>No position selected yet.</b>", width=500)
    export_status = Div(text="", width=700)

    def refresh_selected():
        if selected_positions:
            items = "".join(
                f"<li>position <b>{p}</b> &nbsp;({len(cells_by_pos[p])} cells)</li>"
                for p in sorted(selected_positions))
            selected_div.text = (f"<b>{len(selected_positions)} position(s) kept:</b>"
                                 f"<ul>{items}</ul>")
        else:
            selected_div.text = "<b>No position selected yet.</b>"

    def make_checkbox_callback(pos_id):
        def _cb(attr, old, new):
            if new:                       # non-empty active list -> ticked
                selected_positions.add(pos_id)
            else:
                selected_positions.discard(pos_id)
            refresh_selected()
        return _cb

    # ---- one bordered, colour-coded card per position --------------------
    def make_position_card(order_idx, pos_id):
        tint = tint_colors[order_idx % 2]
        band = band_colors[order_idx % 2]

        cell_units = []
        for key in cells_by_pos[pos_id]:
            _, cell_idx = _pos_cell(key)

            p_img = figure(width=fig_size['v'], height=fig_size['v'],
                           title=f"cell {cell_idx}", toolbar_location=None)
            p_img.image(image=[data[key]['img']], x=0, y=1, dw=1, dh=1, color_mapper=color_mapper)
            p_img.axis.visible = False
            p_img.grid.visible = False
            p_img.background_fill_color = tint
            p_img.border_fill_color = tint

            p_plot = figure(width=fig_size['v'], height=fig_size['v'], toolbar_location=None)
            ints = data[key].get('intensities', {})
            for i, ch in enumerate(sorted(ints)):
                p_plot.line(x=data[key]['time'], y=ints[ch],
                            line_color=line_colors[i % len(line_colors)])
            p_plot.background_fill_color = tint
            p_plot.border_fill_color = tint

            all_figs.append(p_img)
            all_figs.append(p_plot)
            cell_units.append(column(p_img, p_plot))   # image stacked over its intensity plot

        header = Div(
            text=(f"<div style='background:{band}; color:white; padding:3px 10px; "
                  f"font-weight:bold; border-radius:4px;'>Position {pos_id} "
                  f"&middot; {len(cells_by_pos[pos_id])} cells</div>"),
            width=max(150, base_fig_size))
        keep_cb = CheckboxGroup(labels=[f"keep position {pos_id}"], active=[])
        keep_cb.on_change('active', make_checkbox_callback(pos_id))

        # the whole card is boxed and tinted so it is unambiguous which label /
        # checkbox belongs to which set of cells
        return column(
            row(header, keep_cb),
            row(*cell_units),
            styles={'border': f'2px solid {band}', 'border-radius': '8px',
                    'padding': '6px', 'margin': '6px', 'background': tint},
        )

    # ---- pack positions onto rows; never split a position ----------------
    rows, current, count = [], [], 0
    for order_idx, pos_id in enumerate(ordered_positions):
        k = len(cells_by_pos[pos_id])
        if current and count + k > max_cells_per_row:
            rows.append(row(*current))
            current, count = [], 0
        current.append(make_position_card(order_idx, pos_id))
        count += k
    if current:
        rows.append(row(*current))
    cells_layout = column(*rows)

    # ---- zoom controls ---------------------------------------------------
    def zoom(factor):
        fig_size['v'] = int(max(70, min(600, fig_size['v'] * factor)))
        for f in all_figs:
            f.width = fig_size['v']
            f.height = fig_size['v']
    zoom_in = Button(label="Zoom +", width=90)
    zoom_out = Button(label="Zoom -", width=90)
    zoom_in.on_click(lambda: zoom(1.25))
    zoom_out.on_click(lambda: zoom(0.8))

    # ---- CSV export ------------------------------------------------------
    def _csv_path():
        if source_file:
            p = Path(source_file)
            return str(p.with_name(p.stem + '_positions.csv'))
        return os.path.join(os.getcwd(), 'positions.csv')

    def export_csv():
        out = _csv_path()
        with open(out, 'w', newline='') as fh:
            writer = csv.writer(fh)
            # extra categories can be added as further columns later
            writer.writerow(['position', 'n_cells', 'keep'])
            for pos in ordered_positions:
                writer.writerow([pos, len(cells_by_pos[pos]), pos in selected_positions])
        export_status.text = f"Saved <code>{out}</code> ({len(ordered_positions)} positions)"
        print('Saved CSV:', out)
    export_button = Button(label="Export positions CSV", button_type="success", width=200)
    export_button.on_click(export_csv)

    print_button = Button(label="Print kept positions", button_type="primary", width=180)
    print_button.on_click(lambda: print("kept positions:", sorted(selected_positions)))

    controls = row(Div(text="<b>Zoom plots:</b>", width=80), zoom_in, zoom_out)

    tabs = Tabs(tabs=[
        TabPanel(child=cells_layout, title="Cells by position"),
        TabPanel(child=column(selected_div, row(print_button, export_button), export_status),
                 title="Selected positions"),
    ])

    parts = []
    if top_row is not None:
        parts.append(top_row)
    parts.append(controls)
    parts.append(tabs)
    return column(*parts)


def modify_doc_czi(doc):
    try:
        doc.add_root(build_cells_dashboard())
        logging.info("App loaded successfully.")
    except Exception as e:
        logging.error(f"Error in modify_doc_czi: {e}", exc_info=True)


def modify_doc(doc):

    def make_period_plots():

        exp_period=time_data['exp_period']

        period_diff={}
        for pos in time_data:
            if pos=='exp_period':continue
            for time in range(len(time_data[pos])):
                try:
                    period_diff[time].append(time_data[pos][time] - exp_period*time -  time_data[pos][0])
                except KeyError:
                    period_diff[time]=[]
                    period_diff[time].append(time_data[pos][time] - exp_period*time -  time_data[pos][0])

        period_mean = [0 for i in range(len(period_diff))]
        period_std = [0 for i in range(len(period_diff))]
        time = [i*exp_period/60000. for i in range(len(period_diff))]
        time = np.array(time)
        for p in period_diff:
            period_mean[p]=np.mean(period_diff[p])/1000.
            period_std[p]=np.std(period_diff[p])/1000.

        period_mean = np.array(period_mean)
        period_std = np.array(period_std)

        p_period_vs_frame = figure(width=500, height=400, title=f"Average deviation from expectations", x_axis_label='time [min]', y_axis_label='Deviation [sec]')
        p_period_vs_frame.line(x=time, y=period_mean, line_color='blue')
        x_period_vs_frame=np.hstack((time, time[::-1]))
        y_period_vs_frame=np.hstack((period_mean - period_std, (period_mean + period_std)[::-1]))
        source_period_vs_frame = ColumnDataSource(dict(x=x_period_vs_frame, y=y_period_vs_frame))
        glyph = Patch(x="x", y="y", fill_color="#a6cee3", fill_alpha=0.3, line_color="#a6cee3", line_alpha=0.3)
        p_period_vs_frame.add_glyph(source_period_vs_frame, glyph)


        period_diff={}
        for pos in time_data:
            if pos=='exp_period':continue
            for time in range(len(time_data[pos])):
                try:
                    period_diff[pos].append(time_data[pos][time] - exp_period*time -  time_data[pos][0])
                except KeyError:
                    period_diff[pos]=[]
                    period_diff[pos].append(time_data[pos][time] - exp_period*time -  time_data[pos][0])


        period_mean = [0 for i in range(len(period_diff))]
        period_std = [0 for i in range(len(period_diff))]
        position = [i for i in range(len(period_diff))]
        position = np.array(position)
        for p in period_diff:
            period_mean[p]=np.mean(period_diff[p])/1000.
            period_std[p]=np.std(period_diff[p])/1000.

        period_mean = np.array(period_mean)
        period_std = np.array(period_std)

        p_period_vs_pos = figure(width=500, height=400, title=f"Average deviation from expectations", x_axis_label='position', y_axis_label='Deviation [sec]')
        p_period_vs_pos.line(x=position, y=period_mean, line_color='blue')
        x_period_vs_pos=np.hstack((position, position[::-1]))
        y_period_vs_pos=np.hstack((period_mean - period_std, (period_mean + period_std)[::-1]))
        source_period_vs_pos = ColumnDataSource(dict(x=x_period_vs_pos, y=y_period_vs_pos))
        glyph2 = Patch(x="x", y="y", fill_color="#a6cee3", fill_alpha=0.3, line_color="#a6cee3", line_alpha=0.3)
        p_period_vs_pos.add_glyph(source_period_vs_pos, glyph2)

        return row(p_period_vs_frame, p_period_vs_pos)

    try:
        layout = build_cells_dashboard(top_row=make_period_plots())
        doc.add_root(layout)
        logging.info("App loaded successfully.")
    except Exception as e:
        logging.error(f"Error in modify_doc: {e}", exc_info=True)



def run_server():
    # Bind the server to localhost and allow access from the specified origin
    server = Server({'/': modify_doc}, num_procs=1, port=5006, allow_websocket_origin=["localhost:5006"])

    # Start the Bokeh server
    server.start()
    
    # Show the app in a new browser window
    server.io_loop.add_callback(server.show, "/")
    
    # Start the IOLoop without a conflict (since nest_asyncio is applied)
    try:
        server.io_loop.start()
    except RuntimeError:
        # If the loop is already running, continue without restarting it
        print('loop is already running')
        pass
        

def run_server_czi():
    # Bind the server to localhost and allow access from the specified origin
    server = Server({'/': modify_doc_czi}, num_procs=1, port=5010, allow_websocket_origin=["localhost:5010"])

    # Start the Bokeh server
    server.start()
    
    # Show the app in a new browser window
    server.io_loop.add_callback(server.show, "/")
    
    # Start the IOLoop without a conflict (since nest_asyncio is applied)
    try:
        server.io_loop.start()
    except RuntimeError:
        # If the loop is already running, continue without restarting it
        print('loop is already running')
        pass



def load_model(model_path):
    num_classes_detect = 2
    model_detect = load_model_detect(model_path, num_classes_detect)
    return model_detect

