import os
import csv
import threading
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
from bokeh.models import CheckboxGroup, Div, Button, Tabs, TabPanel, Select
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


# ====================== lazy / streaming nd2 reading ========================
# `process()` above loads the whole file with nd2.imread (all positions/times in
# RAM at once). The helpers below instead read ONE position at a time via dask,
# so they scale to very large (100+ GB) files.

def _position_dim(dims):
    """Return the name of the multi-position axis in an nd2 xarray, or None."""
    for cand in ('P', 'S', 'M', 'Position', 'position', 'points'):
        if cand in dims:
            return cand
    return None


def _read_position_block(xarr, pos_dim, pos_id):
    """Lazily read a single position and return it as a (T, C, Y, X) numpy array.

    Only this position is pulled into memory (a few hundred MB), never the whole
    file. Any stray axis (e.g. a Z stack) is reduced to its first index."""
    sub = xarr.isel({pos_dim: pos_id}) if pos_dim is not None else xarr
    for d in list(sub.dims):                 # drop anything that is not T/C/Y/X
        if d not in ('T', 'C', 'Y', 'X'):
            sub = sub.isel({d: 0})
    for d in ('T', 'C'):                      # guarantee a time and a channel axis exist
        if d not in sub.dims:
            sub = sub.expand_dims(d)
    sub = sub.transpose('T', 'C', 'Y', 'X')
    return np.asarray(sub.to_numpy())


def _detect_and_store(block, pos_id, low_crop, high_crop, model_detect, device,
                      max_factor=1.5, time_minutes=None):
    """Detect cells in one position block (T, C, Y, X) and store them in `data`.

    Channel 0 is bright-field (used for detection and the thumbnail); channels
    >= 1 are fluorescence channels whose per-cell max intensity is tracked over
    time. Returns the number of cells kept. Mirrors `process()`'s semantics but
    is safe on the first frame (the original indexed an empty list)."""
    n_time, n_ch = block.shape[0], block.shape[1]
    bf_t0 = block[0, 0]                       # bright-field, first timepoint

    image_prepro = preprocess_image_pytorch(bf_t0).to(device)
    n_kept = 0
    with torch.no_grad():
        predictions = model_detect(image_prepro)
        for idx, box in enumerate(predictions[0]['boxes']):
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            if float(predictions[0]['scores'][idx].cpu().numpy()) < 0.8:
                continue
            if (x_max - x_min) * (y_max - y_min) < 150:
                continue

            y0, y1 = int(y_min * low_crop), int(y_max * high_crop)
            x0, x1 = int(x_min * low_crop), int(x_max * high_crop)

            key = 'pos{}_cell{}'.format(pos_id, idx)
            data[key] = {}

            crop = bf_t0[y0:y1, x0:x1]
            mn, mx = float(np.min(crop)), float(np.max(crop))
            if mx > mn:
                thumb = ((crop - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                thumb = np.zeros_like(crop, dtype=np.uint8)
            data[key]['img'] = thumb

            intensities = {}
            for ch in range(1, n_ch):
                series = []
                for t in range(n_time):
                    v = float(block[t, ch, y0:y1, x0:x1].max())
                    if series:                # carry previous value on saturation / spikes
                        prev_v = series[-1]
                        if v / 65536. > 0.8 or v > prev_v * max_factor:
                            v = prev_v
                    series.append(v)
                series = np.array(series, dtype=float)
                smn, smx = series.min(), series.max()
                series = (series - smn) / (smx - smn) if smx > smn else series * 0.
                if ch == 2:
                    series = series + 1       # vertical offset so the 2nd channel is readable
                intensities[ch] = series

            data[key]['time'] = (np.array(time_minutes) if time_minutes is not None
                                 else np.arange(n_time))
            data[key]['intensities'] = intensities
            n_kept += 1
    return n_kept


def _open_nd2_lazy(file, verbose=False):
    """Open an nd2 as a lazy (dask-backed) xarray. Returns (file_handle, xarr,
    pos_dim, n_pos). Caller must close the handle."""
    f = nd2.ND2File(Path(os.path.join(file)).as_posix())
    xarr = f.to_xarray(delayed=True, squeeze=True)
    if verbose:
        print('nd2 sizes:', dict(f.sizes), '| xarray dims:', tuple(xarr.dims))
    pos_dim = _position_dim(xarr.dims)
    n_pos = int(xarr.sizes[pos_dim]) if pos_dim is not None else 1
    return f, xarr, pos_dim, n_pos


def process_lazy(file, low_crop, high_crop, model_detect, n=-9999, max_factor=1.5, verbose=False):
    """Memory-light drop-in replacement for `process()`.

    Reads the nd2 one position at a time (via ``ND2File.to_xarray(delayed=True)``)
    instead of loading the whole file, so it works on very large files. Fills the
    same global `data` / `time_data` dicts, so `run_server()` displays the result
    unchanged. For live display while reading, use `run_server_stream()` instead."""
    global source_file
    source_file = file
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    get_timelaps(file)

    f, xarr, pos_dim, n_pos = _open_nd2_lazy(file, verbose=verbose)
    try:
        for pos_id in range(n_pos):
            if n > 0 and pos_id >= n:
                break
            block = _read_position_block(xarr, pos_dim, pos_id)
            tmin = [t / 60000. for t in time_data.get(pos_id, [])] or None
            n_kept = _detect_and_store(block, pos_id, low_crop, high_crop,
                                       model_detect, device, max_factor, tmin)
            del block
            print('[nd2] pos {}: {} cells'.format(pos_id, n_kept))
    finally:
        f.close()

# ============================ dashboard building ============================
# alternating colours so neighbouring positions are visually distinct
_BAND_COLORS = ["#1f77b4", "#ff7f0e"]   # strong colour for the header band
_TINT_COLORS = ["#eaf2fb", "#fff3e6"]   # matching light figure background
_LINE_COLORS = ['blue', 'black', 'green', 'red', 'purple']


def _pos_cell(key):
    # keys look like 'pos{pos_id}_cell{idx}'
    pos_part, cell_part = key.split('_')
    return int(pos_part[3:]), int(cell_part[4:])


def _group_cells_by_pos():
    """Group the keys of the global `data` dict by position id. Iterates a
    snapshot of the keys so it is safe to call while a background worker is still
    adding cells."""
    cells_by_pos = {}
    for key in list(data):
        if 'img' not in data[key]:
            continue
        pos_id, _ = _pos_cell(key)
        cells_by_pos.setdefault(pos_id, []).append(key)
    return cells_by_pos


def _make_position_card(order_idx, pos_id, cells_by_pos, ctx):
    """Build one bordered, colour-coded card holding every cell of a position.

    ``ctx`` carries the per-dashboard shared state: ``fig_size`` (mutable dict
    used by the zoom buttons), ``color_mapper``, ``all_figs`` (list every figure
    is registered in), ``selected_positions`` and ``on_keep`` (checkbox callback
    factory)."""
    tint = _TINT_COLORS[order_idx % 2]
    band = _BAND_COLORS[order_idx % 2]
    fig_size = ctx['fig_size']
    color_mapper = ctx['color_mapper']

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
                        line_color=_LINE_COLORS[i % len(_LINE_COLORS)])
        p_plot.background_fill_color = tint
        p_plot.border_fill_color = tint

        ctx['all_figs'].append(p_img)
        ctx['all_figs'].append(p_plot)
        cell_units.append(column(p_img, p_plot))   # image stacked over its intensity plot

    header = Div(
        text=(f"<div style='background:{band}; color:white; padding:3px 10px; "
              f"font-weight:bold; border-radius:4px;'>Position {pos_id} "
              f"&middot; {len(cells_by_pos[pos_id])} cells</div>"),
        width=max(150, fig_size['v']))
    keep_cb = CheckboxGroup(labels=[f"keep position {pos_id}"],
                            active=[0] if pos_id in ctx['selected_positions'] else [])
    keep_cb.on_change('active', ctx['on_keep'](pos_id))

    # the whole card is boxed and tinted so it is unambiguous which label /
    # checkbox belongs to which set of cells
    return column(
        row(header, keep_cb),
        row(*cell_units),
        styles={'border': f'2px solid {band}', 'border-radius': '8px',
                'padding': '6px', 'margin': '6px', 'background': tint},
    )


def _pack_position_cards(position_ids, cells_by_pos, ctx, color_index=None):
    """Pack positions onto rows, never splitting a position, up to
    ``ctx['max_cells_per_row']`` cells per row. ``color_index`` maps a position
    id to the index used for its alternating colour (keeps the colour stable
    across pages); defaults to the enumeration order."""
    rows, current, count = [], [], 0
    mcr = ctx['max_cells_per_row']
    for i, pos_id in enumerate(position_ids):
        k = len(cells_by_pos[pos_id])
        oidx = color_index.get(pos_id, i) if color_index is not None else i
        if current and count + k > mcr:
            rows.append(row(*current))
            current, count = [], 0
        current.append(_make_position_card(oidx, pos_id, cells_by_pos, ctx))
        count += k
    if current:
        rows.append(row(*current))
    return column(*rows)


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
    # ---- shared per-dashboard state --------------------------------------
    selected_positions = set()
    ctx = {
        'fig_size': {'v': base_fig_size},   # mutated by the zoom buttons
        'color_mapper': LinearColorMapper(palette=Greys256, low=0, high=255),
        'all_figs': [],                     # every figure, so zoom can resize them
        'selected_positions': selected_positions,
        'max_cells_per_row': max_cells_per_row,
    }
    cells_by_pos = _group_cells_by_pos()
    ordered_positions = sorted(cells_by_pos)

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
    ctx['on_keep'] = make_checkbox_callback

    # group cells into bordered cards and pack several positions per row
    cells_layout = _pack_position_cards(ordered_positions, cells_by_pos, ctx)

    # ---- zoom controls ---------------------------------------------------
    def zoom(factor):
        ctx['fig_size']['v'] = int(max(70, min(600, ctx['fig_size']['v'] * factor)))
        for f in ctx['all_figs']:
            f.width = ctx['fig_size']['v']
            f.height = ctx['fig_size']['v']
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


def build_stream_dashboard(doc, stream_state, cells_per_page=24, max_cells_per_row=6,
                           base_fig_size=180, refresh_ms=1500):
    """Paginated dashboard that fills in live while a background worker reads the
    file. Only one page worth of cells is materialised in the browser at a time,
    so the page stays light even for hundreds of positions. A periodic callback
    (on the Bokeh IO loop) reads the global `data`/`stream_state` the worker
    thread fills, refreshes the status/page label every tick, and rebuilds the
    cards only when the visible page actually changes."""
    selected_positions = set()
    ctx = {
        'fig_size': {'v': base_fig_size},
        'color_mapper': LinearColorMapper(palette=Greys256, low=0, high=255),
        'all_figs': [],
        'selected_positions': selected_positions,
        'max_cells_per_row': max_cells_per_row,
    }

    status_div    = Div(text="<b>Starting…</b>", width=700)
    page_label    = Div(text="", width=300)
    selected_div  = Div(text="<b>No position selected yet.</b>", width=500)
    export_status = Div(text="", width=700)
    page_container = column()
    page = {'i': 0, 'follow': True, 'rendered': None, 'size': cells_per_page}

    def refresh_selected():
        if selected_positions:
            cbp = _group_cells_by_pos()
            items = "".join(
                f"<li>position <b>{p}</b> &nbsp;({len(cbp.get(p, []))} cells)</li>"
                for p in sorted(selected_positions))
            selected_div.text = (f"<b>{len(selected_positions)} position(s) kept:</b>"
                                 f"<ul>{items}</ul>")
        else:
            selected_div.text = "<b>No position selected yet.</b>"

    def make_checkbox_callback(pos_id):
        def _cb(attr, old, new):
            if new:
                selected_positions.add(pos_id)
            else:
                selected_positions.discard(pos_id)
            refresh_selected()
        return _cb
    ctx['on_keep'] = make_checkbox_callback

    # ---- pagination by *cells* (never splitting a position) --------------
    def compute_pages(positions, cells_by_pos):
        budget = page['size']
        pages, cur, cnt = [], [], 0
        for p in positions:
            k = len(cells_by_pos[p])
            if cur and cnt + k > budget:     # a position with > budget cells gets its own page
                pages.append(cur)
                cur, cnt = [], 0
            cur.append(p)
            cnt += k
        if cur:
            pages.append(cur)
        return pages or [[]]

    def current_view():
        cells_by_pos = _group_cells_by_pos()
        positions = sorted(cells_by_pos)
        pages = compute_pages(positions, cells_by_pos)
        i = (len(pages) - 1) if page['follow'] else min(max(0, page['i']), len(pages) - 1)
        return cells_by_pos, positions, pages, i

    def update_label(i, total, page_positions):
        if page_positions:
            page_label.text = (f"Page {i + 1}/{total} — positions "
                               f"{page_positions[0]}–{page_positions[-1]}")
        else:
            page_label.text = f"Page {i + 1}/{total} — (no cells yet)"
        prev_btn.disabled = (i <= 0)
        next_btn.disabled = (i >= total - 1)

    def render_page():
        cells_by_pos, positions, pages, i = current_view()
        page['i'] = i
        color_index = {p: idx for idx, p in enumerate(positions)}   # stable colour per position
        page_positions = pages[i]
        ctx['all_figs'] = []                                        # only current-page figures stay live
        page_container.children = [
            _pack_position_cards(page_positions, cells_by_pos, ctx, color_index)]
        page['rendered'] = page_positions
        update_label(i, len(pages), page_positions)

    def refresh():
        st = stream_state
        verb = 'Done' if st.get('done') else 'Processing'
        msg = (f"<b>{verb}:</b> {st.get('pos_done', 0)}/{st.get('pos_total', '?')} "
               f"positions read · {len(_group_cells_by_pos())} with cells · "
               f"{len(selected_positions)} kept")
        if st.get('error'):
            msg += f" · <span style='color:red'>ERROR: {st['error']}</span>"
        status_div.text = msg
        # rebuild cards only when the visible slice changed, but always keep the
        # page label / total fresh so "/N" grows while you sit on an earlier page
        _, _, pages, i = current_view()
        page['i'] = i
        page_positions = pages[i]
        if page_positions != page['rendered']:
            render_page()
        else:
            update_label(i, len(pages), page_positions)

    # ---- navigation ------------------------------------------------------
    def go_prev():
        page['follow'] = False
        page['i'] = max(0, page['i'] - 1)
        render_page()
    def go_next():
        page['follow'] = False
        page['i'] += 1                     # current_view() clamps to the last page
        render_page()
    def go_latest():
        page['follow'] = True
        render_page()
    prev_btn   = Button(label="◀ Prev", width=80)
    next_btn   = Button(label="Next ▶", width=80)
    latest_btn = Button(label="Follow latest", button_type="primary", width=120)
    prev_btn.on_click(go_prev)
    next_btn.on_click(go_next)
    latest_btn.on_click(go_latest)

    # ---- cells-per-page dropdown -----------------------------------------
    size_options = ['6', '12', '24', '48', '100', 'all']
    size_select = Select(title="Cells/page",
                         value=str(cells_per_page) if str(cells_per_page) in size_options else '24',
                         options=size_options, width=90)
    def on_size_change(attr, old, new):
        anchor = page['rendered'][0] if page['rendered'] else None   # keep the top position in view
        page['size'] = 10 ** 9 if new == 'all' else int(new)
        page['follow'] = False
        if anchor is not None:
            cells_by_pos = _group_cells_by_pos()
            for idx, pg in enumerate(compute_pages(sorted(cells_by_pos), cells_by_pos)):
                if anchor in pg:
                    page['i'] = idx
                    break
        render_page()
    size_select.on_change('value', on_size_change)

    # ---- zoom (batched so every plot resizes in a single update) ---------
    def zoom(factor):
        ctx['fig_size']['v'] = int(max(70, min(600, ctx['fig_size']['v'] * factor)))
        v = ctx['fig_size']['v']
        doc.hold()                          # collect all the size changes...
        try:
            for f in ctx['all_figs']:
                f.width = v
                f.height = v
        finally:
            doc.unhold()                    # ...then push them to the browser at once
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
        cells_by_pos = _group_cells_by_pos()
        out = _csv_path()
        with open(out, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['position', 'n_cells', 'keep'])
            for pos in sorted(cells_by_pos):
                writer.writerow([pos, len(cells_by_pos[pos]), pos in selected_positions])
        export_status.text = f"Saved <code>{out}</code> ({len(cells_by_pos)} positions)"
        print('Saved CSV:', out)
    export_button = Button(label="Export positions CSV", button_type="success", width=200)
    export_button.on_click(export_csv)

    print_button = Button(label="Print kept positions", button_type="primary", width=180)
    print_button.on_click(lambda: print("kept positions:", sorted(selected_positions)))

    nav = row(prev_btn, next_btn, latest_btn, size_select, page_label)
    controls = row(Div(text="<b>Zoom plots:</b>", width=80), zoom_in, zoom_out)
    tabs = Tabs(tabs=[
        TabPanel(child=column(nav, page_container), title="Cells by position"),
        TabPanel(child=column(selected_div, row(print_button, export_button), export_status),
                 title="Selected positions"),
    ])

    render_page()                                  # initial (likely empty) page
    doc.add_periodic_callback(refresh, refresh_ms)
    return column(status_div, controls, tabs)


def _stream_worker(file, low_crop, high_crop, model_detect, n, max_factor, state, verbose):
    """Background-thread body: lazily read the nd2 and fill `data` position by
    position, updating `state` for the UI. Touches only plain dicts, never Bokeh
    models (the periodic callback on the IO loop does all rendering)."""
    global source_file
    source_file = file
    try:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        get_timelaps(file)
        f, xarr, pos_dim, n_pos = _open_nd2_lazy(file, verbose=verbose)
        state['pos_total'] = min(n, n_pos) if n > 0 else n_pos
        try:
            for pos_id in range(n_pos):
                if n > 0 and pos_id >= n:
                    break
                block = _read_position_block(xarr, pos_dim, pos_id)
                tmin = [t / 60000. for t in time_data.get(pos_id, [])] or None
                n_kept = _detect_and_store(block, pos_id, low_crop, high_crop,
                                           model_detect, device, max_factor, tmin)
                del block
                state['pos_done'] = pos_id + 1
                state['dirty'] = True
                print('[nd2] pos {}: {} cells'.format(pos_id, n_kept))
        finally:
            f.close()
    except Exception as e:
        state['error'] = repr(e)
        logging.error("stream worker failed: %s", e, exc_info=True)
    finally:
        state['done'] = True
        state['dirty'] = True


def run_server_stream(file, low_crop, high_crop, model_detect, n=-9999, max_factor=1.5,
                      port=5007, cells_per_page=24, max_cells_per_row=6, base_fig_size=180,
                      verbose=False):
    """Start a Bokeh server that reads the nd2 in the background and displays
    positions as they are detected, one page at a time.

    Usage from the notebook (no separate process() call needed):
        model = prev.load_model(model_path)
        prev.run_server_stream(file, low_crop, high_crop, model)
    """
    data.clear()
    time_data.clear()
    state = {'pos_total': '?', 'pos_done': 0, 'done': False, 'dirty': True,
             'error': None, 'started': False}

    def modify(doc):
        doc.add_root(build_stream_dashboard(doc, state, cells_per_page=cells_per_page,
                                            max_cells_per_row=max_cells_per_row,
                                            base_fig_size=base_fig_size))
        if not state['started']:               # launch the reader once, on first connect
            state['started'] = True
            threading.Thread(
                target=_stream_worker,
                args=(file, low_crop, high_crop, model_detect, n, max_factor, state, verbose),
                daemon=True).start()
        logging.info("Stream app loaded.")

    server = Server({'/': modify}, num_procs=1, port=port,
                    allow_websocket_origin=[f"localhost:{port}"])
    server.start()
    server.io_loop.add_callback(server.show, "/")
    try:
        server.io_loop.start()
    except RuntimeError:
        print('loop is already running')
        pass


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

