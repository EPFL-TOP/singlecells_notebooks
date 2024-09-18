import os
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
from bokeh.layouts import gridplot
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc
from bokeh.server.server import Server
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.transform import linear_cmap
from bokeh.palettes import Greys256  # Grayscale palette

import nest_asyncio

import logging
logging.basicConfig(level=logging.DEBUG)

nest_asyncio.apply()
data={}
time_data={}
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

    time_data = {'exp_period':exp_period}

    for pos in range(num_pos):
        time_data[pos]=[timesteps[num_pos*frame+pos] for frame in range(num_frames)]

    period_diff={}
    for pos in time_data:
        if pos=='exp_period':continue
        for time in range(len(time_data[pos])-1):
            if time==0:continue
            try:
                period_diff[time].append(exp_period*time - time_data[pos][time])
                print('time ', time, ' exp_period*time ', exp_period*time, ' time_data[pos][time] ',time_data[pos][time])
            except KeyError:
                period_diff[time]=[]



def preprocess_image_pytorch(image_array):
    transform = ToTensorNormalize()
    image = transform(image_array)
    return image.unsqueeze(0)  # Add batch dimension


def process(file, low_crop, high_crop, model_detect, n=-9999):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    get_timelaps(file)

    current_file=os.path.join(file)
    time_lapse_path = Path(current_file)
    print('time_lapse_path = ',time_lapse_path)
    time_lapse = nd2.imread(time_lapse_path.as_posix())
    time_lapse = time_lapse.transpose(1,0,2,3,4)

    print(time_lapse.shape)#(81=t, 110=pos, 3, 512, 512)


    for pos_id, pos in enumerate(time_lapse):
        if n>0 and n==pos_id:break
        pos = pos.transpose(1,0,2,3)
        BF_images = pos[0]

        image_prepro = preprocess_image_pytorch(BF_images[0]).to(device)
        with torch.no_grad():
            predictions = model_detect(image_prepro)
            print(predictions)
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
                print('=========== ','pos{}_cell{}'.format(pos_id,idx))
            
                #rect = patches.Rectangle((x_min*0.85, y_min*0.85), x_max*1.15 - x_min*0.85, y_max*1.15 - y_min*0.85, linewidth=1, edgecolor='white', facecolor='none')
                #ax.add_patch(rect)

                #cropped_img = BF_images[0][int(y_min*0.85):int(y_max*1.15), int(x_min*0.85):int(x_max*1.15)]
                #ax.imshow(cropped_img, cmap='gray')
                #plt.show()


                intensities={}
                time=[]
                for ch_id, ch_img in enumerate(pos):
                    if ch_id == 0:
                        time=[i for i in range(len(ch_img))]
                        time=np.array(time)
                        continue
                    intensities[ch_id]=[]
                    for img in ch_img:
                        intensities[ch_id].append(img[int(y_min*low_crop):int(y_max*high_crop), int(x_min*low_crop):int(x_max*high_crop)].max())
                for ch in intensities:
                    intensities[ch]=np.array(intensities[ch])
                    max_value = np.max(intensities[ch])
                    min_value = np.min(intensities[ch])
                    intensity_normalized = (intensities[ch] - min_value)/(max_value-min_value)
                    intensities[ch]=intensity_normalized
                    if ch==2:intensities[ch]=intensity_normalized+1
                data['pos{}_cell{}'.format(pos_id,idx)]['time']=time
                data['pos{}_cell{}'.format(pos_id,idx)]['intensities']=intensities
    del time_lapse


def modify_doc(doc):
    print('time_data ',time_data)

    try:
        # Your Bokeh app code goes here, for example:
        layout = create_bokeh_layout()  # Make sure this function works as expected
        doc.add_root(layout)
        logging.info("App loaded successfully.")
    except Exception as e:
        logging.error(f"Error in modify_doc: {e}", exc_info=True)



def create_bokeh_layout():

    plots = []
    n_columns = 6
    color_mapper = LinearColorMapper(palette=Greys256, low=0, high=255)  # Adjust low and high according to your data range
    for pos in data:

        print(pos)
        image = data[pos]['img']
    
        p_img = figure(width=300, height=300, title=f"Image {pos}")
        p_img.image(image=[image], x=0, y=1, dw=1, dh=1, color_mapper=color_mapper)
        p_img.axis.visible = False
        p_img.grid.visible = False
        p_img.axis.visible = False
        p_img.grid.visible = False

        p_plot = figure(width=300, height=300, title=f"Intensity {pos}")
        for ch in data[pos]['intensities']:
            if ch==1:  p_plot.line(x=data[pos]['time'], y=data[pos]['intensities'][ch], line_color='blue')
            elif ch==2:p_plot.line(x=data[pos]['time'], y=data[pos]['intensities'][ch], line_color='black')

        plots.append(p_img)
        plots.append(p_plot)

    grid = gridplot(plots, ncols=n_columns)
    return grid




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
        




def load_model(model_path):
    num_classes_detect = 2
    model_detect = load_model_detect(model_path, num_classes_detect)
    return model_detect

