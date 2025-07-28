from aicspylibczi import CziFile
import numpy as np
import sys, os
import tifffile
import matplotlib.pyplot as plt
sys.path.append(os.path.join("utils"))
import utils_preview as prev
model=prev.load_model(r"C:\Users\helsens\software\singleCell_catalog\cell_detection_model.pth")

path=r"H:\PROJECTS-03\Feyza\forClement\New-01.czi"
path=r"Y:\raw_data\microscopy\cell_culture\250711-confiner\New-02.czi"

prev.process_czi(path, low_crop=0.85, high_crop=1.15, model_detect=model, n=-9999)
prev.run_server_czi()
sys.exit(3)


czi = CziFile(path)
dims = czi.get_dims_shape()
n_cells = len(dims)
scene_count = czi.get_dims_shape()[0]['S']
print('scne ',scene_count)
print('N cells ',n_cells)
print('dims ',dims)
#sys.exit(3)

def rescale(image):
	imin, imax = image.min(), image.max()
	print(imin, '  ',imax)
	if imin==imax:
		return np.zeros_like(image, dtype=np.uint8)
	scaled = 255 * (image.astype(np.float32)-imin)/(imin-imax)
	print (scaled.min(), '  ', scaled.max())
	return scaled.astype(np.uint8)


if len(dims)>1:
	for s in range(len(dims)):
		n_time=dims[s]["T"][1]
		print('processing scene ',s, '  ntime=',n_time)
		arr, ldims = czi.read_image(S=s,T=0,C=0)
		sample = arr.squeeze()
		for t in range(n_time):
			arr_t, dim_t=czi.read_image(S=s,T=t,C=2)
			image = arr_t.squeeze()
			prev.process_czi_image(image, low_crop=0.85, high_crop=1.15, model_detect=model, n=-9999, show=True)
			if t==2:break
		if s==5:break

elif len(dims)==1:
	n_time=dims[0]["T"][1]
	for s in range(scene_count[0],scene_count[1]):
		print('processing scene ',s, '  ntime=',n_time)
		arr, ldims = czi.read_image(S=s,T=0,C=0)
		sample = arr.squeeze()
		for t in range(n_time):
			arr_t, dim_t=czi.read_image(S=s,T=t,C=2)
			image = arr_t.squeeze()
			prev.process_czi_image(image, low_crop=0.85, high_crop=1.15, model_detect=model, n=-9999, show=True)
			if t==2:break
		if s==5:break

