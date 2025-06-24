from aicspylibczi import CziFile
import numpy as np
import sys, os
import tifffile
import matplotlib.pyplot as plt
sys.path.append(os.path.join("utils"))
import utils_preview as prev
model=prev.load_model(r"C:\Users\helsens\software\singleCell_catalog\cell_detection_model.pth")

path=r"H:\PROJECTS-03\Feyza\forClement\New-01.czi"

prev.process_czi(path, low_crop=0.85, high_crop=1.15, model_detect=model, n=-9999)
prev.run_server_czi()



czi = CziFile(path)
dims = czi.get_dims_shape()
n_cells = len(dims)
print('N cells ',n_cells)


def rescale(image):
	imin, imax = image.min(), image.max()
	print(imin, '  ',imax)
	if imin==imax:
		return np.zeros_like(image, dtype=np.uint8)
	scaled = 255 * (image.astype(np.float32)-imin)/(imin-imax)
	print (scaled.min(), '  ', scaled.max())
	return scaled.astype(np.uint8)




for s in range(len(dims)):
	n_time=dims[s]["T"][1]
	print('processing scene ',s, '  ntime=',n_time)
	arr, ldims = czi.read_image(S=s,T=0,C=0)
	sample = arr.squeeze()
	for t in range(n_time):
		arr_t, dim_t=czi.read_image(S=10,T=t,C=2)
		image = arr_t.squeeze()
		prev.process_czi_image(image, low_crop=0.85, high_crop=1.15, model_detect=model, n=-9999, show=True)
		if t==2:break

	if s==5:break

