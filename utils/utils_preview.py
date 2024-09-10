import os
import numpy  as np
import nd2
from pathlib import Path

file='/Volumes/upoates/raw_data/microscopy/cell_culture/wscepfl0117/wscepfl0117.nd2'
current_file=os.path.join(file)
time_lapse_path = Path(current_file)
print('time_lapse_path = ',time_lapse_path)
time_lapse = nd2.imread(time_lapse_path.as_posix())



print(time_lapse.shape())