import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
class DataBuffer:
    def __init__(self):
        self.indices = {}
        self.data = {}
        self.data_id = {}

    def append(self, stamp, data, name):

        if name in self.data:
            self.data[name].append(data)
            self.indices[name].append(stamp)
            self.data_id[name].append(self.data_id[name][-1] + 1)
        else:
            self.data[name] = [data]
            self.indices[name] = [stamp]
            self.data_id[name] = [0]

datapath = "/media/chenyu/T7/Data/extract_trajectories/for_vis/traj_0.pkl"
with open(datapath,"rb") as f:
    data = pkl.load(f)

keys = ["LF_shank_fixed_LF_FOOT", "prediction_LF_shank_fixed_LF_FOOT", "pointcloud_LF_shank_fixed_LF_FOOT"]
for k in keys:
    timeArr = data.indices[k]
    plt.plot(timeArr, np.array(data.data[k])[:,2], label = k)
print(np.array(data.data["LF_shank_fixed_LF_FOOT"]))
print(np.array(data.data["pose"]))
plt.legend()
plt.savefig("timePlot.jpg")