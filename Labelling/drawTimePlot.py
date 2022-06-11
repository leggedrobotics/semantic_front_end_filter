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

    def concat(self, data):
        """
        Contact with another databuffer (this will ruin data_id)
        """
        self.data.update({k: self.data.get(k,[])+v for k,v in data.data.items()} )
        self.indices.update({k: self.indices.get(k,[])+v for k,v in data.indices.items()} )

fig = plt.figure(figsize=(21, 3))

wholeData = DataBuffer()
time0 = 0
for i in range(3):
    datapath = "/media/chenyu/T7/Data/extract_trajectories/for_vis/traj_%d.pkl"%i
    with open(datapath,"rb") as f:
        data = pkl.load(f)
    # print({k: (np.array(v).shape,v[0]) for k,v in data.indices.items()})
    data.indices = {k:list(time0+np.array(v)-v[0] )for k,v in data.indices.items()}
    wholeData.concat(data)
    time0 = list(data.indices.values())[0][-1]

keys = ["LF_shank_fixed_LF_FOOT", "prediction_LF_shank_fixed_LF_FOOT", "pointcloud_LF_shank_fixed_LF_FOOT"]
labels = ["Actual foothold", "Prediction", "Point cloud"]
for k,n in zip(keys,labels):
    timeArr = wholeData.indices[k]
    lw = 2 if k != "LF_shank_fixed_LF_FOOT" else 4
    plt.plot(timeArr, np.array(wholeData.data[k])[:,2], label = n, lw = lw)
# print(np.array(wholeData.data["LF_shank_fixed_LF_FOOT"]))
# print(np.array(dwholeDataata.data["pose"]))
plt.xlabel("Time (s)",fontsize = 16)
plt.ylabel("Height (m)",fontsize = 16)
plt.xticks(fontsize= 10)
plt.yticks(fontsize= 10)
plt.ylim((-7,1))
# plt.title("")
plt.legend(fontsize = 16)
plt.savefig("timePlot.jpg",bbox_inches='tight', pad_inches=0.1)