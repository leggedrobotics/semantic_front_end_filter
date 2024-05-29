# Construct Map from Robot Feet Trajectories

## Map Generation

GetGroundfromTrajs.py provides a class **GFT**, which allows you to generate a grid map from FeetTrajs.msgpack or load a grid map from GroundMap.msgpack. The examples files can be downloaded [here](https://drive.google.com/drive/folders/1Lx5QfLrfS0vk_88-UAJolm3D_ovZh5wS)

You can instantiate class **GFT** with one and only one of these two files. Then you can use multi API to access or visualize the grid map. For example, 

```python
## Get a Grid Map by Gaussian Process. 
# Since the Gaussian Process will consume some time(about 30s), if you only want to use a sparse grid map, you can also set InitializeGP = False and fit with Gaussian Process later by GPT::initializeGPMap().
gft = GFT(FeetTrajsFile='./Examples/FeetTrajs.msgpack', InitializeGP = True)
gft.save('./Examples/', GPMap=True)

# Load Grid Map File
gftload = GFT(GroundMapFile='./Examples/GroundMap.msgpack')

# Get Height
print(gftload.getHeight(0, 0))

xlist = np.zeros(3)
ylist = np.zeros(3)
print(gftload.getHeight(xlist, ylist))

# Get the whole map and confidence
GPMap = gftload.getGPMap()

# Visualize
gftload.visualizeGPMap()

## To avoid the drift from long distance travel, you can choose only build local maps based on local footholds, which is also used in this project.

saveLocalMaps(feet_traj='./Examples/FeetTrajs.msgpack', save_path = './Examples/')
```

## Definations

**Sparse Grid Map**: A grid map constructed only by the filling the grid map with foot contacts.

**Gaussian Process Map**: A grid map constructed by fitting Gaussian Process on Sparse Grid Map.