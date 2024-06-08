# Construct Map from Robot Feet Trajectories

## Map Generation

The file ground_from_trajs.py provides a class **GFT**, which allows you to generate a grid map from FeetTrajs.msgpack or load a grid map from GroundMap.msgpack. The examples files can be downloaded [here](https://drive.google.com/drive/folders/1m1XzdB_q6GBZjpP_csMFxQ3IIILvXtjO). Then you can put them under the folder `semantic_front_end_filter/utils/labelling/examples`

You can instantiate class **GFT** with one and only one of these two files. Then you can use multi API to access or visualize the grid map. For example, 

```python
## Get a Grid Map by Gaussian Process. 
# Since the Gaussian Process will take about 30s, if you only want to use a sparse grid map, you can also set InitializeGP = False and fit with Gaussian Process later by GPT::initializeGPMap().
gft = GFT(FeetTrajsFile='./examples/FeetTrajs.msgpack', InitializeGP = True)
gft.save('./examples/', GPMap=True)

# Load Grid Map File
gftload = GFT(GroundMapFile='./examples/GroundMap.msgpack')

# Get Height
print(gftload.getHeight(0, 0))

xlist = np.zeros(3)
ylist = np.zeros(3)
print(gftload.getHeight(xlist, ylist))

# Get the whole map and confidence
GPMap = gftload.getGPMap()

# Visualize
gftload.visualizeGPMap()

```

## Definations

**Sparse Grid Map**: A grid map constructed only by the filling the grid map with foot contacts.

**Gaussian Process Map**: A grid map constructed by fitting Gaussian Process on Sparse Grid Map.