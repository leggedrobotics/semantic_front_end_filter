#include <ros/ros.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <cmath>
#include <msgpack.hpp>
#include <fstream>
#include <cmath>

using namespace grid_map;
struct GridMapStruc
{
    double res;
    int xNormal;
    int yNormal;
    std::vector<double> xRealRange;
    std::vector<double> yRealRange;
    double meanHeight;
    std::vector<std::vector<double>> GroundArray;
    std::vector<std::vector<double>> GPMap;
    std::vector<std::vector<double>> Confidence;
    MSGPACK_DEFINE_MAP(res, xNormal, yNormal, xRealRange, yRealRange, meanHeight, GroundArray, GPMap, Confidence)
};
// {"res":0.1,"xNormal":-274,"yNormal":-334,"xRealRange":[-27.3202,28.2702],"yRealRange":[-33.3456,16.306],"meanHeight":-1.2049}

int main(int argc, char** argv)
{ 
  // Initialize node and publisher.
  ros::init(argc, argv, "grid_map_simple_demo");
  ros::NodeHandle nh("~");
  ros::Publisher publisher = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
  
  // Get file path
  // ros::NodeHandle nh;
  std::string map_file_path;
  nh.getParam("map_file_path", map_file_path);

  // Get data from file
  std::ifstream gridmapFile;
  std::streampos size;

  gridmapFile.open(map_file_path, std::ios::in|std::ios::binary);
  // if(gridmapFile.is_open()){}

  std::string buffer((std::istreambuf_iterator<char>(gridmapFile)),
                  std::istreambuf_iterator<char>());
  msgpack::object_handle upd;
  std::size_t offset = 0;
  msgpack::unpack(upd, buffer.data(), buffer.size(), offset);
  msgpack::object ob = upd.get();
  // std::cout << ob<< std::endl;
  
  GridMapStruc gm;
  ob.convert(gm);

  gridmapFile.close();


  // Create grid map.
  GridMap map({"elevation"});
  map.setFrameId("map");
  Position gmPosition((gm.xRealRange[1] + gm.xRealRange[0])/2. ,(gm.yRealRange[1] + gm.yRealRange[0])/2. );
  
  // Position gmPosition(0 ,100);

  std::cout<<gmPosition<<std::endl;
  // map.setPosition(gmPosition);
  // map.move(gmPosition);

  // map.setGeometry(Length(1.2, 2.0), 0.03);
  map.setGeometry(Length(gm.xRealRange[1] - gm.xRealRange[0], gm.yRealRange[1] - gm.yRealRange[0]), gm.res);
  ROS_INFO("Created map with size %f x %f m (%i x %i cells).",
    map.getLength().x(), map.getLength().y(),
    map.getSize()(0), map.getSize()(1));
  map.setPosition(gmPosition);
  // Work with grid map in a loop.
  ros::Rate rate(0.1);
  while (nh.ok()) {

    // Add data to grid map.
    ros::Time time = ros::Time::now();
    for (GridMapIterator it(map); !it.isPastEnd(); ++it) {
      Position position;
      map.getPosition(*it, position);
      // map.at("elevation", *it) = -0.04 + 0.2 * std::sin(3.0 * time.toSec() + 5.0 * position.y()) * position.x();
      double grid_x = floor(position.x()/gm.res)-gm.xNormal;
      double grid_y = floor(position.y()/gm.res)-gm.yNormal;
      if(grid_x >=0 && grid_x < gm.GroundArray.size() && grid_y >=0 && grid_y < gm.GroundArray[0].size()){
        if (abs(gm.GPMap[grid_x][grid_y])>0.01){
          map.at("elevation", *it) = gm.GPMap[grid_x][grid_y];
        }
        else
          // map.at("elevation", *it) = gm.GPMap[grid_x][grid_y];
          map.at("elevation", *it) = gm.meanHeight;
        // if (abs(map.at("elevation", *it))<=0.001){
        //   map.at("elevation", *it) = gm.meanHeight;
        // }
      }
    }

    // Publish grid map.
    map.setTimestamp(time.toNSec());
    grid_map_msgs::GridMap message;
    GridMapRosConverter::toMessage(map, message);
    publisher.publish(message);
    ROS_INFO_THROTTLE(1.0, "Grid map (timestamp %f) published.", message.info.header.stamp.toSec());

    // Wait for next cycle.
    rate.sleep();
  }

  return 0;
}
