import rospy # for `Duration`
import rosbag
import tf2_py
import genpy

class RosbagPlayer:
    """
    This 'simulator' holds a rosbag, 
        execute registered callback functions when play
        also holds tf_buffer to facilitate an easy look up
    Note: this is only for processing bag data, not for adding new topics and messages
    """
    def __init__(self, rosbagpath):

        self.bag = rosbag.Bag(rosbagpath)
        duration = self.bag.get_end_time() - self.bag.get_start_time()
        self.tf_buffer = tf2_py.BufferCore(rospy.Duration(duration))
        
        # read the whole tf history
        for topic, msg, t in self.bag.read_messages(topics=['/tf_static']):
            for transform in msg.transforms:
                self.tf_buffer.set_transform_static(transform, 'rosbag')

        tf_times = []
        for topic, msg, t in self.bag.read_messages(topics=['/tf']):
            for transform in msg.transforms:
                self.tf_buffer.set_transform(transform, 'rosbag')
                tf_times.append(transform.header.stamp)

        self._callbacks = {}
        self._shared_var = {}
        self._ros_publishers = {}

    def register_callback(self, topic, func):
        """
        arg topic: the topic of the callback function
        arg func: have the signature: (topic, msg, t, tf_buffer, shared_var) 
        """
        self._callbacks[topic] = func

    def play(self, start_time=None, end_time=None, rate=None):
        """
        Play the rosbag and call the callbacks
        """
        start_time = start_time if start_time is None else rospy.Time(start_time)
        end_time = end_time if end_time is None else rospy.Time(end_time)
        rate = rospy.Rate(rate) if rate is not None else None # 1000hz
        for topic, msg, t in self.bag.read_messages(
            topics=list(self._callbacks.keys()),
            start_time=start_time, end_time=end_time):
            self._callbacks[topic](topic, msg, t, self.tf_buffer, self._shared_var)
            if rate is not None: rate.sleep() 

    def add_publisher_of_topic(self, topic, queue_size=1):
        """
        Generate a publisher callback and add it into self._callbacks
        The publisher just publish the message into ros environment
        """
        typeinfo, topicinfo = self.bag.get_type_and_topic_info()
        msgtype_name = topicinfo[topic][0]
        msgtype_t = genpy.message.get_message_class(msgtype_name)
        pub = rospy.Publisher(topic, msgtype_t , queue_size=queue_size)
        self._ros_publishers[topic] = pub
        def publish_cb(topic, msg, t, tf_buffer, shared_var):
            pub.publish(msg)
        self._callbacks[topic] = publish_cb

