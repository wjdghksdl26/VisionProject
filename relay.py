import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point
import time
#from avoidance_maneuver_srv.srv import CustomMsg, CustomMsgRequest

pub = rospy.Publisher('/test', String, queue_size=1)
#avoidance = rospy.ServiceProxy('/avoidance_maneuver', CustomMsg)
#avoidance_obj = CustomMsgRequest()

def callback(data):
    if data.z < 1:
        if data.x < 197.0:
            if data.y < 120.0:
                #direction = "object @ left upward"
                direction = "avoid right downward"
                #avoidance_obj = "up"
                #result = avoidance(avoidance_obj)
            elif data.y > 120.0:
                #direction = "object @ left downward"
                direction = "avoid right upward"
                #avoidance_obj = "up"
                #result = avoidance(avoidance_obj)
        elif data.x > 197.0:
            if data.y < 120.0:
                #direction = "object @ right upward"
                direction = "avoid left downward"
                #avoidance_obj = "up"
                #result = avoidance(avoidance_obj)
            elif data.y > 120.0:
                #direction = "object @ right downward"
                direction = "avoid left upward"
                #avoidance_obj = "up"
                #result = avoidance(avoidance_obj)
        pub.publish(direction)
    else:
        direction = "NO"
        pub.publish(direction)

def listener():
    rospy.init_node('relay')
    rospy.Subscriber('/moving_obj_coords', Point, callback)
    rospy.spin()

if __name__ == "__main__":
    listener()