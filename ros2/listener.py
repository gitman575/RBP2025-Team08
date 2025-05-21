import os
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

sys.path.append(os.getcwd())


class My_subscriber(Node):
    def __init__(self):
        super().__init__('chatter_subscriber')
        #TODO-------------------
        self.subscription = self.create_subscription(
            Twist,
            '/turtle1/cmd_vel',
            self.listener_callback,
            10)
        #-----------------------
        self.subscription  # prevent unused variable warning
        self.total_score = 0

    def listener_callback(self, msg):
        print(msg.linear.x)
            

def main(args=None):
    #TODO-------------------------
    rclpy.init(args=args)
    subscriber1 = My_subscriber()
    rclpy.spin(subscriber1)
    #-------------------------------
    subscriber1.destroy_node()
    rclpy.shutdown()
    print("finished")


if __name__ == '__main__':
    main()



