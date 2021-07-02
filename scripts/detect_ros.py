#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import time
import argparse
import pathlib
import matplotlib

import tensorflow as tf

# ROS related imports
import rospy
from std_msgs.msg import String , Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

# Object detection module imports
import object_detection
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


##################################RECUP PARAM#####################################################################

node_name = str(rospy.get_param('node_name'))
model_name = str(rospy.get_param('model_name'))
label_name = str(rospy.get_param('label_name'))
topic_src = str(rospy.get_param('topic_src'))
topic_dest = str(rospy.get_param('topic_dest'))

###################################INIT TENSORFLOW + MODEL########################################################

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


model_path = os.path.join(os.path.dirname(sys.path[0]),'data','models', model_name,'saved_model')
label_path = os.path.join(os.path.dirname(sys.path[0]),'data','labels',label_name)

detection_model = tf.saved_model.load(model_path)
category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)


####################################DETECTOR CLASS##########################################################################

class Detector:

	#initialisation du node
	def __init__(self):
		self.image_pub = rospy.Publisher(topic_dest,Image, queue_size=1)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber(topic_src, Image, self.image_cb, queue_size=1, buff_size=2**24)
		print("node init")

	#fonction de callback quand on reçoit une image
	def image_cb(self, data):
		print("img reçu")
		t1 = time.time()

		#formatage de l'image du format image de ros vers un format exploitable par opencv
		try:
			cv_image_in = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
		image_in=cv2.cvtColor(cv_image_in,cv2.COLOR_BGR2RGB)
		image_modif = image_in
		print("img convert in")

		#analyse de l'image
		output_dict = run_inference_for_single_image(detection_model,image_in)
		
		#modification de l'image avec les resultats de l'analyse
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_modif,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			category_index,
			instance_masks=output_dict.get('detection_masks_reframed', None),
			use_normalized_coordinates=True,
			line_thickness=8)

		cv2.waitKey(0)

		#reformatage et envoi sur le topic de publication
		image_out=cv2.cvtColor(image_modif, cv2.COLOR_BGR2RGB)
		cv_image_out = Image()
		try:
			cv_image_out = self.bridge.cv2_to_imgmsg(image_out,"bgr8")
			cv_image_out.header = data.header
		except CvBridgeError as e:
			print(e)
		print("img convert out")
		self.image_pub.publish(cv_image_out)
		print("img send")
		t2 = time.time()

		print("execution time")
		print(t2-t1)
		print("#####")


####################FONCTIONS ANALYSE IMAGE################################

def run_inference_for_single_image(model, image):
	image = np.asarray(image)
	# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
	input_tensor = tf.convert_to_tensor(image)
	# The model expects a batch of images, so add an axis with `tf.newaxis`.
	input_tensor = input_tensor[tf.newaxis,...]
	
	# Run inference
	output_dict = model(input_tensor)

	# All outputs are batches tensors.
	# Convert to numpy arrays, and take index [0] to remove the batch dimension.
	# We're only interested in the first num_detections.
	num_detections = int(output_dict.pop('num_detections'))
	output_dict = {key: value[0, :num_detections].numpy()
				   for key, value in output_dict.items()}
	output_dict['num_detections'] = num_detections

	# detection_classes should be ints.
	output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
	# Handle models with masks:
	if 'detection_masks' in output_dict:
		# Reframe the the bbox mask to the image size.
		detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
									output_dict['detection_masks'], output_dict['detection_boxes'],
									image.shape[0], image.shape[1])      
		detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
		output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
	
	return output_dict

###########################MAIN############################################################################

def main(args):
	rospy.init_node(node_name)
	obj=Detector()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("ShutDown")
	cv2.destroyAllWindows()

if __name__=='__main__':
	main(sys.argv)
