"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import time

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def draw_boxes(frame, result, width, height, p):
    count = 0
    p1 = None
    p2 = None
    for box in result[0][0]:
        conf = box[2]
        if conf >= p:
            p1 = (int(box[3] * width), int(box[4] * height))
            p2 = (int(box[5] * width), int(box[6] * height))
            cv2.rectangle(frame, p1, p2, (255, 0, 0))
            count += 1
    return frame, count, (p1, p2)


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    ### TODO: Load the model through `infer_network` ###
    num_requests = 2
    infer_network.load_model(args.model, num_requests, args.device, args.cpu_extension)
    network_shape = infer_network.get_input_shape()

    stream_flag = False
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        stream_flag = True
    else:
        assert(os.path.isfile(args.input), "file doesn't exist")

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    # additional parameters
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    current_req_id = 0
    next_req_id = 1

    last_count = 0
    people_count = 0
    total_count = 0
    
    no_box = 0
    prev_box_x = 0
    start_time = 0
    duration = 0
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        
        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (network_shape[3], network_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(image, next_req_id)
        
        ### TODO: Wait for the result ###
        if infer_network.wait(current_req_id) == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(current_req_id)

            ### TODO: Extract any desired stats from the results ###
            frame, people_count, box = draw_boxes(frame, result, width, height, prob_threshold)
            box_w = frame.shape[1]
            top_left, bottom_right = box
                
            if people_count > last_count:
                start_time = time.time()
                total_count += people_count - last_count
                no_box = 0
                client.publish("person", json.dumps({"total":total_count}))
            elif people_count < last_count:
                if no_box <= 20:
                    people_count = last_count
                    no_box += 1
                elif prev_box_x < box_w - 200:
                    people_count = last_count
                    no_box = 0
                else:
                    duration = int(time.time() - start_time)
                    client.publish("person/duration", json.dumps({"duration":duration}))
                    
            if top_left != None and bottom_right != None:
                prev_box_x = int((top_left[0] + bottom_right[0]) / 2)
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###            
            last_count = people_count      
            client.publish("person", json.dumps({"count":people_count}))

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        if stream_flag:
            cv2.imwrite('output_image.jpg', frame)
        
        current_req_id, next_req_id = next_req_id, current_req_id
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    # Disconnect from MQTT
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
