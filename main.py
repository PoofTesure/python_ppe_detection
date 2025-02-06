# Import the InferencePipeline object
from inference import InferencePipeline

from inference.core.interfaces.camera.entities import VideoFrame

# import opencv to display our annotated images
import cv2
# import supervision to help visualize our predictions
import supervision as sv

import time

label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoundingBoxAnnotator()

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # get the text labels for each prediction
    labels = [p["class"] for p in predictions["predictions"]]
    print(labels)
    # load our predictions into the Supervision Detections api

    hat_count = labels.count("Hardhat")
    person_count = labels.count("Person")
    glove_count = labels.count("Gloves")/2
    vest_count = labels.count("Safety Vest")

    sum_count = hat_count + vest_count
    print (hat_count)
    print (person_count)
    print (glove_count)
    print (vest_count)

    violation = False

    detections = sv.Detections.from_inference(predictions)
    # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
    image = label_annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    image = box_annotator.annotate(image, detections=detections)
    # display the annotated image
    local_time = time.ctime(time.time())
    print("Local time:", local_time)

    height, width, channels = image.shape

    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org1 = (0, height-2)

    org = (0,50)

    # fontScale
    fontScale = 0.5
    
    # Blue color in BGR
    color = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.putText() method
    image = cv2.putText(image, local_time, org1, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    
    if(hat_count < person_count):
        image = cv2.putText(image, "No Hat", org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
        violation = True
    
    #Deteksi sarung tangan tidak akurat jadi saya matikan
    #if(glove_count/2 < person_count):
    #    image = cv2.putText(image, "No Glove", org, font, 
    #            fontScale, color, thickness, cv2.LINE_AA)
    #   violation = True

    if(vest_count < person_count):
        image = cv2.putText(image, "No Vest", org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
        violation = True

    fileName = "./violation_image/" + str(time.time()) + ".png"
    print(fileName)
    
    if(violation):
        print("Violation")
        print(cv2.imwrite(fileName, image))

    cv2.imshow("Predictions", image)
    cv2.waitKey(1)

# initialize a pipeline object
# Video reference=0 bisa disesuaikan dengan webcam
pipeline = InferencePipeline.init(
    model_id="construction-site-safety/27", # Roboflow model to use
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction=my_custom_sink, # Function to run after each prediction
    api_key="htIBAVUHNFJE7C8rKomO"
)
pipeline.start()
pipeline.join()