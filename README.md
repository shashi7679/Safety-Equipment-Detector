This project includes 3 Section. Files of all sections are in seprate directory.

we are trying to detect Hardhat and Head trained on a custom dataset.

Section 1 : In section 1, we are trying to get an inference on youtube video from a construction site.
Section 2 : Here we are trying to build a bounding box which resembles the color of hardhat detected.
Section 3 : In section 3, we are trying to test our custom model on our test data by drawing bounding box and generating annotations from the same.

All of my results can be replicated at your end as well.

### Steps to Run
- First clone yolov5 repo.
```
git clone git@github.com:ultralytics/yolov5.git

cd yolov5

pip install -r requirements.txt
```
- Get back to original directory 
```
pip install -r requirements.txt

```
## Section 1
- video_inference.py : Script for live inference

## Section 2
- video_inference_color.py : Script for live inference with adaptive color as the color of helmets.

## Section 3
- inference.py : Stores results of inference on test samples in output_frames dir.
- gen_xml_anno.py : Stores XML annotations for each of the test sample.