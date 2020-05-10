# FaceRecognition
A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source. There are multiple methods in which facial recognition systems work, but in general, they work by comparing selected facial features from given image with faces within a database. It is also described as a Biometric Artificial Intelligence based application that can uniquely identify a person by analyzing patterns based on the person's facial textures and shape
## Requirement
* opencv-python 
* face_recognition
* numpy
* pandas

```bash
pip install opencv-python face_recognition numpy pandas
```
## Usage
### generate dataset
* cropface.py crop person face and stor in ./data directory with folder name and persons identity to show while prediction
```python
python cropface.py
```

### prediction
* captureVideo.py captureVideo from camera and draw bounding box aroound with predction name
```python
python captureVideo.py
```
