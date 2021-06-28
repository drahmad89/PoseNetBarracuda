# PoseNet Barracuda
## A pose estimation model integration in Unity 



## Features

- Draw skeleton for 17 human body keypoints 
- The model works on singe and multiple body pose estimation but the postprocessing part only covers single body scenario




## Installation

Thie project Unity installed to run.
The packages used in the projects:
- Barracuda
- MAthematics
- Burst
- TextMeshPro
- Test Framework
- Visual Studio code edtiro

## Run

- Download MP4 video [clip](https://www.pexels.com/video/emotions-dancing-amusement-park-portrait-4841885/) and move the clip under videos folder in the project
- Download PoseNet [model](https://github.com/tensorflow/tfjs-models/tree/master/posenet) in TensorflowJS format 
- Use [tfjs](https://pypi.org/project/tfjs-graph-converter/) to savedmodel converter to create a tensorflow savedmodel
- Use [tf2onnx](https://github.com/onnx/tensorflow-onnx) to create onnx version of the model
- Move the onnx file under Assets/Models
- Open the project on Unity by simply double clicking on lastscene in Scenes
- Drag the videoclip inside the Video Player object and also drag the onnx model into the PoseEstimator object
- Run the project and enjoy the Dance!

## License

MIT

**Free Software**


