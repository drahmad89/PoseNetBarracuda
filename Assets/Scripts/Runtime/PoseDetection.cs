using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Video;
using System.Collections.Generic;


namespace Pose.Detection
{
    /// <summary>
    /// Pose Detection using Unity Barracuda.
    /// 1. Use an existing pose detection model (HRNet, DeepPose etc) and convert it to onnx model.
    /// 2. Render video to a RenderTarget and use it as the input for the model.
    /// 3. Preprocess the render target before passing it to the model. You can use Compute Shaders for this.
    /// 4. Each model requires a specific color space and resolution, you need to convert the render target first.
    /// 5. The `Model` object along with `IWorker` engine classes from Barracuda package will help you with inference.
    /// 6. Convert the output key points with confidence to actual pose and joint values.
    /// 7. Drive a skeleton/gameobjects with joint values.
    /// </summary>
    public class PoseDetection : MonoBehaviour
    {
        //[SerializeField] private VideoPlayer videoPlayer;
        [SerializeField] private ComputeShader preprocessingShader;

        [Space]
        [SerializeField] private int imageHeight = 480;
        [SerializeField] private int imageWidth = 480;
        [SerializeField] private string INPUT_NAME = "sub_2";

        [Space]
        [SerializeField] private GameObject videoQuad;
        [SerializeField] private RenderTexture videoTexture;
        [SerializeField] private RenderTexture inputTexture;
        [SerializeField] private NNModel modelAsset;
        [SerializeField] private GameObject[] keypoints;
        [SerializeField] private WorkerFactory.Type workerType = WorkerFactory.Type.Auto;
        [SerializeField] private int minConfidence = 30;


        private int _videoHeight;
        private int _videoWidth;

        //private WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

        private const int numKeypoints = 17;
        private IWorker engine;
        private Model m_RunTimeModel;
        private string heatmap_output = "float_heatmaps";
        private string offsets_output = "float_short_offsets";
        private string prediction_prob = "heatmap_predictions";
        // Estimated 2D keypoint locations in videoTexture and their associated confidence values
        private float[][] keypointLocations = new float[numKeypoints][];

        //Shader Property
        private static readonly int MainTex = Shader.PropertyToID("_MainTex");

        void Start()
        {

            //TODO: Create Model from onnx asset and compile it to an object
            heatmap_output = "float_heatmaps";
            offsets_output ="float_short_offsets";
            prediction_prob = "heatmap_predictions";
            m_RunTimeModel = ModelLoader.Load(modelAsset);
            var modelBuilder = new ModelBuilder(m_RunTimeModel);

            //TODO: Add Layers to model
            modelBuilder.Sigmoid(prediction_prob, heatmap_output);
            //TODO: Create Worker Engine
            engine = WorkerFactory.CreateWorker(workerType, modelBuilder.model);


            Transform videoQuad = GameObject.Find("videoQuad").transform;
            GameObject videoPlayer = GameObject.Find("Video Player");
            _videoHeight = (int)videoPlayer.GetComponent<VideoPlayer>().height;
            _videoWidth = (int)videoPlayer.GetComponent<VideoPlayer>().width;

 
            // Create a new videoTexture using the current video dimensions
            videoTexture = new RenderTexture(_videoWidth, _videoHeight, 24, RenderTextureFormat.ARGB32);
            videoPlayer.GetComponent<VideoPlayer>().targetTexture = videoTexture;

            //Release the current videoTexture
            videoTexture.Release();

            //Apply Texture to Quad
            //videoQuad = GameObject.Find("videoQuad");
            videoQuad.gameObject.GetComponent<MeshRenderer>().material.SetTexture(MainTex, videoTexture);
            videoQuad.transform.localScale = new Vector3(_videoWidth, _videoHeight, videoQuad.transform.localScale.z);
            //videoQuad.transform.position = new Vector3(0, 0, 1);
            videoQuad.transform.position = new Vector3(_videoWidth/2, _videoHeight/2, 1);

            //Move Camera to keep Quad in view
            GameObject mainCamera = GameObject.Find("Main Camera");
            if (mainCamera != null)
            {
                mainCamera.transform.position = new Vector3(_videoWidth / 2, _videoHeight / 2, -(_videoWidth / 2));
                //mainCamera.transform.position = new Vector3(0, 0, -(_videoWidth / 2));
                mainCamera.GetComponent<Camera>().orthographicSize = _videoHeight / 2;
            }


        }


        // Unity method that is called every tick/frame
        private void Update()
        {
            videoQuad.SetActive(false);
            Texture2D processedImage = PreprocessTexture();

            //TODO: Create Tensor 
            Tensor input = new Tensor(processedImage, channels: 3);
            //TODO: Execute Engine
            var inputs = new Dictionary<string, Tensor> {
                        { INPUT_NAME, input }
                   };
            engine.Execute(inputs);
            //TODO: Process Results
            ProcessResults(engine.PeekOutput(prediction_prob), engine.PeekOutput(offsets_output));

            //TODO: Draw Skeleton
            DrawSkeleton();
            //TODO: Clean up tensors and other resources
            Destroy(processedImage);
        }

        private void OnDisable()
        {
            //TODO: Release the inference engine
            engine.Dispose();
            //Release videoTexture
            videoTexture.Release();
        }


        #region Additional Methods

        private void ProcessResults(Tensor heatmaps, Tensor offsets)
        {
            // Determine the estimated key point locations using the heatmaps and offsets tensors

            // Calculate the stride used to scale down the inputImage
            float stride = (imageHeight - 1) / (heatmaps.shape.height - 1);
            stride -= (stride % 8);

            int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);
            int maxDimension = Mathf.Max(videoTexture.width, videoTexture.height);

            //Recalculate scale for the keypoints
            var scale = (float)minDimension / (float)Mathf.Min(imageWidth, imageHeight);
            var adjustedScale = (float)maxDimension / (float)minDimension;

            // Iterate through heatmaps
            for (int k = 0; k < numKeypoints; k++)
            {
                //Find Location of keypoint
                var locationInfo = LocateKeyPoint(heatmaps, offsets, k);

                // The (x, y) coordinates contains the confidence value in the current heatmap
                var coords = locationInfo.Item1;
                var offsetVector = locationInfo.Item2;
                var confidenceValue = locationInfo.Item3;

                //Calulate X position and Y position
                var xPos = (coords[0] * stride + offsetVector[0]) * scale;
                var yPos = (imageHeight - (coords[1] * stride + offsetVector[1])) * scale;
                if (videoTexture.width > videoTexture.height)
                {
                    xPos *= adjustedScale;
                }
                else
                {
                    yPos *= adjustedScale;
                }

                keypointLocations[k] = new float[] { xPos, yPos, confidenceValue };
            }
        }

        private (float[], float[], float) LocateKeyPoint(Tensor heatmaps, Tensor offsets, int i)
        {
            //Find the heatmap index that contains the highest confidence value and the associated offset vector
            var maxConfidence = 0f;
            var coords = new float[2];
            var offsetVector = new float[2];

            // Iterate through heatmap columns
            for (int y = 0; y < heatmaps.shape.height; y++)
            {
                // Iterate through column rows
                for (int x = 0; x < heatmaps.shape.width; x++)
                {
                    if (heatmaps[0, y, x, i] > maxConfidence)
                    {
                        maxConfidence = heatmaps[0, y, x, i];
                        coords = new float[] { x, y };
                        offsetVector = new float[]
                        {
                            offsets[0, y, x, i + numKeypoints],
                            offsets[0, y, x, i]
                        };
                    }
                }
            }
            return (coords, offsetVector, maxConfidence);
        }

        private Texture2D PreprocessTexture()
        {
            //Apply any kind of preprocessing if required - Resize, Color values scaled etc

            Texture2D imageTexture = new Texture2D(videoTexture.width,
                videoTexture.height, TextureFormat.RGBA32, false);

            Graphics.CopyTexture(videoTexture, imageTexture);
            Texture2D tempTex = Resize(imageTexture, imageHeight, imageWidth);
            Destroy(imageTexture);

            // TODO: Apply model-specific preprocessing
            imageTexture = PreprocessNetwork(tempTex);

            Destroy(tempTex);
            return imageTexture;
        }

        private Texture2D PreprocessNetwork(Texture2D inputImage)
        {
            // Use Compute Shaders (GPU) to preprocess your image
            // Each model requires a specific color space - RGB 
            // Values need to scaled to what it was trained on

            var numthreads = 8;
            var kernelHandle = preprocessingShader.FindKernel("Preprocess");
            var rTex = new RenderTexture(inputImage.width,
                inputImage.height, 24, RenderTextureFormat.ARGBHalf);
            rTex.enableRandomWrite = true;
            rTex.Create();

            preprocessingShader.SetTexture(kernelHandle, "Result", rTex);
            preprocessingShader.SetTexture(kernelHandle, "InputImage", inputImage);
            preprocessingShader.Dispatch(kernelHandle, inputImage.height
                                                       / numthreads,
                inputImage.width / numthreads, 1);

            RenderTexture.active = rTex;
            Texture2D nTex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBAHalf, false);
            Graphics.CopyTexture(rTex, nTex);
            RenderTexture.active = null;

            Destroy(rTex);
            return nTex;
        }

        private Texture2D Resize(Texture2D image, int newWidth, int newHeight)
        {
            RenderTexture rTex = RenderTexture.GetTemporary(newWidth, newHeight, 24);
            RenderTexture.active = rTex;

            Graphics.Blit(image, rTex);
            Texture2D nTex = new Texture2D(newWidth, newHeight, TextureFormat.RGBA32, false);

            Graphics.CopyTexture(rTex, nTex);
            RenderTexture.active = null;

            RenderTexture.ReleaseTemporary(rTex);
            return nTex;
        }


        private void DrawSkeleton()
        {
            for (int k = 0; k < numKeypoints; k++)
            {
                if (keypointLocations[k][2] >= minConfidence / 100f)
                {
                    keypoints[k].SetActive(true);
                }
                else
                {
                    keypoints[k].SetActive(false);
                }

                Vector3 newPos = new Vector3(keypointLocations[k][0], keypointLocations[k][1], -1f);

                keypoints[k].transform.position = newPos;
            }
        }
        #endregion
    }

}