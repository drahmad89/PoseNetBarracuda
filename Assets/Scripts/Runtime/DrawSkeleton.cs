using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawSkeleton : MonoBehaviour
{
    [Tooltip("The list of key point GameObjects that make up the pose skeleton")]
    public GameObject[] keypoints;

    private GameObject[] lines;
    private LineRenderer[] lineRenderers;
    private int[][] jointPairs;
    private float lineWidth = 5.0f;

    void Start()
    {
        int numPairs = keypoints.Length + 1;
        lines = new GameObject[numPairs];
        lineRenderers = new LineRenderer[numPairs];
        jointPairs = new int[numPairs][];
        InitializeSkeleton();
    }

    void LateUpdate()
    {
        RenderSkeleton();
    }

    /// <summary>
    /// Create a line between the key point specified by the start and end point indices
    /// </summary>
    /// <param name="pairIndex"></param>
    /// <param name="startIndex"></param>
    /// <param name="endIndex"></param>
    /// <param name="width"></param>
    /// <param name="color"></param>
    private void InitializeLine(int pairIndex, int startIndex, int endIndex, float width, Color color)
    {
        jointPairs[pairIndex] = new int[] { startIndex, endIndex };

        string name = $"{keypoints[startIndex].name}_to_{keypoints[endIndex].name}";
        lines[pairIndex] = new GameObject(name);
        
        lineRenderers[pairIndex] = lines[pairIndex].AddComponent<LineRenderer>();
        lineRenderers[pairIndex].material = new Material(Shader.Find("Unlit/Color"));
        lineRenderers[pairIndex].material.color = color;
        
        lineRenderers[pairIndex].positionCount = 2;

        lineRenderers[pairIndex].startWidth = width;
        lineRenderers[pairIndex].endWidth = width;
    }

    /// <summary>
    /// Initialize the pose skeleton
    /// </summary>
    private void InitializeSkeleton()
    {
        // Nose to left eye
        InitializeLine(0, 0, 1, lineWidth, Color.red);
        // Nose to right eye
        InitializeLine(1, 0, 2, lineWidth, Color.red);
        // Left eye to left ear
        InitializeLine(2, 1, 3, lineWidth, Color.red);
        // Right eye to right ear
        InitializeLine(3, 2, 4, lineWidth, Color.red);

        // Left shoulder to right shoulder
        InitializeLine(4, 5, 6, lineWidth, Color.magenta);
        // Left shoulder to left hip
        InitializeLine(5, 5, 11, lineWidth, Color.magenta);
        // Right shoulder to right hip
        InitializeLine(6, 6, 12, lineWidth, Color.magenta);
        // Left shoulder to right hip
        InitializeLine(7, 5, 12, lineWidth, Color.magenta);
        // Right shoulder to left hip
        InitializeLine(8, 6, 11, lineWidth, Color.magenta);
        // Left hip to right hip
        InitializeLine(9, 11, 12, lineWidth, Color.magenta);

        // Left Arm
        InitializeLine(10, 5, 7, lineWidth, Color.blue);
        InitializeLine(11, 7, 9, lineWidth, Color.blue);
        // Right Arm
        InitializeLine(12, 6, 8, lineWidth, Color.blue);
        InitializeLine(13, 8, 10, lineWidth, Color.blue);

        // Left Leg
        InitializeLine(14, 11, 13, lineWidth, Color.green);
        InitializeLine(15, 13, 15, lineWidth, Color.green);
        // Right Leg
        InitializeLine(16, 12, 14, lineWidth, Color.green);
        InitializeLine(17, 14, 16, lineWidth, Color.green);
    }

    /// <summary>
    /// Draw the pose skeleton based on the latest location data
    /// </summary>
    private void RenderSkeleton()
    {
        // Iterate through the joint pairs
        for (int i = 0; i < jointPairs.Length; i++)
        {
            // Set the start point index
            int startpointIndex = jointPairs[i][0];
            // Set the end poin indext
            int endpointIndex = jointPairs[i][1];

            // Set the GameObject for the starting key point
            GameObject startingKeyPoint = keypoints[startpointIndex];
            // Set the GameObject for the ending key point
            GameObject endingKeyPoint = keypoints[endpointIndex];

            Vector3 startPos = new Vector3(startingKeyPoint.transform.position.x,
                                           startingKeyPoint.transform.position.y,
                                           startingKeyPoint.transform.position.z);
            // Get the ending position for the line
            Vector3 endPos = new Vector3(endingKeyPoint.transform.position.x,
                                         endingKeyPoint.transform.position.y,
                                         endingKeyPoint.transform.position.z);

            // Check if both the starting and ending key points are active
            if (startingKeyPoint.activeInHierarchy && endingKeyPoint.activeInHierarchy)
            {
                // Activate the line
                lineRenderers[i].gameObject.SetActive(true);
                // Update the starting position
                lineRenderers[i].SetPosition(0, startPos);
                // Update the ending position
                lineRenderers[i].SetPosition(1, endPos);
            }
            else
            {
                // Deactivate the line
                lineRenderers[i].gameObject.SetActive(false);
            }
        }
    }
}
