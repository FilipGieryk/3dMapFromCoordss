using UnityEngine;
using UnityEditor;
using System.IO;
using System.Text.RegularExpressions;

public class TerrainBatchImporter : EditorWindow
{
    string folderPath = "";

    [MenuItem("Tools/Batch Terrain Importer")]
    public static void ShowWindow()
    {
        GetWindow<TerrainBatchImporter>("Batch Terrain Importer");
    }

    void OnGUI()
    {
        GUILayout.Label("Batch Terrain Importer", EditorStyles.boldLabel);

        if (GUILayout.Button("Select Folder"))
        {
            folderPath = EditorUtility.OpenFolderPanel("Select Folder with Heightmaps", "", "");
        }
        GUILayout.Label("Folder: " + folderPath);

        if (GUILayout.Button("Import Terrains"))
        {
            if (!string.IsNullOrEmpty(folderPath))
                ImportTerrains();
            else
                EditorUtility.DisplayDialog("Error", "Please select a folder first.", "OK");
        }
    }

    void ImportTerrains()
    {
        string[] files = Directory.GetFiles(folderPath, "*.raw");
        if (files.Length == 0)
        {
            EditorUtility.DisplayDialog("No RAW files found", "No .raw files found in the selected folder.", "OK");
            return;
        }

        // Regex: tile_{xMin}_{yMin}_{width}_{length}_{minHeight}_{maxHeight}_rasterized{sizeX}x{sizeY}
        Regex tileRegex = new Regex(@"tile_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_rasterized(\d+)x(\d+)", RegexOptions.IgnoreCase);

        foreach (string file in files)
        {
            string filename = Path.GetFileNameWithoutExtension(file);
            Match match = tileRegex.Match(filename);
            if (!match.Success)
            {
                Debug.LogWarning("Filename does not match pattern: " + filename);
                continue;
            }

            float xMin = float.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
            float yMin = float.Parse(match.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
            float width = float.Parse(match.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
            float length = float.Parse(match.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
            float minHeight = float.Parse(match.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
            float maxHeight = float.Parse(match.Groups[6].Value, System.Globalization.CultureInfo.InvariantCulture);
            int sizeX = int.Parse(match.Groups[7].Value);
            int sizeY = int.Parse(match.Groups[8].Value);

            float heightRange = maxHeight - minHeight;

            // Create TerrainData
            TerrainData terrainData = new TerrainData();
            terrainData.heightmapResolution = sizeX;
            terrainData.size = new Vector3(width, heightRange, length);

            // Read RAW file
            float[,] heights = new float[sizeY, sizeX];
            using (FileStream fs = new FileStream(file, FileMode.Open, FileAccess.Read))
            {
                byte[] buffer = new byte[2];
                for (int y = 0; y < sizeY; y++)
                {
                    for (int x = 0; x < sizeX; x++)
                    {
                        fs.Read(buffer, 0, 2);
                        ushort value = (ushort)((buffer[1] << 8) | buffer[0]); // Little endian
                        heights[y, x] = value / 65535f;
                    }
                }
            }
            terrainData.SetHeights(0, 0, heights);

            // Place using real-world coordinates and minHeight for Y
            GameObject terrainObj = Terrain.CreateTerrainGameObject(terrainData);
            terrainObj.name = $"Terrain_{xMin}_{yMin}";
            terrainObj.transform.position = new Vector3(xMin, minHeight, yMin);
        }

        EditorUtility.DisplayDialog("Done", "Terrains imported and placed!", "OK");
    }
}