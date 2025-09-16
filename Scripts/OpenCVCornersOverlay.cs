using System;
using System.Collections.Generic;
using UnityEngine;

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;

// Resolve Rect type conflict explicitly.
using URect = UnityEngine.Rect;

[DisallowMultipleComponent]
public class OpenCVCornersXYZOverlay : MonoBehaviour
{
    [Header("Camera & Raycast")]
    public Camera cam;
    public LayerMask raycastMask = ~0;
    public float maxRaycastDistance = 1000f;

    // Expose read-only access to the hit points
    public IReadOnlyList<Vector3> LastWorldHits => _lastWorldHits;

    [Header("Capture")]
    [Tooltip("Downscale for speed; affects both capture and corner detection resolution.")]
    [Range(0.25f, 1f)] public float scale = 0.5f;
    [Tooltip("Process every Nth frame to save CPU.")]
    public int processEveryNFrames = 2;

    [Header("goodFeaturesToTrack (Shi–Tomasi)")]
    public int maxCorners = 300;
    [Range(0.001f, 0.2f)] public float qualityLevel = 0.01f;
    [Range(1f, 50f)] public float minDistance = 8f;
    [Range(3, 15)] public int blockSize = 3;
    public bool useHarris = false;
    [Range(0.01f, 0.2f)] public float harrisK = 0.04f;

    [Header("Overlay & Logging")]
    [Tooltip("Draws small white squares at detected corner locations on the Game view (works in builds).")]
    public bool drawOnGameView = true;
    [Range(1, 8)] public int overlayDotSize = 3;

    [Tooltip("Draw yellow gizmo spheres at hit XYZ (Editor only).")]
    public bool drawWorldGizmos = true;
    public float worldGizmoSize = 0.1f;
    public Color worldGizmoColor = Color.yellow;

    [Tooltip("Max number of XYZ lines to log per processed frame to avoid console spam.")]
    public int logMaxPerFrame = 20;
    public bool logEveryFrame = true;

    [Header("ROI & Filtering")]
    [Tooltip("If true, detect corners only inside roi01 (normalized [0..1], origin top-left).")]
    public bool useROI = false;
    [Tooltip("Normalized ROI [0..1] in capture image pixels (x,y,w,h), origin top-left.")]
    public URect roi01 = new URect(0, 0, 1, 1);
    [Tooltip("If set, only accept raycast hits whose collider is under this Transform.")]
    public Transform onlyUnderRoot = null;

    // Internals
    RenderTexture _rt;
    Texture2D _tex;
    int _frameCount;

    Mat _rgba; // CV_8UC4
    Mat _gray; // CV_8UC1
    Mat _mask; // ROI mask; empty Mat means no mask

    readonly List<Vector2> _cornersUV = new();   // (u,v) in capture image pixels
    readonly List<Vector3> _lastWorldHits = new();

    void Start()
    {
        if (!cam)
        {
            Debug.LogError("OpenCVCornersXYZOverlay: Assign a Camera.");
            enabled = false;
            return;
        }
        SetupRT(); // creates _rt/_tex and pre-sizes mats
    }

    // Public setters for coordinator scripts
    public void SetROI01(URect r01) { roi01 = r01; useROI = true; }
    public void ClearROI() { useROI = false; }
    public void SetOnlyUnderRoot(Transform root) { onlyUnderRoot = root; }

    // Optional one-shot processing helper
    public void ProcessOnceImmediate()
    {
        int keep = processEveryNFrames;
        int fc = _frameCount;
        processEveryNFrames = 1;
        _frameCount = 0;
        LateUpdate();
        processEveryNFrames = keep;
        _frameCount = fc;
    }

    // Convert capture pixel to screen point
    Vector2 UVToScreenPoint(Vector2 uv, int W, int H)
    {
        float vx = uv.x / W;
        float vy = 1f - (uv.y / H); // flip to Unity screen Y-up

        var r = cam.rect; // normalized [0..1]
        float sx = (r.xMin + vx * r.width)  * Screen.width;
        float sy = (r.yMin + vy * r.height) * Screen.height;
        return new Vector2(sx, sy);
    }

    void SetupRT()
    {
        int w = Mathf.Max(64, Mathf.RoundToInt(cam.pixelWidth * scale));
        int h = Mathf.Max(64, Mathf.RoundToInt(cam.pixelHeight * scale));

        if (_rt) Destroy(_rt);
        if (_tex) Destroy(_tex);

        _rt = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32) { antiAliasing = 1 };
        _tex = new Texture2D(w, h, TextureFormat.RGBA32, false);

        EnsureMatsMatchTexture(w, h);
    }

    void EnsureMatsMatchTexture(int W, int H)
    {
        if (_rgba == null || _rgba.cols() != W || _rgba.rows() != H || _rgba.type() != CvType.CV_8UC4)
        {
            _rgba?.Dispose();
            _rgba = new Mat(H, W, CvType.CV_8UC4);
        }
        if (_gray == null || _gray.cols() != W || _gray.rows() != H || _gray.type() != CvType.CV_8UC1)
        {
            _gray?.Dispose();
            _gray = new Mat(H, W, CvType.CV_8UC1);
        }
        if (_mask == null) _mask = new Mat(); // keep empty unless ROI is enabled
    }

    void BuildMaskIfNeeded(int W, int H)
    {
        if (!useROI)
        {
            if (_mask != null && !_mask.empty())
            {
                _mask.release();
                _mask = new Mat(); // empty -> no mask
            }
            return;
        }

        // Allocate/resize to CV_8UC1
        if (_mask == null || _mask.empty() || _mask.cols() != W || _mask.rows() != H || _mask.type() != CvType.CV_8UC1)
            _mask = new Mat(H, W, CvType.CV_8UC1);

        _mask.setTo(new Scalar(0));

        int x = Mathf.Clamp(Mathf.RoundToInt(roi01.x      * W), 0, W - 1);
        int y = Mathf.Clamp(Mathf.RoundToInt(roi01.y      * H), 0, H - 1);
        int w = Mathf.Clamp(Mathf.RoundToInt(roi01.width  * W), 1, W - x);
        int h = Mathf.Clamp(Mathf.RoundToInt(roi01.height * H), 1, H - y);

        Imgproc.rectangle(_mask, new Point(x, y), new Point(x + w - 1, y + h - 1), new Scalar(255), -1);
    }

    bool IsUnder(Transform t, Transform ancestor)
    {
        for (var p = t; p != null; p = p.parent) if (p == ancestor) return true;
        return false;
    }

    void LateUpdate()
    {
        if (++_frameCount % processEveryNFrames != 0) return;

        // Resize RT if window or scale changed
        int tw = Mathf.RoundToInt(cam.pixelWidth * scale);
        int th = Mathf.RoundToInt(cam.pixelHeight * scale);
        if (_rt.width != tw || _rt.height != th)
        {
            SetupRT(); // also resizes Mats
        }

        // Capture camera -> Texture2D
        var prev = cam.targetTexture;
        cam.targetTexture = _rt;
        cam.Render();
        RenderTexture.active = _rt;
        _tex.ReadPixels(new URect(0, 0, _rt.width, _rt.height), 0, 0, false);
        _tex.Apply(false);
        RenderTexture.active = null;
        cam.targetTexture = prev;

        int W = _rt.width, H = _rt.height;
        EnsureMatsMatchTexture(W, H);

        // Unity Texture2D -> OpenCV Mat (RGBA) -> GRAY
        Utils.texture2DToMat(_tex, _rgba);
        Imgproc.cvtColor(_rgba, _gray, Imgproc.COLOR_RGBA2GRAY);

        // Build/clear ROI mask as requested
        BuildMaskIfNeeded(W, H);

        // Detect corners
        using (var corners = new MatOfPoint())
        {
            Imgproc.goodFeaturesToTrack(
                _gray, corners, maxCorners, qualityLevel, minDistance,
                _mask, blockSize, useHarris, harrisK);

            // Store for overlay
            _cornersUV.Clear();
            var pts = corners.toArray();
            for (int i = 0; i < pts.Length; i++)
                _cornersUV.Add(new Vector2((float)pts[i].x, (float)pts[i].y));
        }

        // Raycast to get world XYZ (for corners that hit colliders)
        _lastWorldHits.Clear();
        int logs = 0;
        for (int i = 0; i < _cornersUV.Count; i++)
        {
            var uv = _cornersUV[i];

            // Map capture (u,v) -> GameView pixel coords for ScreenPointToRay
            float sx = uv.x / W * cam.pixelWidth;
            float sy = (1f - uv.y / H) * cam.pixelHeight; // flip Y for Unity

            Ray ray = cam.ScreenPointToRay(new Vector3(sx, sy, 0));
            if (Physics.Raycast(ray, out RaycastHit hit, maxRaycastDistance, raycastMask))
            {
                if (onlyUnderRoot != null && !IsUnder(hit.collider.transform, onlyUnderRoot)) goto skipAdd;

                _lastWorldHits.Add(hit.point);

                if (logEveryFrame && logs < logMaxPerFrame)
                {
                    Debug.Log($"[CornersXYZ] Feat {i:D3} pixel=({uv.x:F1},{uv.y:F1})  XYZ={hit.point}");
                    logs++;
                }
            }
        skipAdd:;
        }
    }

    void OnGUI()
    {
        if (!drawOnGameView || _cornersUV.Count == 0 || !cam || _rt == null) return;

        int W = _rt.width, H = _rt.height;
        int s = Mathf.Max(1, overlayDotSize);

        for (int i = 0; i < _cornersUV.Count; i++)
        {
            var sp = UVToScreenPoint(_cornersUV[i], W, H); // Screen coords, origin bottom-left
            float yGUI = Screen.height - sp.y;             // IMGUI origin top-left

            var r = new URect(sp.x - s, yGUI - s, 2 * s, 2 * s);
            GUI.DrawTexture(r, Texture2D.whiteTexture);
        }
    }

    void OnDrawGizmos()
    {
        if (!Application.isPlaying || !drawWorldGizmos) return;

        Gizmos.color = worldGizmoColor;
        for (int i = 0; i < _lastWorldHits.Count; i++)
            Gizmos.DrawSphere(_lastWorldHits[i], worldGizmoSize);
    }

    void OnDestroy()
    {
        _rgba?.Dispose();
        _gray?.Dispose();
        _mask?.Dispose();

        if (_rt) Destroy(_rt);
        if (_tex) Destroy(_tex);
    }
}
