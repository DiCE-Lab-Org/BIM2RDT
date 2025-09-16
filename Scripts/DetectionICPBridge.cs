// DetectionICPBridge.cs
using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using URect = UnityEngine.Rect;

[DisallowMultipleComponent]
public class DetectionICPBridge : MonoBehaviour
{
    [Header("Inputs")]
    [Tooltip("Corners+XYZ provider with ROI/subtree filtering (the patched OpenCVCornersXYZOverlay).")]
    public OpenCVCornersXYZOverlay featureProvider;
    public ICPAligner icp;
    public Camera cam;
    public DetectionOverlay detectionSource;

    [Header("Tag Roots")]
    [Tooltip("Tag on the real-world hierarchy root.")]
    public string realWorldTag = "realWorld";
    [Tooltip("Tag on the BIM hierarchy root (the one to update).")]
    public string bimTag = "BIM";

    [Header("Raycast")]
    [Tooltip("Optional override. Set to a mask that includes ONLY scene layers.")]
    public LayerMask sceneRaycastMask = ~0;
    public bool overrideFeatureProviderMask = true;

    [Header("Selection & ROI")]
    public string targetLabel = "";
    [Range(0f,1f)] public float minScore = 0.50f;
    public Vector2 roiPad01 = new Vector2(0.02f, 0.02f);
    [Range(0f, 0.25f)] public float minBoxArea01 = 0.01f;

    [Header("Gating")]
    public int minHits = 20;
    public float cooldownSec = 0.3f;
    public bool requireStable = true;
    public int stableForN = 2;
    [Range(0.3f, 0.95f)] public float iouStableThresh = 0.6f;
    public bool pickTopOnly = true;
    public KeyCode triggerKey = KeyCode.None;

    // internal roots resolved via tags
    Transform _sceneRoot;
    Transform _bimRoot;

    // state
    readonly Dictionary<string, Transform> _sceneByLabel = new();
    readonly Dictionary<string, Transform> _bimByLabel = new();
    readonly Dictionary<string, Stable> _stable = new();
    float _lastRunAt = -999f;

    struct Stable { public URect roi; public int count; }

    void Awake()
    {
        if (!featureProvider || !icp || !cam || !detectionSource)
        {
            Debug.LogError("DetectionICPBridge: assign featureProvider, icp, cam, detectionSource.");
            enabled = false; return;
        }

        _sceneRoot = FindByTagSafe(realWorldTag);
        _bimRoot   = FindByTagSafe(bimTag);

        if (!_sceneRoot || !_bimRoot)
        {
            Debug.LogError($"DetectionICPBridge: missing tagged roots. realWorldTag='{realWorldTag}' found={_sceneRoot}; bimTag='{bimTag}' found={_bimRoot}");
            enabled = false; return;
        }
        IndexChildren();
    }

    void Update()
    {
        if (!TryGetLatestFrame(out var f)) return;

        bool manual = (triggerKey != KeyCode.None && Input.GetKeyDown(triggerKey));

        var dets = f.detections ?? Array.Empty<Detection>();
        if (dets.Length == 0) return;

        var cand = new List<(Detection d, string label)>(dets.Length);
        for (int i = 0; i < dets.Length; i++)
        {
            var d = dets[i];
            if (d == null || d.score < minScore) continue;

            var label = SafeLabel(d);
            if (!string.IsNullOrEmpty(targetLabel) && !LabelEquals(label, targetLabel)) continue;
            if (!_sceneByLabel.ContainsKey(label) || !_bimByLabel.ContainsKey(label)) continue;

            cand.Add((d, label));
        }
        if (cand.Count == 0) return;

        cand.Sort((a,b) => b.d.score.CompareTo(a.d.score));
        int count = pickTopOnly ? 1 : cand.Count;

        for (int i = 0; i < count; i++)
        {
            var (d, label) = cand[i];
            var sceneNode = _sceneByLabel[label];
            var bimNode   = _bimByLabel[label];

            // bbox -> normalized ROI [0..1], origin top-left
            float nx1 = Mathf.Clamp01((float)d.bbox_xyxy[0] / Mathf.Max(1, f.width));
            float ny1 = Mathf.Clamp01((float)d.bbox_xyxy[1] / Mathf.Max(1, f.height));
            float nx2 = Mathf.Clamp01((float)d.bbox_xyxy[2] / Mathf.Max(1, f.width));
            float ny2 = Mathf.Clamp01((float)d.bbox_xyxy[3] / Mathf.Max(1, f.height));

            float px = roiPad01.x, py = roiPad01.y;
            float x = Mathf.Clamp01(nx1 - px);
            float y = Mathf.Clamp01(ny1 - py);
            float w = Mathf.Clamp01((nx2 - nx1) + 2*px);
            float h = Mathf.Clamp01((ny2 - ny1) + 2*py);
            if (x + w > 1f) w = 1f - x;
            if (y + h > 1f) h = 1f - y;

            var roi = new URect(x, y, w, h);
            if (!manual && roi.width * roi.height < minBoxArea01) continue;

            // collect features on the REAL WORLD subtree only
            LayerMask prev = featureProvider.raycastMask;
            if (overrideFeatureProviderMask) featureProvider.raycastMask = sceneRaycastMask;
            featureProvider.SetROI01(roi);
            featureProvider.SetOnlyUnderRoot(sceneNode);
            featureProvider.ProcessOnceImmediate();
            if (overrideFeatureProviderMask) featureProvider.raycastMask = prev;

            int hits = featureProvider.LastWorldHits?.Count ?? 0;
            if (!manual && hits < Mathf.Max(6, minHits)) continue;

            if (requireStable && !manual && !IsStable(label, roi)) continue;
            if (!manual && (Time.time - _lastRunAt) < Mathf.Max(0f, cooldownSec)) continue;

            // align BIM subtree to the scene features
            icp.WorldHitsProvider = featureProvider;
            icp.B = bimNode;        // BIM gets updated
            icp.RunICP();
            _lastRunAt = Time.time;
        }
    }

    void IndexChildren()
    {
        _sceneByLabel.Clear(); _bimByLabel.Clear();

        foreach (Transform c in _sceneRoot.GetComponentsInChildren<Transform>(true))
        {
            if (c == _sceneRoot) continue;
            string key = c.name.Trim().ToLowerInvariant();
            if (!_sceneByLabel.ContainsKey(key)) _sceneByLabel.Add(key, c);
        }
        foreach (Transform c in _bimRoot.GetComponentsInChildren<Transform>(true))
        {
            if (c == _bimRoot) continue;
            string key = c.name.Trim().ToLowerInvariant();
            if (!_bimByLabel.ContainsKey(key)) _bimByLabel.Add(key, c);
        }
    }

    static bool LabelEquals(string a, string b) =>
        string.Equals(a?.Trim(), b?.Trim(), StringComparison.OrdinalIgnoreCase);

    static string SafeLabel(Detection d)
    {
        if (d == null) return "(unknown)";
        var t = d.GetType();

        var p = t.GetProperty("label", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (p != null) { var s = p.GetValue(d) as string; if (!string.IsNullOrEmpty(s)) return s.Trim().ToLowerInvariant(); }
        var f = t.GetField("label", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (f != null) { var s = f.GetValue(d) as string; if (!string.IsNullOrEmpty(s)) return s.Trim().ToLowerInvariant(); }

        string[] idNames = { "class_id", "classId", "category_id", "categoryId", "id" };
        foreach (var name in idNames)
        {
            var pp = t.GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (pp != null) { try { return $"id:{Convert.ToInt32(pp.GetValue(d))}"; } catch {} }
            var ff = t.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (ff != null) { try { return $"id:{Convert.ToInt32(ff.GetValue(d))}"; } catch {} }
        }
        return "(unknown)";
    }

    bool IsStable(string label, URect roi)
    {
        string key = label ?? "";
        _stable.TryGetValue(key, out var s);
        float iou = IoU01(s.roi, roi);
        s.count = (iou >= iouStableThresh) ? (s.count + 1) : 1;
        s.roi = roi;
        _stable[key] = s;
        return s.count >= Mathf.Max(1, stableForN);
    }

    static float IoU01(URect a, URect b)
    {
        if (a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0) return 0f;
        float x1 = Mathf.Max(a.xMin, b.xMin), y1 = Mathf.Max(a.yMin, b.yMin);
        float x2 = Mathf.Min(a.xMax, b.xMax), y2 = Mathf.Min(a.yMax, b.yMax);
        float iw = Mathf.Max(0f, x2 - x1), ih = Mathf.Max(0f, y2 - y1);
        float inter = iw * ih;
        float uni = a.width * a.height + b.width * b.height - inter;
        return (uni > 0f) ? inter / uni : 0f;
    }

    // -------- frame access without a concrete FrameMessage type --------
    struct FrameLite { public int width, height; public Detection[] detections; }

    bool TryGetLatestFrame(out FrameLite frame)
    {
        frame = default;
        if (!detectionSource) return false;

        var detsRO = detectionSource.LastDetections;
        if (detsRO == null || detsRO.Count == 0) return false;

        var dets = new Detection[detsRO.Count];
        for (int i = 0; i < dets.Length; i++) dets[i] = detsRO[i];

        frame = new FrameLite {
            width = detectionSource.ImageWidth,
            height = detectionSource.ImageHeight,
            detections = dets
        };
        return true;
    }

    static Transform FindByTagSafe(string tagName)
    {
        try
        {
            var go = GameObject.FindGameObjectWithTag(tagName);
            return go ? go.transform : null;
        }
        catch (UnityException)
        {
            return null;
        }
    }
}
