using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using System.IO;

[DisallowMultipleComponent]
public class ICPAligner : MonoBehaviour
{
    [Header("IFC Export")]
[Tooltip("Translation offset applied in Unity world BEFORE IFC axis remap (use Add/Subtract toggle).")]
public Vector3 ifcOffset = new Vector3(-40.72f, 0f, 23.44f);
[Tooltip("If true: shifted = unity + offset. If false: shifted = unity - offset.")]
public bool ifcOffsetAdd = true;
[Tooltip("Apply Unity->IFC axis remap (Unity X,Y,Z -> IFC X,Z,Y). Disable if not needed.")]
public bool ifcAxisRemap = true;
    [Header("Objects")]
    public Transform A;
    public Transform B;
    [Tooltip("Optional. If set, MoveAtoB_FromBA_Local will move this transform.")]
    public Transform aToMove;
    [Header("Gravity")]
    public Vector3 gravityUpWorld = Vector3.up;
    [Header("Gravity bias")]
    [Tooltip("Fraction of current tilt corrected per ICP iteration (0..1)")]
    public float uprightBias = 0.05f;   // tiny by default
    public bool enableUprightBias = true;
    [Header("Movement Control")]
    [Tooltip("How to handle child objects during alignment")]
    public MovementMode movementMode = MovementMode.Standard;
    [Tooltip("When using PreserveChildren, also preserve child rotations")]
    public bool preserveChildRotations = true;
    [Tooltip("When using RelativeToParent, move in parent's local space")]
    public bool moveInParentSpace = false;
    [Header("Label logging")]
    public bool writeJsonLabel = true;
    [Tooltip("Relative path goes under persistentDataPath")]
    public string labelJsonPath = "icp_labels.json";
    [Tooltip("Optional semantic tag for this object/class")]
    public string objectLabel = "default";
    [Header("ICP (optional)")]
    public bool runICP = false;
    [Tooltip("Component that exposes: public IReadOnlyList<Vector3> LastWorldHits")]
    public UnityEngine.Object WorldHitsProvider;
    [Tooltip("If no provider, these world points are used")]
    public List<Vector3> FallbackWorldPoints = new();
    [Tooltip("Max model vertices used each iteration (0 = use all)")]
    public int sampleCount = 1500;
    public bool includeSkinned = true;
    public float maxPairDistance = 2.0f;
    [Range(0f, 0.8f)] public float trimFraction = 0.3f;
    public int maxIterations = 30;
    public float epsTranslation = 1e-3f;
    public float epsRotationDeg = 0.2f;
    public bool verbose = true;
    public bool drawPairs = true;
    public float gizmoSize = 0.01f;
    public Color pairColor = Color.cyan;

    [Header("Normal gating")]
    public bool useNormalAngleGate = true;
    [Tooltip("Accept if angle(model normal, hit direction) <= this")]
    public float normalAngleMaxDeg = 45f;
    [Tooltip("Treat ±normal as equivalent")]
    public bool twoSidedNormals = true;

    [Header("Robust settings")]
    [Tooltip("Huber kernel delta [m]")]
    public float huberDelta = 0.02f;
    [Tooltip("Minimum pairs per iteration")]
    public int minPairs = 20;
    [Tooltip("Random yaw restarts for coarse init (0 = none)")]
    public int restarts = 0;
    public float yawJitterDeg = 10f;

    [Header("Hotkeys")]
    public KeyCode runIcpKey = KeyCode.I;
    public KeyCode printKey  = KeyCode.P;
    public KeyCode moveKey   = KeyCode.M;
    // ADD inside ICPAligner (data model + helpers)
    [Serializable]
    class TransformLabelEntry
    {
        public string scene;
        public string hierarchyPath;
        public string objectName;
        public string label;
        public string action;
        public string timestampUtc;
        public float[] deltaT;   // [x,y,z]
        public float[] deltaR;   // quaternion [x,y,z,w]
        public float[] finalPos; // [x,y,z]
        public float[] finalRot; // quaternion [x,y,z,w]
    // NEW: offset (pre-IFC) & IFC-space data
    public float[] offsetPos;      // after applying offset (no axis remap)
    public float[] ifcPos;         // final IFC position
    public float[] ifcDeltaT;      // deltaT in IFC frame
    public float[] ifcRot;         // IFC quaternion
    public float[] ifcEuler;   
    }
    [Serializable]
    class TransformLabelDB { public List<TransformLabelEntry> entries = new(); }

    static string HierarchyPath(Transform t) {
        if (!t) return "";
        var path = t.name;
        var p = t.parent;
        while (p) { path = p.name + "/" + path; p = p.parent; }
        return path;
    }
    static float[] V3(Vector3 v) => new float[]{ v.x, v.y, v.z };
    static float[] Q(Quaternion q) => new float[]{ q.x, q.y, q.z, q.w };

    string ResolveJsonPath() {
        if (string.IsNullOrEmpty(labelJsonPath)) labelJsonPath = "icp_labels.json";
        return Path.IsPathRooted(labelJsonPath)
            ? labelJsonPath
            : Path.Combine(Application.persistentDataPath, labelJsonPath);
    }
    TransformLabelDB LoadDB(string path) {
        try {
            if (File.Exists(path)) {
                var json = File.ReadAllText(path);
                var db = JsonUtility.FromJson<TransformLabelDB>(json);
                return db ?? new TransformLabelDB();
            }
        } catch { /* ignore parse errors and start fresh */ }
        return new TransformLabelDB();
    }
    void SaveDB(string path, TransformLabelDB db) {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir)) Directory.CreateDirectory(dir);
        var json = JsonUtility.ToJson(db, true);
        File.WriteAllText(path, json);
    }
    static float[] Euler(Quaternion q) {
    var e = q.eulerAngles; return new float[]{ e.x, e.y, e.z };
}

// IFC helpers
Vector3 ApplyOffset(Vector3 unityPos) =>
    ifcOffsetAdd ? (unityPos + ifcOffset) : (unityPos - ifcOffset);

Vector3 UnityToIfcVector(Vector3 v)
{
    if (!ifcAxisRemap) return v;
    // Unity (X,Y,Z) -> IFC (X,Z,Y)
    return new Vector3(v.x, v.z, v.y);
}
Vector3 UnityToIfcPosition(Vector3 shiftedPos) => UnityToIfcVector(shiftedPos);

Quaternion UnityToIfcRotation(Quaternion q)
{
    if (!ifcAxisRemap) return q;
    // Remap by transforming basis vectors and reconstructing
    var fwdU = q * Vector3.forward;  // Unity Z
    var upU  = q * Vector3.up;       // Unity Y
    var fwdI = UnityToIfcVector(fwdU);
    var upI  = UnityToIfcVector(upU);
    if (fwdI.sqrMagnitude < 1e-10f || upI.sqrMagnitude < 1e-10f)
        return q;
    return Quaternion.LookRotation(fwdI.normalized, upI.normalized);
}
void LogLabel(Transform target, string action, Quaternion dR, Vector3 dT) {
    if (!writeJsonLabel || !target) return;

    // 1. Raw Unity
    var finalPosUnity = target.position;
    var finalRotUnity = target.rotation;

    // 2. Apply offset in Unity frame (pre-axis remap)
    var offsetPos = ApplyOffset(finalPosUnity);

    // 3. Convert to IFC frame
    var ifcPos = UnityToIfcPosition(offsetPos);
    var ifcDeltaT = UnityToIfcVector(dT);          // (constant translation frame shift does not alter delta)
    var ifcRot = UnityToIfcRotation(finalRotUnity);

    var entry = new TransformLabelEntry {
        scene = target.gameObject.scene.path,
        hierarchyPath = HierarchyPath(target),
        objectName = target.name,
        label = objectLabel,
        action = action,
        timestampUtc = DateTime.UtcNow.ToString("o"),

        // Unity (existing)
        deltaT = V3(dT),
        deltaR = Q(dR),
        finalPos = V3(finalPosUnity),
        finalRot = Q(finalRotUnity),

        // New
        offsetPos = V3(offsetPos),
        ifcPos = V3(ifcPos),
        ifcDeltaT = V3(ifcDeltaT),
        ifcRot = Q(ifcRot),
        ifcEuler = Euler(ifcRot)
    };

    var path = ResolveJsonPath();
    var db = LoadDB(path);
    db.entries.Add(entry);
    SaveDB(path, db);
}

    public enum MovementMode
    {
        Standard,
        PreserveChildren,
        IsolateObject,
        MeshOnly
    }
    // 2) Add helper (place with other statics)
    static Quaternion UprightBias(Transform t, Vector3 worldUp, float bias01)
    {
        if (bias01 <= 0f || !t) return Quaternion.identity;
        worldUp = worldUp.sqrMagnitude > 1e-12f ? worldUp.normalized : Vector3.up;

        Vector3 curUp = (t.rotation * Vector3.up).normalized;
        float angDeg = Vector3.Angle(curUp, worldUp);
        if (angDeg < 1e-3f) return Quaternion.identity;

        Vector3 axis = Vector3.Cross(curUp, worldUp);
        if (axis.sqrMagnitude < 1e-12f) return Quaternion.identity;

        return Quaternion.AngleAxis(angDeg * Mathf.Clamp01(bias01), axis.normalized);
}

    // working buffers
    readonly List<Vector3> _scenePts   = new();  // world hits
    readonly List<Vector3> _modelLocal = new();  // model samples (B-local)
    readonly List<Vector3> _srcWorld   = new();  // transformed model pts (world)
    readonly List<Vector3> _dstWorld   = new();  // matched scene pts (world)
    readonly List<float>   _dists      = new();  // distances for trimming

    // analytic normals (from meshes)
    readonly List<Vector3> _modelLocalNormals = new();
    readonly List<Vector3> _srcNormalsW = new();

    // child preservation
    private readonly List<Transform> childrenToPreserve = new();
    private readonly List<Vector3> originalChildPositions = new();
    private readonly List<Quaternion> originalChildRotations = new();

    // mesh-only movement
    private readonly Dictionary<Renderer, Vector3> originalRendererPositions = new();
    private readonly Dictionary<Renderer, Quaternion> originalRendererRotations = new();

    void Update()
    {
        if (printKey != KeyCode.None && Input.GetKeyDown(printKey)) PrintRelative();
        if (moveKey  != KeyCode.None && Input.GetKeyDown(moveKey))  MoveAtoB_UsingCurrent();
        if (runICP && runIcpKey != KeyCode.None && Input.GetKeyDown(runIcpKey)) RunICP();
    }

    [ContextMenu("Print B→A local")]
    public void PrintRelative()
    {
        if (!A || !B) { Debug.LogError("Set A and B."); return; }
        var (R_BA, t_BA) = Compute_B_to_A_Local(B, A);
        Debug.Log($"B→A local  t={t_BA}  Rquat={R_BA}  Reuler={R_BA.eulerAngles}");
    }

    [ContextMenu("Move A to B (using current B→A)")]
    // MODIFY MoveAtoB_UsingCurrent(): capture delta and log
    [ContextMenu("Move A to B (using current B→A)")]
    public void MoveAtoB_UsingCurrent()
    {
        if (!A || !B) { Debug.LogError("Set A and B."); return; }
        var (R_BA, t_BA) = Compute_B_to_A_Local(B, A);

        var target = aToMove ? aToMove : A;
        var p0 = target.position; var q0 = target.rotation;

        MoveAtoB_FromBA_Local(target, B, R_BA, t_BA);

        // world-frame delta x' = dR * x + dT
        var dR = target.rotation * Quaternion.Inverse(q0);
        var dT = target.position - (dR * p0);
        LogLabel(target, "MoveAtoB", dR, dT);
    }


    [ContextMenu("Run ICP (align B to world points)")]
    public void RunICP(float? bias01 = null)
    {
        if (!B) { Debug.LogError("Set B (modelRoot)."); return; }
        if (!TryGetWorldHits(out var hits) || hits.Count < 6)
        {
            Debug.LogWarning($"ICP: need >=6 world points. Got {hits?.Count ?? 0}.");
            return;
        }

        _scenePts.Clear();
        _scenePts.AddRange(hits);

        // Use analytic mesh normals
        MeshToPointCloudLocalWithNormals(B, _modelLocal, _modelLocalNormals, sampleCount, includeSkinned);
        if (_modelLocal.Count < 3) { Debug.LogWarning("ICP: model has insufficient vertices."); return; }

        if (verbose) Debug.Log($"[ICP] start  scenePts={_scenePts.Count} modelSamples={_modelLocal.Count} model={B.name} mode={movementMode}");

        PrepareMovementMode(B);

        var p0 = B.position;
        var q0 = B.rotation;

        float bestRms = float.PositiveInfinity;
        Vector3 bestPos = p0;
        Quaternion bestRot = q0;

        try
        {
            int attempts = Mathf.Max(1, restarts + 1);
            for (int attempt = 0; attempt < attempts; attempt++)
            {
                B.SetPositionAndRotation(p0, q0);

                if (attempt > 0)
                {
                    float step = Mathf.Min(yawJitterDeg, 180f);
                    float mag = ((attempt + 1) / 2) * step;
                    float sign = (attempt % 2 == 1) ? 1f : -1f;
                var qYaw = Quaternion.AngleAxis(sign * mag, gravityUpWorld.normalized);

                    B.rotation = qYaw * B.rotation;
                }

                float currMaxPair = Mathf.Max(1e-6f, maxPairDistance);
                float prevRms = float.PositiveInfinity;

                for (int iter = 0; iter < maxIterations; iter++)
                {
                    BuildSourceWorldPointsAndNormals(B, _modelLocal, _modelLocalNormals, _srcWorld, _srcNormalsW);

                    _dstWorld.Clear();
                    _dists.Clear();
                    int valid = 0;

                    // initial correspondences with distance + normal gating
                    for (int i = 0; i < _srcWorld.Count; i++)
                    {
                        Nearest(_scenePts, _srcWorld[i], out int idx, out float d2);
                        float d = Mathf.Sqrt(d2);
                        if (d <= currMaxPair)
                        {
                            if (useNormalAngleGate && i < _srcNormalsW.Count)
                            {
                                Vector3 n = _srcNormalsW[i];
                                if (n.sqrMagnitude > 1e-12f)
                                {
                                    Vector3 dir = (_scenePts[idx] - _srcWorld[i]).normalized;
                                    float angGate = Vector3.Angle(n.normalized, dir);
                                    if (twoSidedNormals) angGate = Mathf.Min(angGate, 180f - angGate);
                                    if (angGate > normalAngleMaxDeg)
                                    {
                                        _dstWorld.Add(Vector3.positiveInfinity);
                                        _dists.Add(float.PositiveInfinity);
                                        continue;
                                    }
                                }
                            }
                            _dstWorld.Add(_scenePts[idx]);
                            _dists.Add(d);
                            valid++;
                        }
                        else
                        {
                            _dstWorld.Add(Vector3.positiveInfinity);
                            _dists.Add(float.PositiveInfinity);
                        }
                    }

                    // relax distance cap if too few pairs
                    int relax = 0;
                    int need = Mathf.Max(6, minPairs);
                    while (valid < need && relax < 3)
                    {
                        currMaxPair *= 1.5f;
                        valid = 0;
                        for (int i = 0; i < _srcWorld.Count; i++)
                        {
                            if (!float.IsInfinity(_dists[i])) { valid++; continue; }
                            Nearest(_scenePts, _srcWorld[i], out int idx, out float d2);
                            float d = Mathf.Sqrt(d2);
                            if (d <= currMaxPair)
                            {
                                if (useNormalAngleGate && i < _srcNormalsW.Count)
                                {
                                    Vector3 n = _srcNormalsW[i];
                                    if (n.sqrMagnitude > 1e-12f)
                                    {
                                        Vector3 dir = (_scenePts[idx] - _srcWorld[i]).normalized;
                                        float angGate = Vector3.Angle(n.normalized, dir);
                                        if (twoSidedNormals) angGate = Mathf.Min(angGate, 180f - angGate);
                                        if (angGate > normalAngleMaxDeg) continue;
                                    }
                                }
                                _dstWorld[i] = _scenePts[idx];
                                _dists[i] = d;
                                valid++;
                            }
                        }
                        relax++;
                    }
                    if (valid < need) { if (verbose) Debug.LogWarning($"[ICP] too few valid pairs. {valid}"); break; }

                    // adaptive trimming
                    float tIter = (maxIterations > 1) ? (iter / (float)(maxIterations - 1)) : 1f;
                    float trim = Mathf.Clamp01(Mathf.Lerp(Mathf.Min(0.8f, trimFraction * 1.25f), trimFraction, tIter));
                    int kept = TrimByPercentile(_srcWorld, _dstWorld, _dists, trim);
                    if (kept < need) { if (verbose) Debug.LogWarning("[ICP] too few pairs after trim."); break; }

                    // compact lists
                    var src = new List<Vector3>(kept);
                    var dst = new List<Vector3>(kept);
                    for (int i = 0; i < _srcWorld.Count; i++)
                    {
                        if (float.IsInfinity(_dstWorld[i].x)) continue;
                        src.Add(_srcWorld[i]);
                        dst.Add(_dstWorld[i]);
                    }

                    // robust weights
                    var residuals = new List<float>(src.Count);
                    for (int i = 0; i < src.Count; i++) residuals.Add(Vector3.Distance(src[i], dst[i]));
                    float sigma = RobustScale(residuals);
                    float delta = (huberDelta > 0f) ? huberDelta : 1.5f * sigma;
                    if (delta <= 1e-6f) delta = Mathf.Max(epsTranslation * 10f, 1e-4f);

                    var w = new List<float>(src.Count);
                    for (int i = 0; i < residuals.Count; i++) w.Add(HuberWeight(residuals[i], delta));

                    // solve
                    if (!WeightedRigidFit(src, dst, w, out var Rdelta, out var tdelta))
                    {
                        Debug.LogWarning("[ICP] rigid fit failed.");
                        break;
                    }

                    float ang = Quaternion.Angle(Quaternion.identity, Rdelta);
                    float trans = tdelta.magnitude;
                    float b = Mathf.Clamp01(bias01 ?? uprightBias);
                    var Rfit = Rdelta;
                    var tfit = tdelta;

                    // log pre-bias (what ICP solved for)
                    Debug.Log($"[ICP] {b}");

                    if (enableUprightBias && b > 0f)
                    {
                        var Rb = UprightBias(B, gravityUpWorld, b);
                        Vector3 cb = WeightedCentroid(dst, w);   // world-space anchor

                        Rdelta = Rb * Rdelta;
                        tdelta = Rb * (tdelta - cb) + cb;        // pivot at cb
                    }

                    ApplyTransformation(B, Rdelta, tdelta, 0f);

                    // rms
                    float rms = 0f;
                    for (int i = 0; i < src.Count; i++) rms += (src[i] - dst[i]).sqrMagnitude;
                    rms = Mathf.Sqrt(rms / Mathf.Max(1, src.Count));

                    if (verbose)
                        Debug.Log($"[ICP] iter={iter} pairs={src.Count} rms={rms:F4} dAng={ang:F3} dTrans={trans:F4} trim={trim:F2}");

                    if (Mathf.Abs(prevRms - rms) < 1e-6f && ang <= epsRotationDeg && trans <= epsTranslation) break;
                    prevRms = rms;
                    if (ang <= epsRotationDeg && trans <= epsTranslation) break;
                }

                // score attempt
                BuildSourceWorldPointsAndNormals(B, _modelLocal, _modelLocalNormals, _srcWorld, _srcNormalsW);
                float sum2 = 0f; int cnt = 0;
                _dstWorld.Clear();
                for (int i = 0; i < _srcWorld.Count; i++)
                {
                    Nearest(_scenePts, _srcWorld[i], out int idx, out float d2);
                    float d = Mathf.Sqrt(d2);
                    if (d <= maxPairDistance) { _dstWorld.Add(_scenePts[idx]); sum2 += d2; cnt++; }
                    else _dstWorld.Add(Vector3.positiveInfinity);
                }
                float score = (cnt > 0) ? Mathf.Sqrt(sum2 / cnt) : float.PositiveInfinity;

                if (score < bestRms)
                {
                    bestRms = score;
                    bestPos = B.position;
                    bestRot = B.rotation;
                }
            }

            // commit best and refresh gizmo pairs
            B.SetPositionAndRotation(bestPos, bestRot);
            BuildSourceWorldPointsAndNormals(B, _modelLocal, _modelLocalNormals, _srcWorld, _srcNormalsW);
            _dstWorld.Clear();
            for (int i = 0; i < _srcWorld.Count; i++)
            {
                Nearest(_scenePts, _srcWorld[i], out int idx, out float d2);
                float d = Mathf.Sqrt(d2);
                _dstWorld.Add(d <= maxPairDistance ? _scenePts[idx] : Vector3.positiveInfinity);
            }

            var Rtotal = B.rotation * Quaternion.Inverse(q0);
            var ttotal = B.position - (Rtotal * p0);
            Debug.Log($"[ICP] ΔB world  t={ttotal}  Rquat={Rtotal}  Reuler={Rtotal.eulerAngles}");
            LogLabel(B, "ICP", Rtotal, ttotal);
            if (A)
            {
                var (R_BA, t_BA) = Compute_B_to_A_Local(B, A);
                Debug.Log($"[ICP] B→A local  t={t_BA}  Rquat={R_BA}  Reuler={R_BA.eulerAngles}");
            }
        }
        finally
        {
            CleanupMovementMode(B);
        }
    }

    // movement mode plumbing
    void PrepareMovementMode(Transform target)
    {
        switch (movementMode)
        {
            case MovementMode.PreserveChildren:
                PrepareChildPreservation(target); break;
            case MovementMode.IsolateObject:
                IsolateObjectChildren(target); break;
            case MovementMode.MeshOnly:
                PrepareMeshOnlyMovement(target); break;
        }
    }
    void CleanupMovementMode(Transform target)
    {
        switch (movementMode)
        {
            case MovementMode.IsolateObject:
                RestoreObjectChildren(target); break;
        }
    }
    void PrepareChildPreservation(Transform target)
    {
        childrenToPreserve.Clear();
        originalChildPositions.Clear();
        originalChildRotations.Clear();
        for (int i = 0; i < target.childCount; i++)
        {
            var child = target.GetChild(i);
            childrenToPreserve.Add(child);
            originalChildPositions.Add(child.position);
            originalChildRotations.Add(child.rotation);
        }
    }
    void IsolateObjectChildren(Transform target)
    {
        childrenToPreserve.Clear();
        originalChildPositions.Clear();
        originalChildRotations.Clear();
        for (int i = target.childCount - 1; i >= 0; i--)
        {
            var child = target.GetChild(i);
            childrenToPreserve.Add(child);
            originalChildPositions.Add(child.localPosition);
            originalChildRotations.Add(child.localRotation);
            child.SetParent(null, true);
        }
    }
    void RestoreObjectChildren(Transform target)
    {
        if (childrenToPreserve.Count == 0) return;
        for (int i = 0; i < childrenToPreserve.Count; i++)
        {
            if (!childrenToPreserve[i]) continue;
            childrenToPreserve[i].SetParent(target, false);
            childrenToPreserve[i].localPosition = originalChildPositions[i];
            childrenToPreserve[i].localRotation = originalChildRotations[i];
        }
        childrenToPreserve.Clear();
        originalChildPositions.Clear();
        originalChildRotations.Clear();
    }
    void PrepareMeshOnlyMovement(Transform target)
    {
        originalRendererPositions.Clear();
        originalRendererRotations.Clear();
        var renderers = target.GetComponentsInChildren<Renderer>();
        foreach (var r in renderers)
        {
            originalRendererPositions[r] = r.transform.position;
            originalRendererRotations[r] = r.transform.rotation;
        }
    }
    static Vector3 WeightedCentroid(List<Vector3> pts, List<float> w)
{
    double sw = 0;
    Vector3 c = Vector3.zero;
    int n = Mathf.Min(pts.Count, w.Count);
    for (int i = 0; i < n; i++)
    {
        float wi = Mathf.Max(1e-8f, w[i]);
        sw += wi; c += wi * pts[i];
    }
    return (sw > 0) ? (c / (float)sw) : Vector3.zero;
}

    void ApplyTransformation(Transform target, Quaternion deltaRotation, Vector3 deltaTranslation, float? bias01 = null)
    {
        float bias = Mathf.Clamp01(bias01 ?? uprightBias);
        if (enableUprightBias && bias > 0f)
        {
            var Rb = UprightBias(target, gravityUpWorld, bias);
            deltaRotation    = Rb * deltaRotation;
            deltaTranslation = Rb * deltaTranslation; // needed for correct left-multiply composition
        }

        switch (movementMode)
        {
            case MovementMode.Standard:
                ApplyStandardTransformation(target, deltaRotation, deltaTranslation); break;
            case MovementMode.PreserveChildren:
                ApplyStandardTransformation(target, deltaRotation, deltaTranslation);
                RestoreChildrenPositions(); break;
            case MovementMode.IsolateObject:
                ApplyStandardTransformation(target, deltaRotation, deltaTranslation); break;
            case MovementMode.MeshOnly:
                ApplyMeshOnlyTransformation(target, deltaRotation, deltaTranslation); break;
        }
    }


void ApplyStandardTransformation(Transform target, Quaternion dR, Vector3 dT)
{
    if (moveInParentSpace && target.parent)
    {
        var p = target.parent;
        var dRloc = Quaternion.Inverse(p.rotation) * dR * p.rotation;
        var dTloc = p.InverseTransformVector(dT);     // handles parent scale

        target.localRotation = dRloc * target.localRotation;
        target.localPosition = dRloc * target.localPosition + dTloc;
    }
    else
    {
        target.SetPositionAndRotation(dR * target.position + dT, dR * target.rotation);
    }
}

    void RestoreChildrenPositions()
    {
        for (int i = 0; i < childrenToPreserve.Count; i++)
        {
            if (!childrenToPreserve[i]) continue;
            childrenToPreserve[i].position = originalChildPositions[i];
            if (preserveChildRotations) childrenToPreserve[i].rotation = originalChildRotations[i];
        }
    }
    void ApplyMeshOnlyTransformation(Transform target, Quaternion deltaRotation, Vector3 deltaTranslation)
    {
        Matrix4x4 deltaMatrix = Matrix4x4.TRS(deltaTranslation, deltaRotation, Vector3.one);
        foreach (var kv in originalRendererPositions)
        {
            var r = kv.Key; if (!r) continue;
            Vector3 originalPos = kv.Value;
            Quaternion originalRot = originalRendererRotations[r];
            Vector3 currentPos = deltaMatrix.MultiplyPoint3x4(originalPos);
            Quaternion currentRot = deltaRotation * originalRot;
            r.transform.SetPositionAndRotation(currentPos, currentRot);
        }
    }

    // world hits
    bool TryGetWorldHits(out List<Vector3> hits)
    {
        hits = null;
        if (WorldHitsProvider)
        {
            if (TryExtractFromObject(WorldHitsProvider, out hits))
            {
                if (verbose) Debug.Log($"[ICP] provider {WorldHitsProvider.GetType().Name} -> {hits.Count} pts");
                return hits.Count >= 6;
            }
            if (WorldHitsProvider is GameObject go)
            {
                var comps = go.GetComponents<Component>();
                foreach (var c in comps)
                {
                    if (!c) continue;
                    if (TryExtractFromObject(c, out hits))
                    {
                        if (verbose) Debug.Log($"[ICP] provider {c.GetType().Name} (component) -> {hits.Count} pts");
                        return hits.Count >= 6;
                    }
                }
            }
            Debug.LogWarning($"[ICP] Provider set ({WorldHitsProvider.GetType().Name}) but no usable points found. " +
                             $"Expose Vector3 world points as LastWorldHits/WorldHits or GetLastWorldHits()/GetWorldHits().");
        }
        if (FallbackWorldPoints != null && FallbackWorldPoints.Count >= 6)
        {
            hits = new List<Vector3>(FallbackWorldPoints);
            if (verbose) Debug.Log($"[ICP] fallback list -> {hits.Count} pts");
            return true;
        }
        return false;
    }
    static bool TryExtractFromObject(object obj, out List<Vector3> hits)
    {
        hits = null; if (obj == null) return false;
        var t = obj.GetType();
        string[] propNames = { "LastWorldHits", "WorldHits", "WorldPoints", "Points" };
        string[] methodNames = { "GetLastWorldHits", "GetWorldHits", "GetWorldPoints", "GetPoints" };

        foreach (var name in propNames)
        {
            var p = t.GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (p == null) continue;
            var val = p.GetValue(obj, null);
            if (ExtractVecs(val, out hits)) return true;
        }
        foreach (var name in propNames)
        {
            var f = t.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (f == null) continue;
            var val = f.GetValue(obj);
            if (ExtractVecs(val, out hits)) return true;
        }
        foreach (var name in methodNames)
        {
            var m = t.GetMethod(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (m == null || m.GetParameters().Length != 0) continue;
            var val = m.Invoke(obj, null);
            if (ExtractVecs(val, out hits)) return true;
        }
        return false;
    }
    static bool ExtractVecs(object src, out List<Vector3> pts)
    {
        pts = null; if (src == null) return false;
        if (src is Vector3[] arr && arr.Length > 0) { pts = new List<Vector3>(arr); return true; }
        if (src is List<Vector3> list && list.Count > 0) { pts = new List<Vector3>(list); return true; }
        if (src is System.Collections.Generic.IEnumerable<Vector3> gen)
        {
            pts = new List<Vector3>(); foreach (var v in gen) pts.Add(v);
            if (pts.Count > 0) return true; pts = null; return false;
        }
        if (src is System.Collections.IEnumerable any)
        {
            pts = new List<Vector3>(); foreach (var v in any) if (v is Vector3 vv) pts.Add(vv);
            if (pts.Count > 0) return true; pts = null; return false;
        }
        return false;
    }

    void OnDrawGizmos()
    {
        if (!drawPairs || _srcWorld.Count == 0 || _dstWorld.Count == 0) return;
        Gizmos.color = pairColor;
        for (int i = 0; i < _srcWorld.Count; i++)
        {
            var a = _srcWorld[i];
            var b = _dstWorld[i];
            if (float.IsInfinity(b.x)) continue;
            Gizmos.DrawSphere(a, gizmoSize);
            Gizmos.DrawSphere(b, gizmoSize);
            Gizmos.DrawLine(a, b);
        }
    }

    // ---------- Public statics ----------

    // x_A = R_BA * x_B + t_BA  (B→A in A-local)
    public static (Quaternion R_BA, Vector3 t_BA) Compute_B_to_A_Local(Transform B, Transform A)
    {
        var T_WB = Matrix4x4.TRS(B.position, B.rotation, Vector3.one);
        var T_WA = Matrix4x4.TRS(A.position, A.rotation, Vector3.one);
        var T_BA = T_WA.inverse * T_WB;
        var R_BA = RotationFromMatrix(T_BA);
        var t_BA = (Vector3)T_BA.GetColumn(3);
        return (R_BA, t_BA);
    }

    // Move A to B using B→A in A-local.
    public static void MoveAtoB_FromBA_Local(Transform A, Transform B, Quaternion R_BA, Vector3 t_BA)
    {
        var R_AB = Quaternion.Inverse(R_BA);
        var t_AB = -(R_AB * t_BA);
        A.SetPositionAndRotation(B.TransformPoint(t_AB), B.rotation * R_AB);
    }

    public static Quaternion RotationFromMatrix(Matrix4x4 m)
    {
        var f = ((Vector3)m.GetColumn(2)).normalized; // Z
        var u = ((Vector3)m.GetColumn(1)).normalized; // Y
        return Quaternion.LookRotation(f, u);
    }

    // ---------- Helpers ----------

    // Point cloud with analytic normals from Unity meshes
    static void MeshToPointCloudLocalWithNormals(Transform root, List<Vector3> outLocalPts, List<Vector3> outLocalNrms, int maxPoints, bool includeSkinned)
    {
        outLocalPts.Clear(); outLocalNrms.Clear();
        if (!root) return;

        var toRoot = root.worldToLocalMatrix;
        int budget = maxPoints > 0 ? Mathf.Max(1, maxPoints) : int.MaxValue;

        var filters = root.GetComponentsInChildren<MeshFilter>(true);
        foreach (var f in filters)
        {
            var m = f ? f.sharedMesh : null;
            if (!m || !m.isReadable) continue;
            var verts = m.vertices; var nrms = m.normals;
            if (verts == null || verts.Length == 0) continue;

            int remaining = Mathf.Max(1, budget - outLocalPts.Count);
            int stride = (budget == int.MaxValue) ? 1 : Mathf.Max(1, Mathf.CeilToInt((float)verts.Length / remaining));

            var toLocal = toRoot * f.transform.localToWorldMatrix;
            for (int i = 0; i < verts.Length && outLocalPts.Count < budget; i += stride)
            {
                outLocalPts.Add(toLocal.MultiplyPoint3x4(verts[i]));
                Vector3 n = (nrms != null && nrms.Length == verts.Length)
                    ? toRoot.MultiplyVector(f.transform.TransformDirection(nrms[i]))
                    : Vector3.zero;
                outLocalNrms.Add(n.normalized);
            }
            if (outLocalPts.Count >= budget) return;
        }

        if (!includeSkinned) return;

        var skinneds = root.GetComponentsInChildren<SkinnedMeshRenderer>(true);
        foreach (var smr in skinneds)
        {
            if (!smr) continue;
            var baked = new Mesh();
            smr.BakeMesh(baked, true);
            var verts = baked.vertices; var nrms = baked.normals;
            if (verts != null && verts.Length > 0)
            {
                int remaining = Mathf.Max(1, budget - outLocalPts.Count);
                int stride = (budget == int.MaxValue) ? 1 : Mathf.Max(1, Mathf.CeilToInt((float)verts.Length / remaining));

                var toLocal = toRoot * smr.transform.localToWorldMatrix;
                for (int i = 0; i < verts.Length && outLocalPts.Count < budget; i += stride)
                {
                    outLocalPts.Add(toLocal.MultiplyPoint3x4(verts[i]));
                    Vector3 n = (nrms != null && nrms.Length == verts.Length)
                        ? toRoot.MultiplyVector(smr.transform.TransformDirection(nrms[i]))
                        : Vector3.zero;
                    outLocalNrms.Add(n.normalized);
                }
            }
#if UNITY_EDITOR
            if (!Application.isPlaying) UnityEngine.Object.DestroyImmediate(baked);
            else UnityEngine.Object.Destroy(baked);
#else
            UnityEngine.Object.Destroy(baked);
#endif
            if (outLocalPts.Count >= budget) return;
        }
    }

    static void MeshToPointCloudLocal(Transform root, List<Vector3> outLocal, int maxPoints = 0, bool includeSkinned = true)
    {
        outLocal.Clear();
        if (!root) return;

        var toRoot = root.worldToLocalMatrix;
        int budget = maxPoints > 0 ? Mathf.Max(1, maxPoints) : int.MaxValue;

        var filters = root.GetComponentsInChildren<MeshFilter>(true);
        foreach (var f in filters)
        {
            var m = f ? f.sharedMesh : null;
            if (!m || !m.isReadable) continue;
            var verts = m.vertices;
            if (verts == null || verts.Length == 0) continue;

            int remaining = Mathf.Max(1, budget - outLocal.Count);
            int stride = (budget == int.MaxValue) ? 1 : Mathf.Max(1, Mathf.CeilToInt((float)verts.Length / remaining));

            var toLocal = toRoot * f.transform.localToWorldMatrix;
            for (int i = 0; i < verts.Length && outLocal.Count < budget; i += stride)
                outLocal.Add(toLocal.MultiplyPoint3x4(verts[i]));

            if (outLocal.Count >= budget) return;
        }

        if (!includeSkinned) return;

        var skinneds = root.GetComponentsInChildren<SkinnedMeshRenderer>(true);
        foreach (var smr in skinneds)
        {
            if (!smr) continue;
            var baked = new Mesh();
            smr.BakeMesh(baked, true);
            var verts = baked.vertices;
            if (verts != null && verts.Length > 0)
            {
                int remaining = Mathf.Max(1, budget - outLocal.Count);
                int stride = (budget == int.MaxValue) ? 1 : Mathf.Max(1, Mathf.CeilToInt((float)verts.Length / remaining));

                var toLocal = toRoot * smr.transform.localToWorldMatrix;
                for (int i = 0; i < verts.Length && outLocal.Count < budget; i += stride)
                    outLocal.Add(toLocal.MultiplyPoint3x4(verts[i]));
            }
#if UNITY_EDITOR
            if (!Application.isPlaying) UnityEngine.Object.DestroyImmediate(baked);
            else UnityEngine.Object.Destroy(baked);
#else
            UnityEngine.Object.Destroy(baked);
#endif
            if (outLocal.Count >= budget) return;
        }
    }

    static void BuildSourceWorldPointsAndNormals(Transform root, List<Vector3> localPts, List<Vector3> localNrms, List<Vector3> outPtsW, List<Vector3> outNrmsW)
    {
        outPtsW.Clear(); outNrmsW.Clear();
        var M = root.localToWorldMatrix;
        for (int i = 0; i < localPts.Count; i++)
        {
            outPtsW.Add(M.MultiplyPoint3x4(localPts[i]));
            Vector3 n = (i < localNrms.Count) ? M.MultiplyVector(localNrms[i]).normalized : Vector3.zero;
            outNrmsW.Add(n);
        }
    }
    static void BuildSourceWorldPoints(Transform root, List<Vector3> localPts, List<Vector3> outWorld)
    {
        outWorld.Clear();
        var M = root.localToWorldMatrix;
        for (int i = 0; i < localPts.Count; i++) outWorld.Add(M.MultiplyPoint3x4(localPts[i]));
    }
    static void Nearest(List<Vector3> cloud, Vector3 p, out int bestIdx, out float bestD2)
    {
        bestIdx = -1; bestD2 = float.PositiveInfinity;
        for (int i = 0; i < cloud.Count; i++)
        {
            float d2 = (cloud[i] - p).sqrMagnitude;
            if (d2 < bestD2) { bestD2 = d2; bestIdx = i; }
        }
    }
    static int TrimByPercentile(List<Vector3> src, List<Vector3> dst, List<float> dists, float trim)
    {
        if (src.Count == 0) return 0;
        var idx = new List<int>(src.Count);
        for (int i = 0; i < src.Count; i++) if (!float.IsInfinity(dst[i].x)) idx.Add(i);
        if (idx.Count == 0) return 0;

        int keep = Mathf.Max(6, Mathf.RoundToInt(idx.Count * (1f - trim)));
        idx.Sort((a, b) => dists[a].CompareTo(dists[b]));
        for (int r = keep; r < idx.Count; r++) dst[idx[r]] = new Vector3(float.PositiveInfinity, 0, 0);
        return keep;
    }
    static float HuberWeight(float r, float delta)
    {
        float a = Mathf.Abs(r);
        return (a <= delta) ? 1f : (delta / a);
    }
    static float RobustScale(List<float> residuals)
    {
        if (residuals.Count == 0) return 0f;
        var tmp = new List<float>(residuals);
        tmp.Sort();
        float med = tmp[tmp.Count / 2];
        for (int i = 0; i < tmp.Count; i++) tmp[i] = Mathf.Abs(tmp[i] - med);
        tmp.Sort();
        float mad = tmp[tmp.Count / 2];
        return 1.4826f * mad + 1e-12f;
    }
    static bool WeightedRigidFit(List<Vector3> Apts, List<Vector3> Bpts, List<float> w, out Quaternion R, out Vector3 t)
    {
        R = Quaternion.identity; t = Vector3.zero;
        int n = Apts.Count;
        if (n != Bpts.Count || n != w.Count || n < 3) return false;

        double sw = 0;
        Vector3 ca = Vector3.zero, cb = Vector3.zero;
        for (int i = 0; i < n; i++)
        {
            float wi = Mathf.Max(1e-8f, w[i]);
            sw += wi; ca += wi * Apts[i]; cb += wi * Bpts[i];
        }
        if (sw <= 0) return false;
        ca /= (float)sw; cb /= (float)sw;

        double Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0;
        for (int i = 0; i < n; i++)
        {
            float wi = Mathf.Max(1e-8f, w[i]);
            Vector3 a = Apts[i] - ca;
            Vector3 b = Bpts[i] - cb;
            Sxx += wi * a.x * b.x; Sxy += wi * a.x * b.y; Sxz += wi * a.x * b.z;
            Syx += wi * a.y * b.x; Syy += wi * a.y * b.y; Syz += wi * a.y * b.z;
            Szx += wi * a.z * b.x; Szy += wi * a.z * b.y; Szz += wi * a.z * b.z;
        }

        double tr = Sxx + Syy + Szz;
        double n00 = tr, n01 = Syz - Szy, n02 = Szx - Sxz, n03 = Sxy - Syx;
        double n11 = Sxx - Syy - Szz, n12 = Sxy + Syx, n13 = Szx + Sxz;
        double n22 = -Sxx + Syy - Szz, n23 = Syz + Szy;
        double n33 = -Sxx - Syy + Szz;

        double q0 = 1, q1 = 0, q2 = 0, q3 = 0;
        for (int k = 0; k < 30; k++)
        {
            double r0 = n00 * q0 + n01 * q1 + n02 * q2 + n03 * q3;
            double r1 = n01 * q0 + n11 * q1 + n12 * q2 + n13 * q3;
            double r2 = n02 * q0 + n12 * q1 + n22 * q2 + n23 * q3;
            double r3 = n03 * q0 + n13 * q1 + n23 * q3;
            double norm = Math.Sqrt(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3);
            if (norm < 1e-12) break;
            q0 = r0 / norm; q1 = r1 / norm; q2 = r2 / norm; q3 = r3 / norm;
        }

        R = new Quaternion((float)q1, (float)q2, (float)q3, (float)q0);
        R.Normalize();
        t = cb - (R * ca);
        return true;
    }
}
