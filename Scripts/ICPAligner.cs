using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

[DisallowMultipleComponent]
public class ICPAligner : MonoBehaviour
{
    [Header("Objects")]
    public Transform A;            // reference frame
    public Transform B;            // model to align or report
    [Tooltip("Optional. If set, MoveAtoB_FromBA_Local will move this transform.")]
    public Transform aToMove;

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
    public float epsTranslation = 1e-3f;   // meters
    public float epsRotationDeg = 0.2f;    // degrees
    public bool verbose = true;
    public bool drawPairs = true;
    public float gizmoSize = 0.01f;
    public Color pairColor = Color.cyan;

    [Header("Hotkeys")]
    public KeyCode runIcpKey = KeyCode.I;
    public KeyCode printKey  = KeyCode.P;
    public KeyCode moveKey   = KeyCode.M;

    // working buffers
    readonly List<Vector3> _scenePts = new();     // A hits (world)
    readonly List<Vector3> _modelLocal = new();   // sampled B verts (B-local, relative to B root)
    readonly List<Vector3> _srcWorld = new();     // transformed B points (world)
    readonly List<Vector3> _dstWorld = new();     // matched scene points (world)
    readonly List<float>   _dists    = new();     // distances for trimming

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
    public void MoveAtoB_UsingCurrent()
    {
        if (!A || !B) { Debug.LogError("Set A and B."); return; }
        var (R_BA, t_BA) = Compute_B_to_A_Local(B, A);
        MoveAtoB_FromBA_Local(aToMove ? aToMove : A, B, R_BA, t_BA);
    }

    [ContextMenu("Run ICP (align B to world points)")]
    public void RunICP()
    {
        if (!B) { Debug.LogError("Set B (modelRoot)."); return; }
        if (!TryGetWorldHits(out var hits) || hits.Count < 6)
        {
            Debug.LogWarning($"ICP: need >=6 world points. Got {hits?.Count ?? 0}.");
            return;
        }

        // scene set
        _scenePts.Clear();
        _scenePts.AddRange(hits);

        // sample model B (local, in B-root coordinates)
        MeshToPointCloudLocal(B, _modelLocal, sampleCount, includeSkinned);
        if (_modelLocal.Count < 3) { Debug.LogWarning("ICP: model has insufficient vertices."); return; }

        if (verbose) Debug.Log($"[ICP] start  scenePts={_scenePts.Count} modelSamples={_modelLocal.Count} model={B.name}");

        // capture initial pose
        var p0 = B.position;
        var q0 = B.rotation;

        // ICP loop in world
        for (int iter = 0; iter < maxIterations; iter++)
        {
            // transform samples to world
            BuildSourceWorldPoints(B, _modelLocal, _srcWorld);

            // nearest neighbors
            _dstWorld.Clear();
            _dists.Clear();
            int valid = 0;
            for (int i = 0; i < _srcWorld.Count; i++)
            {
                Nearest(_scenePts, _srcWorld[i], out int idx, out float d2);
                float d = Mathf.Sqrt(d2);
                if (d <= maxPairDistance)
                {
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
            if (valid < 6) { Debug.LogWarning("[ICP] too few valid pairs."); break; }

            // trim worst
            int kept = TrimByPercentile(_srcWorld, _dstWorld, _dists, trimFraction);
            if (kept < 6) { Debug.LogWarning("[ICP] too few pairs after trim."); break; }

            // pack A(src) and B(dst) for fit
            var src = new List<Vector3>(kept);
            var dst = new List<Vector3>(kept);
            for (int i = 0; i < _srcWorld.Count; i++)
            {
                if (float.IsInfinity(_dstWorld[i].x)) continue;
                src.Add(_srcWorld[i]);
                dst.Add(_dstWorld[i]);
            }

            // rigid delta mapping src -> dst
            if (!RigidFit.TryEstimate(src, dst, out var Rdelta, out var tdelta))
            {
                Debug.LogWarning("[ICP] rigid fit failed.");
                break;
            }

            // apply: T_new = T_delta * T_current  (left-multiply in world)
            float ang = Quaternion.Angle(Quaternion.identity, Rdelta);
            float trans = tdelta.magnitude;

            B.SetPositionAndRotation(
                Rdelta * B.position + tdelta,
                Rdelta * B.rotation
            );

            if (verbose)
            {
                float rms = 0f;
                for (int i = 0; i < src.Count; i++) rms += (src[i] - dst[i]).sqrMagnitude;
                rms = Mathf.Sqrt(rms / Mathf.Max(1, src.Count));
                Debug.Log($"[ICP] iter={iter} pairs={src.Count} rms={rms:F4} dAng={ang:F3} dTrans={trans:F4}");
            }

            if (ang <= epsRotationDeg && trans <= epsTranslation) break;
        }

        // report total ΔB in world
        var Rtotal = B.rotation * Quaternion.Inverse(q0);
        var ttotal = B.position - (Rtotal * p0);
        Debug.Log($"[ICP] ΔB world  t={ttotal}  Rquat={Rtotal}  Reuler={Rtotal.eulerAngles}");
        if (A)
        {
            var (R_BA, t_BA) = Compute_B_to_A_Local(B, A);
            Debug.Log($"[ICP] B→A local  t={t_BA}  Rquat={R_BA}  Reuler={R_BA.eulerAngles}");
        }
    }

   // --- Drop-in: replace your TryGetWorldHits + helpers with all of this ---

bool TryGetWorldHits(out List<Vector3> hits)
{
    hits = null;

    // 1) Provider path
    if (WorldHitsProvider)
    {
        // Try the object itself
        if (TryExtractFromObject(WorldHitsProvider, out hits))
        {
            if (verbose) Debug.Log($"[ICP] provider {WorldHitsProvider.GetType().Name} -> {hits.Count} pts");
            return hits.Count >= 6;
        }

        // If a GameObject was dragged, search its components
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

        Debug.LogWarning($"[ICP] Provider set ({WorldHitsProvider.GetType().Name}) but no usable points found on it or its components. " +
                         $"Expose Vector3 world points as LastWorldHits / WorldHits property/field, or GetLastWorldHits()/GetWorldHits() method.");
    }

    // 2) Serialized fallback
    if (FallbackWorldPoints != null && FallbackWorldPoints.Count >= 6)
    {
        hits = new List<Vector3>(FallbackWorldPoints);
        if (verbose) Debug.Log($"[ICP] fallback list -> {hits.Count} pts");
        return true;
    }

    // 3) Nothing
    return false;
}

static bool TryExtractFromObject(object obj, out List<Vector3> hits)
{
    hits = null;
    if (obj == null) return false;

    var t = obj.GetType();

    // candidate member names
    string[] propNames = { "LastWorldHits", "WorldHits", "WorldPoints", "Points" };
    string[] methodNames = { "GetLastWorldHits", "GetWorldHits", "GetWorldPoints", "GetPoints" };

    // Properties
    foreach (var name in propNames)
    {
        var p = t.GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (p == null) continue;
        var val = p.GetValue(obj, null);
        if (ExtractVecs(val, out hits)) return true;
    }

    // Fields
    foreach (var name in propNames)
    {
        var f = t.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (f == null) continue;
        var val = f.GetValue(obj);
        if (ExtractVecs(val, out hits)) return true;
    }

    // Methods (no-arg)
    foreach (var name in methodNames)
    {
        var m = t.GetMethod(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (m == null || m.GetParameters().Length != 0) continue;
        var val = m.Invoke(obj, null);
        if (ExtractVecs(val, out hits)) return true;
    }

    return false;
}

// Accepts IEnumerable<Vector3>, List<Vector3>, Vector3[], or IEnumerable of objects where each is Vector3.
static bool ExtractVecs(object src, out List<Vector3> pts)
{
    pts = null;
    if (src == null) return false;

    if (src is Vector3[] arr && arr.Length > 0)
    {
        pts = new List<Vector3>(arr);
        return true;
    }
    if (src is List<Vector3> list && list.Count > 0)
    {
        pts = new List<Vector3>(list);
        return true;
    }
    if (src is System.Collections.Generic.IEnumerable<Vector3> gen)
    {
        pts = new List<Vector3>();
        foreach (var v in gen) pts.Add(v);
        if (pts.Count > 0) return true;
        pts = null; return false;
    }
    if (src is System.Collections.IEnumerable any)
    {
        pts = new List<Vector3>();
        foreach (var v in any) if (v is Vector3 vv) pts.Add(vv);
        if (pts.Count > 0) return true;
        pts = null; return false;
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

    // Simple mesh -> point cloud in B-local. Optional stride downsampling. No RNG.
    static void MeshToPointCloudLocal(Transform root, List<Vector3> outLocal, int maxPoints = 0, bool includeSkinned = true)
    {
        outLocal.Clear();
        if (!root) return;

        var toRoot = root.worldToLocalMatrix;
        int budget = maxPoints > 0 ? Mathf.Max(1, maxPoints) : int.MaxValue;

        // MeshFilter vertices
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

        // SkinnedMeshRenderer baked vertices
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

        if (outLocal.Count == 0)
            Debug.LogWarning($"No readable vertices under '{root.name}'. Check MeshFilter.sharedMesh, SkinnedMeshRenderer, and Read/Write Enabled.");
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

    // Horn absolute orientation via Davenport q-method (power iteration)
    static class RigidFit
    {
        public static bool TryEstimate(List<Vector3> A, List<Vector3> B, out Quaternion R, out Vector3 t)
        {
            R = Quaternion.identity; t = Vector3.zero;
            int n = A.Count;
            if (n != B.Count || n < 3) return false;

            Vector3 ca = Vector3.zero, cb = Vector3.zero;
            for (int i = 0; i < n; i++) { ca += A[i]; cb += B[i]; }
            ca /= n; cb /= n;

            float Sxx=0, Sxy=0, Sxz=0, Syx=0, Syy=0, Syz=0, Szx=0, Szy=0, Szz=0;
            for (int i = 0; i < n; i++)
            {
                var ap = A[i] - ca; var bp = B[i] - cb;
                Sxx += ap.x*bp.x; Sxy += ap.x*bp.y; Sxz += ap.x*bp.z;
                Syx += ap.y*bp.x; Syy += ap.y*bp.y; Syz += ap.y*bp.z;
                Szx += ap.z*bp.x; Szy += ap.z*bp.y; Szz += ap.z*bp.z;
            }

            float tr = Sxx + Syy + Szz;
            float n00 = tr,  n01 = Syz - Szy, n02 = Szx - Sxz, n03 = Sxy - Syx;
            float n11 = Sxx - Syy - Szz, n12 = Sxy + Syx, n13 = Szx + Sxz;
            float n22 = -Sxx + Syy - Szz, n23 = Syz + Szy;
            float n33 = -Sxx - Syy + Szz;

            float q0=1,q1=0,q2=0,q3=0;
            for (int k = 0; k < 50; k++)
            {
                float r0 = n00*q0 + n01*q1 + n02*q2 + n03*q3;
                float r1 = n01*q0 + n11*q1 + n12*q2 + n13*q3;
                float r2 = n02*q0 + n12*q1 + n22*q2 + n23*q3;
                float r3 = n03*q0 + n13*q1 + n23*q2 + n33*q3;
                float norm = Mathf.Sqrt(r0*r0 + r1*r1 + r2*r2 + r3*r3);
                if (norm < 1e-9f) break;
                q0=r0/norm; q1=r1/norm; q2=r2/norm; q3=r3/norm;
            }

            var q = new Quaternion(q1,q2,q3,q0);
            q.Normalize();
            R = q;
            t = cb - (R * ca);
            return true;
        }
    }
}
