using System.Collections.Generic;
using UnityEngine;

public class MeshSurfaceHitsProvider : MonoBehaviour
{
    [Header("Target to sample")]
    public Transform targetRoot;
    public int sampleCount = 2000;
    public bool includeSkinned = true;
    public int randomSeed = 1234;

    // ICPAligner expects this exact property name and type
    public IReadOnlyList<Vector3> LastWorldHits => _hits;

    readonly List<Vector3> _hits = new();

    [ContextMenu("Rebuild")]
    public void Rebuild()
    {
        _hits.Clear();
        if (!targetRoot) return;

        var tris = new List<Tri>(1024);

        // MeshFilter
        foreach (var f in targetRoot.GetComponentsInChildren<MeshFilter>(true))
        {
            if (!f || !f.sharedMesh) continue;
            AddMeshTris(tris, f.sharedMesh, f.transform.localToWorldMatrix);
        }

        // Skinned
        if (includeSkinned)
        {
            foreach (var smr in targetRoot.GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                if (!smr || !smr.sharedMesh) continue;
                var baked = new Mesh();
                smr.BakeMesh(baked);
                AddMeshTris(tris, baked, smr.transform.localToWorldMatrix);
#if UNITY_EDITOR
                if (!Application.isPlaying) DestroyImmediate(baked);
                else Destroy(baked);
#else
                Destroy(baked);
#endif
            }
        }

        if (tris.Count == 0) return;
        SampleOnTriangles(tris, sampleCount, _hits, randomSeed);
    }

    void OnValidate() { if (sampleCount < 6) sampleCount = 6; }
    void LateUpdate() { /* call when geometry moves (skinned); or call Rebuild() manually */ }

    struct Tri { public Vector3 p0,p1,p2; public float area; public float cdf; }

    static void AddMeshTris(List<Tri> outTris, Mesh mesh, Matrix4x4 toWorld)
    {
        var v = mesh.vertices;
        var t = mesh.triangles;
        for (int i = 0; i < t.Length; i += 3)
        {
            var p0 = toWorld.MultiplyPoint3x4(v[t[i]]);
            var p1 = toWorld.MultiplyPoint3x4(v[t[i+1]]);
            var p2 = toWorld.MultiplyPoint3x4(v[t[i+2]]);
            float area = 0.5f * Vector3.Cross(p1 - p0, p2 - p0).magnitude;
            if (area <= 1e-9f) continue;
            outTris.Add(new Tri { p0 = p0, p1 = p1, p2 = p2, area = area });
        }
    }

    static void SampleOnTriangles(List<Tri> tris, int n, List<Vector3> outPts, int seed)
    {
        // build CDF by area
        float sum = 0f;
        for (int i = 0; i < tris.Count; i++) { sum += tris[i].area; var t = tris[i]; t.cdf = sum; tris[i] = t; }
        var rnd = new System.Random(seed);

        for (int k = 0; k < n; k++)
        {
            // pick triangle by area
            float r = (float)(rnd.NextDouble() * sum);
            int lo = 0, hi = tris.Count - 1, mid = 0;
            while (lo <= hi)
            {
                mid = (lo + hi) >> 1;
                if (r <= tris[mid].cdf) hi = mid - 1; else lo = mid + 1;
            }
            var tri = tris[Mathf.Clamp(lo, 0, tris.Count - 1)];

            // uniform barycentric sample
            float r1 = (float)rnd.NextDouble();
            float r2 = (float)rnd.NextDouble();
            float s1 = Mathf.Sqrt(r1);
            float w0 = 1f - s1;
            float w1 = s1 * (1f - r2);
            float w2 = s1 * r2;

            outPts.Add(w0 * tri.p0 + w1 * tri.p1 + w2 * tri.p2);
        }
    }
}
