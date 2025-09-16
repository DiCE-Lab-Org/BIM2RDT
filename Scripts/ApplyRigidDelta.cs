using UnityEngine;

[DisallowMultipleComponent]
[ExecuteAlways]
public class ApplyRigidDelta : MonoBehaviour
{
    [Header("Target")]
    public Transform target;

    [Header("World-space delta (from: [ICP] ΔB world)")]
    public Vector3 t_world = Vector3.zero;
    public Quaternion R_world = Quaternion.identity;

    [Header("A-local delta (from: [ICP] B→A local)")]
    public Transform aRoot;           // mesh A root
    public Vector3 t_A = Vector3.zero;
    public Quaternion R_A = Quaternion.identity;

    [Header("Control")]
    public KeyCode applyWorldKey = KeyCode.P;
    public KeyCode applyALocalKey = KeyCode.O;
    public bool logAfterApply = true;

    void Reset() { target = transform; }
    void OnValidate() { R_world = Normalize(R_world); R_A = Normalize(R_A); }
    void Update()
    {
        if (!Application.isPlaying) return;
        if (applyWorldKey != KeyCode.None && Input.GetKeyDown(applyWorldKey)) ApplyWorld();
        if (applyALocalKey != KeyCode.None && Input.GetKeyDown(applyALocalKey)) ApplyALocal();
    }

    [ContextMenu("Apply World Δ")]
    public void ApplyWorld()
    {
        if (!target) target = transform;
        var p = target.position;
        var q = target.rotation;
        target.SetPositionAndRotation(R_world * p + t_world, R_world * q);
        if (logAfterApply) Debug.Log($"[ApplyΔ] world  t={t_world}  Rquat={R_world}  Reuler={R_world.eulerAngles}");
    }

    [ContextMenu("Apply World Δ (Inverse)")]
    public void ApplyWorldInverse()
    {
        if (!target) target = transform;
        var Rinv = Quaternion.Inverse(R_world);
        var p = target.position;
        var q = target.rotation;
        target.SetPositionAndRotation(Rinv * (p - t_world), Rinv * q);
        if (logAfterApply) Debug.Log("[ApplyΔ] world inverse");
    }

    [ContextMenu("Apply A-local Δ")]
    public void ApplyALocal()
    {
        if (!target || !aRoot) { Debug.LogWarning("Set target and aRoot."); return; }
        var pW = target.position;
        var qW = target.rotation;

        // convert target pos into A local, apply local delta, convert back
        var pA = aRoot.InverseTransformPoint(pW);
        var pA2 = R_A * pA + t_A;
        var pW2 = aRoot.TransformPoint(pA2);

        // convert local rotation delta into world
        var Rw = aRoot.rotation * R_A * Quaternion.Inverse(aRoot.rotation);

        target.SetPositionAndRotation(pW2, Rw * qW);
        if (logAfterApply) Debug.Log($"[ApplyΔ] A-local  tA={t_A}  R_Aquat={R_A}  R_Aeuler={R_A.eulerAngles}");
    }

    [ContextMenu("Apply A-local Δ (Inverse)")]
    public void ApplyALocalInverse()
    {
        if (!target || !aRoot) { Debug.LogWarning("Set target and aRoot."); return; }
        var pW = target.position;
        var qW = target.rotation;

        var RinvA = Quaternion.Inverse(R_A);
        var pA = aRoot.InverseTransformPoint(pW);
        var pA2 = RinvA * (pA - t_A);
        var pW2 = aRoot.TransformPoint(pA2);

        var Rw = aRoot.rotation * RinvA * Quaternion.Inverse(aRoot.rotation);

        target.SetPositionAndRotation(pW2, Rw * qW);
        if (logAfterApply) Debug.Log("[ApplyΔ] A-local inverse");
    }
    public static void MoveAtoB(Transform A, Transform B, Quaternion R_BA, Vector3 t_BA)
    {
        A.SetPositionAndRotation(B.TransformPoint(t_BA), B.rotation * R_BA);
    }
    static Quaternion Normalize(Quaternion q)
    {
        float n = Mathf.Sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
        if (n < 1e-12f) return Quaternion.identity;
        return new Quaternion(q.x/n, q.y/n, q.z/n, q.w/n);
    }
}
