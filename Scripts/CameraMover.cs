using UnityEngine;
public class CameraMover : MonoBehaviour {
  public Vector3 p0 = new Vector3(0, 1.6f, -2.5f);
  public Vector3 p1 = new Vector3(0, 1.6f, 1.5f);
  public float period = 12f; // seconds
  void Update() {
    float t = 0.5f * (1 - Mathf.Cos(2 * Mathf.PI * (Time.time % period) / period));
    transform.position = Vector3.Lerp(p0, p1, t);
    transform.LookAt(new Vector3(0, 1.0f, 9)); // look toward the work zone
  }
}
