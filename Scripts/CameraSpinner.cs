using UnityEngine;

// Attach this to the Main Camera.
// Mode 1: constant spin (degPerSec).
// Mode 2: sweep back-and-forth between minYaw and maxYaw around the current yaw.

public class CameraSpinner : MonoBehaviour {
  public bool constantSpin = false;
  public float degPerSec = 15f;   // used if constantSpin = true

  public bool sweep = true;
  public float minYaw = -35f;     // degrees relative to start yaw
  public float maxYaw =  35f;     // degrees relative to start yaw
  public float sweepPeriod = 8f;  // seconds for a full left→right→left cycle

  float startYaw;

  void Start() {
    startYaw = transform.eulerAngles.y;
  }

  void Update() {
    if (constantSpin) {
      transform.Rotate(0f, degPerSec * Time.deltaTime, 0f, Space.World);
      return;
    }

    if (sweep) {
      // t goes 0→1→0 with a cosine; map to minYaw..maxYaw
      float t = 0.5f * (1 - Mathf.Cos(2 * Mathf.PI * (Time.time / Mathf.Max(0.01f, sweepPeriod))));
      float yaw = Mathf.Lerp(minYaw, maxYaw, t) + startYaw;
      var e = transform.eulerAngles;
      e.y = yaw;
      transform.eulerAngles = e;
    }
  }
}
