using UnityEngine;
public class Rotator : MonoBehaviour {
  public Vector3 SpeedEuler = new Vector3(0, 45, 0); // deg/sec
  void Update() { transform.Rotate(SpeedEuler * Time.deltaTime); }
}
