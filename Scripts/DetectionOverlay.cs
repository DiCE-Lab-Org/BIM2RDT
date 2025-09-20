// This file must be placed in [Unity Project]/Assets/Scripts for detection to work
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DetectionOverlay : MonoBehaviour {
  // ←← IMPORTANT: drag the UI panel (RawImage) that shows MainCam_RT into this slot in the Inspector.
  public RawImage TargetImage;
  public System.Collections.Generic.IReadOnlyList<Detection> LastDetections => _last;
  public int ImageWidth  => _imgW;
  public int ImageHeight => _imgH;
  // internal state
  List<Detection> _last = new();
  int _imgW = 640, _imgH = 640;
  int _framesSeen;

  // Tidy-up knobs (optional)
  public int MaxBoxes = 50;         // cap per-frame boxes so overlay stays readable
  public float MinBoxSize = 6f;     // skip tiny boxes (in source pixels)

  void Update() {
    // Drain any incoming frame messages from the TCP queue
    while (TcpJsonClient.Inbox.TryDequeue(out var f)) {
      _imgW = Mathf.Max(1, f.image_size.w);
      _imgH = Mathf.Max(1, f.image_size.h);

      _last = new List<Detection>(f.detections ?? new Detection[0]);

      // (Optional) skip tiny boxes
      _last.RemoveAll(d => (d.bbox_xyxy[2] - d.bbox_xyxy[0]) < MinBoxSize ||
                           (d.bbox_xyxy[3] - d.bbox_xyxy[1]) < MinBoxSize);

      // (Optional) cap count
      if (_last.Count > MaxBoxes) {
        _last.Sort((a,b) => b.score.CompareTo(a.score));
        _last = _last.GetRange(0, MaxBoxes);
      }

      _framesSeen++;
    }
  }

  void OnGUI() {
    // Small HUD to confirm we’re receiving data
    GUI.Label(new Rect(8, 8, 220, 20), $"msgs:{_framesSeen} dets:{_last.Count}");

    if (TargetImage == null || _last == null || _last.Count == 0) return;

    // 1) Find the on-screen rectangle of the TargetImage panel (the “TV screen”)
    Rect imgRect = GetScreenRect(TargetImage);

    // 2) Draw each box, mapping from image pixels → panel pixels
    foreach (var d in _last) {
        // Convert from image pixels → normalized [0..1]
        float nx1 = (float)d.bbox_xyxy[0] / _imgW;
        float ny1 = (float)d.bbox_xyxy[1] / _imgH;  // top edge (from top)
        float nx2 = (float)d.bbox_xyxy[2] / _imgW;
        float ny2 = (float)d.bbox_xyxy[3] / _imgH;  // bottom edge (from top)

        // Map to GUI pixels (GUI origin is top-left; our imgRect.y is top edge)
        float rx = imgRect.x + nx1 * imgRect.width;
        float ryTop = imgRect.y + ny1 * imgRect.height;      // ← no flip
        float ryBot = imgRect.y + ny2 * imgRect.height;
        float rw = (nx2 - nx1) * imgRect.width;
        float rh = ryBot - ryTop;

        var r = new Rect(rx, ryTop, rw, rh);


      DrawRect(r, 2);

      // Label background for readability
      var label = $"{d.label} {d.score:0.00}";
      var labRect = new Rect(r.x, r.y - 20, Mathf.Min(260, r.width), 20);
      GUI.color = new Color(0, 0.6f, 0, 0.75f); // translucent green bg
      GUI.DrawTexture(labRect, Texture2D.whiteTexture);
      GUI.color = Color.white;
      GUI.Label(labRect, label);
    }
  }

  // Helper: convert a RawImage’s rectangle into screen-pixel space
  Rect GetScreenRect(RawImage img) {
    var rt = img.rectTransform;
    Vector3[] corners = new Vector3[4];
    rt.GetWorldCorners(corners);
    // corners: 0=bottom-left, 1=top-left, 2=top-right, 3=bottom-right (in world)
    Vector2 bl = RectTransformUtility.WorldToScreenPoint(null, corners[0]);
    Vector2 tl = RectTransformUtility.WorldToScreenPoint(null, corners[1]);
    Vector2 tr = RectTransformUtility.WorldToScreenPoint(null, corners[2]);
    // GUI space has origin at top-left of the monitor
    float x = tl.x;
    float y = Screen.height - tl.y;
    float w = tr.x - tl.x;
    float h = tl.y - bl.y;
    return new Rect(x, y, w, h);
  }

  // Draws a rectangle outline
  void DrawRect(Rect r, int thickness) {
    var t = Texture2D.whiteTexture;
    GUI.color = Color.green;
    GUI.DrawTexture(new Rect(r.x, r.y, r.width, thickness), t);                  // top
    GUI.DrawTexture(new Rect(r.x, r.yMax - thickness, r.width, thickness), t);  // bottom
    GUI.DrawTexture(new Rect(r.x, r.y, thickness, r.height), t);                 // left
    GUI.DrawTexture(new Rect(r.xMax - thickness, r.y, thickness, r.height), t); // right
    GUI.color = Color.white;
  }
}
