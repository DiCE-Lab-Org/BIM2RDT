using System;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class FrameSenderURP : MonoBehaviour {
  [Header("Network")]
  public string Host = "127.0.0.1";
  public int Port = 5577;

  [Header("Capture (assign the same RT set on the Camera)")]
  public RenderTexture CaptureRT;
  [Range(1,100)] public int JpegQuality = 80;
  public int SendEveryNFrames = 1;

  TcpClient _client;
  NetworkStream _stream;
  byte[] _lenBuf = new byte[4];
  Texture2D _tex;
  int _frameCount;

  void Start() {
    if (CaptureRT == null) {
      var cam = GetComponent<Camera>();
      CaptureRT = cam.targetTexture;
      if (CaptureRT == null) Debug.LogError("[FrameSenderURP] CaptureRT not set and Camera.targetTexture is null");
    }
    new Thread(ConnectThread) { IsBackground = true }.Start();
  }

  void ConnectThread() {
    while (true) {
      try {
        _client = new TcpClient();
        _client.NoDelay = true;
        _client.Connect(Host, Port);
        _stream = _client.GetStream();
        Debug.Log("[FrameSenderURP] Connected " + Host + ":" + Port);
        return;
      } catch (Exception e) {
        Debug.LogWarning($"[FrameSenderURP] Connect failed: {e.Message}. Retry in 1s.");
        Thread.Sleep(1000);
      }
    }
  }

  void LateUpdate() {
    if (_stream == null || !_stream.CanWrite || CaptureRT == null) return;
    _frameCount++;
    if ((_frameCount % Mathf.Max(1, SendEveryNFrames)) != 0) return;

    // Ensure CPU texture matches RT size
    if (_tex == null || _tex.width != CaptureRT.width || _tex.height != CaptureRT.height) {
      if (_tex != null) Destroy(_tex);
      _tex = new Texture2D(CaptureRT.width, CaptureRT.height, TextureFormat.RGB24, false);
    }

    // Readback RT → Texture2D
    var prev = RenderTexture.active;
    RenderTexture.active = CaptureRT;
    _tex.ReadPixels(new Rect(0, 0, CaptureRT.width, CaptureRT.height), 0, 0, false);
    _tex.Apply(false, false);
    RenderTexture.active = prev;

    // Encode + send (4-byte big-endian length then JPEG)
    byte[] jpg = _tex.EncodeToJPG(JpegQuality);
    int len = jpg.Length;
    _lenBuf[0] = (byte)((len >> 24) & 0xFF);
    _lenBuf[1] = (byte)((len >> 16) & 0xFF);
    _lenBuf[2] = (byte)((len >> 8) & 0xFF);
    _lenBuf[3] = (byte)(len & 0xFF);

    try {
      _stream.Write(_lenBuf, 0, 4);
      _stream.Write(jpg, 0, len);
    } catch (Exception e) {
      Debug.LogWarning($"[FrameSenderURP] Write failed: {e.Message}");
      try { _stream?.Close(); _client?.Close(); } catch {}
      _stream = null; _client = null;
      new Thread(ConnectThread) { IsBackground = true }.Start();
    }
  }

  void OnDestroy() {
    try { _stream?.Close(); _client?.Close(); } catch {}
    if (_tex) Destroy(_tex);
  }
}
