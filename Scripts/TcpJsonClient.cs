using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable]
public class ImageSize { public int w; public int h; }

[Serializable]
public class Detection {
  public int id;
  public int[] bbox_xyxy;  // [x1,y1,x2,y2]
  public int[] bbox_wh;    // [w,h]
  public string label;
  public float score;
}

[Serializable]
public class FrameMsg {
  public int frame_id;
  public long ts_ms;
  public ImageSize image_size;
  public Detection[] detections;
}

public class TcpJsonClient : MonoBehaviour {
  public string Host = "127.0.0.1";
  public int Port = 5555;

  public static readonly ConcurrentQueue<FrameMsg> Inbox = new();
  Thread _thread;
  volatile bool _running;

  void Start() {
    _running = true;
    _thread = new Thread(Run) { IsBackground = true };
    _thread.Start();
  }

  void OnDestroy() {
    _running = false;
    try { _thread?.Join(200); } catch {}
  }

  void Run() {
    try {
      using var client = new TcpClient();
      client.Connect(Host, Port);
      using var stream = client.GetStream();
      using var reader = new StreamReader(stream, Encoding.UTF8);

      while (_running) {
        var line = reader.ReadLine();
        if (line == null) break;
        var msg = JsonUtility.FromJson<FrameMsg>(line);
        Inbox.Enqueue(msg);
      }
    } catch (Exception e) {
      Debug.LogWarning($"[TcpJsonClient] {e.Message}");
    }
  }
}
