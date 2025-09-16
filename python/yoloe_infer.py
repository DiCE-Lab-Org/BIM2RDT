# python/yoloe_infer.py
import argparse
import time
from pathlib import Path
import yaml
import cv2
import numpy as np
import torch
from ultralytics import YOLOE   # <-- YOLOE class in Ultralytics
from tcp_server import NDJSONServer
from frame_receiver import FrameReceiver

def load_yaml(p: Path):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="configs/model.yaml")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    return ap

def main():
    args = build_argparser().parse_args()
    cfg = load_yaml(Path(args.model))
    prompts_cfg = load_yaml(Path(args.prompts)) if Path(args.prompts).exists() else {}

    weights = cfg.get("weights", "yoloe-11l-seg.pt")
    device = cfg.get("device", "cuda:0")
    imgsz  = int(cfg.get("imgsz", 960))
    conf   = float(cfg.get("conf", 0.25))
    iou    = float(cfg.get("iou", 0.50))
    prompt_mode = cfg.get("prompt_mode", "text")
    # source = cfg.get("source", 0) # Using Unity as input source, do not need camera index
    host   = cfg.get("tcp_host", "127.0.0.1")
    port   = int(cfg.get("tcp_port", 5555))
    # send_masks = bool(cfg.get("send_masks", False))  # Not using masks

    # --- TCP server
    server = NDJSONServer(host, port)
    server.start()

    # --- Start Unity frame receiver
    fr_host = "127.0.0.1"
    fr_port = 5577
    receiver = FrameReceiver(fr_host, fr_port)
    receiver.start()

    # --- Model
    model = YOLOE(weights)
    if device != "cpu" and torch.cuda.is_available():
        model.to(device)
    # fp16 inferences if GPU is available
    half = (device != "cpu" and torch.cuda.is_available())

    # --- Prompt setup
    # Text-prompt mode: use 11%-seg.pt weights and set prompt_mode: "text" in model.yaml
    # Prompt-free mode: use 11%-seg-pf.pt weights and set prompt_mode: "prompt_free" in model.yaml
    names = []
    if prompt_mode == "text":
        names = prompts_cfg.get("text", [])
        if not names:
            print("[warn] prompt_mode=text but prompts.yaml is empty; add labels under 'text:'")
        else:
            # Pre-encode and set classes (works for inference and for export baking)
            # Docs show set_classes(..., get_text_pe(...)) usage.
            pe = model.get_text_pe(names)   # text prompt embeddings
            model.set_classes(names, pe)    # set class list + embeddings
            print(f"[yoloe] text prompts set: {names}")
    else:
        print("[yoloe] using prompt-free weights; ignoring prompts.yaml")

    # --- Video source: Unity Main Camera (must set Unity Game resolution to 640x640)
    frame_id = 0
    try:
        while True:
            frame = receiver.get(timeout=5.0)
            if frame is None:
                # no frame received recently; keep waiting
                continue

            # h, w = frame.shape[:2]
            # imsz = min(h, w)  # 640 as per model.yaml

            results = model.predict(
                source=frame,
                imgsz=imgsz,             # match incoming frame size (expects resolution to be 640x640 from Unity)
                conf=conf,
                iou=iou,
                verbose=False,
                half=(device != "cpu" and torch.cuda.is_available())
)
            # Logging...
            h, w = frame.shape[:2]
            if frame_id % 30 == 0:
                print(f"[infer] frame {frame_id} {w}x{h} - dets so far: ", end="")

            dets = []
            if len(results):
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    clsi = r.boxes.cls.cpu().numpy().astype(int)
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                        label_idx = int(clsi[i])
                        label = r.names.get(label_idx, str(label_idx))
                        dets.append({
                            "id": -1,
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "bbox_wh": [x2 - x1, y2 - y1],
                            "label": label,
                            "score": float(scores[i]),
                            "meta": {"source": "YOLOE"}
                        })
            
            if len(dets) > 50:
                dets.sort(key=lambda a: a["score"], reverse=True)
                dets = dets[:50]
            
            # ...logging
            if frame_id % 30 == 0:
                print(len(dets))
            
            if frame_id % 45 == 0:
                dets.append({
                    "id": -1,
                    "bbox_xyxy": [10, 10, w-10, h-10],   # almost full-frame border
                    "bbox_wh": [w-20, h-20],
                    "label": "CAL",
                    "score": 1.0,
                    "meta": {"source": "calib"}
                })

            server.send({
                "frame_id": frame_id,
                "ts_ms": int(time.time() * 1000),
                "image_size": {"w": w, "h": h},
                "detections": dets
            })
            frame_id += 1
    finally:
        receiver.stop()
        server.stop()

if __name__ == "__main__":
    main()
