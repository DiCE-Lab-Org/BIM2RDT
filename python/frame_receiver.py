# python/frame_receiver.py
import socket, threading, struct, queue
import numpy as np
import cv2

class FrameReceiver:
    def __init__(self, host="127.0.0.1", port=5577, max_q=2):
        self.addr = (host, port)
        self.q = queue.Queue(maxsize=max_q)
        self.running = False
        self.thread = None
        self.clients = []
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.listen(1)
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()
        print(f"[frames] listening on {self.addr[0]}:{self.addr[1]}")

    def _accept_loop(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self.lock:
                    self.clients.append(conn)
                threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()
                print("[frames] client connected")
            except OSError:
                break

    def _client_loop(self, conn):
        try:
            while self.running:
                # read 4-byte big-endian length
                hdr = self._recvall(conn, 4)
                if not hdr: break
                (length,) = struct.unpack(">I", hdr)
                buf = self._recvall(conn, length)
                if not buf: break
                # decode JPEG -> BGR
                arr = np.frombuffer(buf, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None: continue
                # drop if queue is full to keep realtime
                if not self.q.full():
                    self.q.put(frame)
                else:
                    _ = self.q.get_nowait()
                    self.q.put(frame)
        finally:
            try: conn.close() 
            except: pass
            with self.lock:
                if conn in self.clients: self.clients.remove(conn)
            print("[frames] client disconnected")

    def _recvall(self, conn, n):
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def get(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        try: self.sock.close() 
        except: pass
        with self.lock:
            for c in self.clients:
                try: c.close() 
                except: pass
            self.clients.clear()
