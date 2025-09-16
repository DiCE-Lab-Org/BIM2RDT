#
# Small broadcast server that is imported in yoloe_infer.py to send obj to Unity
#

import socket
import threading
import orjson

class NDJSONServer:
    def __init__(self, host="127.0.0.1", port=5555):
        self.addr = (host, port)
        self.clients = []
        self.lock = threading.Lock()
        self.alive = False

    def start(self):
        self.alive = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.listen(5)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"[tcp] listening on {self.addr[0]}:{self.addr[1]}")

    def _accept_loop(self):
        while self.alive:
            try:
                conn, _ = self.sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self.lock:
                    self.clients.append(conn)
                print("[tcp] client connected")
            except OSError:
                break

    def send(self, obj: dict):
        data = orjson.dumps(obj) + b"\n"
        drop = []
        with self.lock:
            for c in self.clients:
                try:
                    c.sendall(data)
                except OSError:
                    drop.append(c)
            for d in drop:
                try:
                    d.close()
                except Exception:
                    pass
                self.clients.remove(d)

    def stop(self):
        self.alive = False
        try:
            self.sock.close()
        except Exception:
            pass
        with self.lock:
            for c in self.clients:
                try:
                    c.close()
                except Exception:
                    pass
            self.clients.clear()
