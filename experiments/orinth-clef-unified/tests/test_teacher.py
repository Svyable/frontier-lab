import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from orinth_clef.teacher import LocalTeacher, TeacherError
from tests.test_schema import REQUEST, RESPONSE


class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        body = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        assert self.path == "/v1/systemone"
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.last_request = body
        encoded = json.dumps(RESPONSE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(encoded)


class TeacherTests(unittest.TestCase):
    def test_reject_remote_host(self):
        with self.assertRaisesRegex(ValueError, "localhost"):
            LocalTeacher("https://example.com")
        with self.assertRaisesRegex(ValueError, "localhost"):
            LocalTeacher("http://0.0.0.0:8001")

    def test_http_roundtrip_and_no_truncation(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            client = LocalTeacher(f"http://127.0.0.1:{server.server_port}")
            self.assertEqual(client.health()["status"], "ok")
            self.assertEqual(client.predict(REQUEST)["answers"]["urgent"]["noul"], 0.8)
            self.assertIs(server.last_request["truncate"], False)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_unreachable_teacher_is_error(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        port = server.server_port
        server.server_close()
        with self.assertRaises(TeacherError):
            LocalTeacher(f"http://127.0.0.1:{port}", timeout=0.5).predict(REQUEST)


if __name__ == "__main__":
    unittest.main()
