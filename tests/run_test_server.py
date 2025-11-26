import http.server
import socketserver
import json
import urllib.parse

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):

    def do_POST(self):
        if self.path == "/submit-demo":
            print("🌟 RECEIVED /submit-demo POST")
            length = int(self.headers.get('Content-Length'))
            body = self.rfile.read(length)
            data = json.loads(body.decode('utf-8'))

            # If this submission comes from multi_step_1 → send Step 2
            if "multi_step_1" in data.get("url", ""):
                resp = {
                    "correct": True,
                    "url": f"http://localhost:{PORT}/tests/multi_step_2.html",
                    "reason": None
                }
            # If this is multi_step_2 → final step, no next URL
            elif "multi_step_2" in data.get("url", ""):
                resp = {
                    "correct": True,
                    "url": None,
                    "reason": None
                }
            # Fallback
            else:
                resp = {"correct": True}

            encoded = json.dumps(resp).encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return

        return super().do_POST()

print(f"Serving tests at http://localhost:{PORT}/tests/multi_step_1.html")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()

