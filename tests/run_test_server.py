import http.server
import socketserver

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    pass

print(f"Serving at http://localhost:{PORT}/tests/demo.html")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
