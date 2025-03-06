from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import os

PORT = 9875
DIRECTORIES = ["png", "html"]

class RestrictedHTTPRequestHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Get the absolute path of the requested file
        path = super().translate_path(path)
        # Restrict access to files outside the specified directories
        allowed = False
        for directory in DIRECTORIES:
            if path.startswith(os.path.abspath(directory)):
                allowed = True
                break

        if not allowed:
            self.send_error(403, "Forbidden")
            return None
        return path

Handler = RestrictedHTTPRequestHandler

httpd = TCPServer(("", PORT), Handler)

print(f"HTTP server serving at port {PORT} and restricting access to {', '.join(DIRECTORIES)} directories")
httpd.serve_forever()
