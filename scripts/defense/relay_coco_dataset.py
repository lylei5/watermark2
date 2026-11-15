"""Loopback-only streaming relay for the public COCO archive; no local files.

Use an SSH reverse forward bound to loopback on server249. Only bounded byte
ranges of this fixed public object are served; this is not a general proxy.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import re
import threading
import urllib.request

UPSTREAM = 'https://s3.amazonaws.com/images.cocodataset.org/zips/test2017.zip'
SIZE = 6646970404
CHUNK = 8 * 1024 * 1024
SLOTS = threading.BoundedSemaphore(4)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/test2017.zip':
            self.send_error(404)
            return
        match = re.fullmatch(r'bytes=(\d+)-(\d+)', self.headers.get('Range', ''))
        if not match:
            self.send_error(400, 'An explicit bounded byte range is required')
            return
        start, end = map(int, match.groups())
        if not 0 <= start <= end < SIZE or end - start + 1 > CHUNK:
            self.send_error(416)
            return
        with SLOTS:
            try:
                # Disambiguate byte-range responses in intermediary caches.
                request = urllib.request.Request(UPSTREAM + f'?wmrange={start}-{end}',
                    headers={'Range': f'bytes={start}-{end}', 'Cache-Control': 'no-cache'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                        raise ValueError('Unexpected upstream byte range')
                    self.send_response(206)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Length', str(end-start+1))
                    self.send_header('Content-Range', response.headers['Content-Range'])
                    self.end_headers()
                    remaining = end-start+1
                    while remaining:
                        block = response.read(min(1024*1024, remaining))
                        if not block:
                            raise EOFError('Short upstream response')
                        self.wfile.write(block)
                        remaining -= len(block)
            except Exception as exc:
                self.log_error('%s', repr(exc))
                self.close_connection = True

    def log_message(self, format, *args):
        # Successful range transfers are recorded by the receiver on249.
        if args and str(args[1:2]) == "('206',)":
            return
        if 'code' in format or 'Unexpected' in format or 'Error' in str(args):
            super().log_message(format, *args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18091)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'COCO public archive relay listening on loopback port {args.port}; no local persistence', flush=True)
    server.serve_forever()
