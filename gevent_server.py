from gevent.wsgi import WSGIServer
from plotly_dash_test import server

http_server = WSGIServer(('0.0.0.0', 5000), server)
http_server.serve_forever()
