#!/usr/bin/python
#coding: utf-8
from gevent.pywsgi import WSGIServer
from server import app

http_server = WSGIServer(('0.0.0.0', 8888), app)
http_server.serve_forever()