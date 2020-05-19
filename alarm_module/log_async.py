import logging
import logging.config
from logging.handlers import QueueHandler
import threading
import socketserver
import struct
import pickle
import sys
from alarm_module.log import setup_logging


class LogRecordStreamHandler(socketserver.StreamRequestHandler):

    def handle(self):
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):

        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger(name)
        logger.handle(record)


class LogAsyncService(socketserver.ThreadingTCPServer):
    """
    logger tcp服务
    """

    allow_reuse_address = True

    def __init__(self, host='localhost',
                 port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = None
        self.lp = threading.Thread(target=self.serve_until_stopped)

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()], [], [], self.timeout)
            if rd:
                self.handle_request()
            abort = self.abort

    def start(self):
        self.lp.start()

    def stop(self):
        self.abort = 1

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class Logger:
    def __init__(self):
        setup_logging()
        self.critical = 50
        self.error = 40
        self.warning = 30
        self.info = 20
        self.debug = 10
        self.notset = 0
        socketHandler = logging.handlers.SocketHandler('localhost', logging.handlers.DEFAULT_TCP_LOGGING_PORT)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(socketHandler)

    def log(self, name, lv, msg, *args, **kwargs):
        logger =logging.getLogger('async.' + name)
        extra = self.__findfunc()
        t = threading.Thread(target=logger.log, args=(lv, msg, *args, ), kwargs={**kwargs, 'extra': extra})
        t.start()

    def __findfunc(self):
        """
        找出日志文件真正的函数栈
        :return:
        """
        currentframe = lambda: sys._getframe(2)
        f = currentframe()
        f = f.f_back
        return {'file_path': f.f_code.co_filename, 'func_name': f.f_code.co_name, 'row': f.f_lineno}
        # try:
        #     raise Exception
        # except:
        #     f = sys.exc_info()[2].tb_frame.f_back
        #     # a = f.f_code
        #     # print(a)
        #     return {'file_path': f.f_code.co_filename, 'func_name': f.f_code.co_name, 'row': f.f_lineno}


logger_async = Logger()


