import logging
import logging.config
import logging.handlers
from multiprocessing import Process, Queue
import random
import time
from alarm_module import setup_logging
from alarm_module import logger_async


def worker_process(q):
    qh = logging.handlers.QueueHandler(q)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(qh)
    levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR,
              logging.CRITICAL]
    loggers = ['foo', 'foo.bar', 'foo.bar.baz',
               'spam', 'spam.ham', 'spam.ham.eggs']
    for i in range(1):
        lvl = random.choice(levels)
        logger = logging.getLogger(random.choice(loggers))
        logger.log(lvl, 'Message no. %d', i)


def worker_process_socket():

    levels = [logging.CRITICAL]
    loggers = ['foo', 'foo.bar', 'foo.bar.baz',
               'spam', 'spam.ham', 'spam.ham.eggs']
    for i in range(1):
        lvl = random.choice(levels)
        ts = time.time()
        try:
            raise Exception('test')
        except:
            logger_async.log(random.choice(loggers), lvl, 'just test', exc_info=True)
        tp = time.time()
        print(tp - ts)


if __name__ == '__main__':
    from alarm_module import LogAsyncService
    workers = []
    with LogAsyncService() as lss:
        for i in range(1):
            wp = Process(target=worker_process_socket, name='worker %d' % (i + 1))
            workers.append(wp)
            wp.start()
        setup_logging()

        for wp in workers:
            wp.join()

