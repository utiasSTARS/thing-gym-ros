""" Various utilities related to running a gui with a thing env. """
from threading import Lock


class ObjLockWrapper:
    def __init__(self, obj):
        self.__wrapped_obj = obj
        self.lock = Lock()

    def __enter__(self):
        self.lock.acquire()
        return self.__wrapped_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()