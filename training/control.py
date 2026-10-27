"""Cooperative Ctrl+C/SIGTERM handling; committed units are never rolled back."""
import signal
import threading


class WorkInterrupted(Exception):
    pass


class StopController:
    def __init__(self):
        self.event=threading.Event(); self.handlers={}; self.signal_number=None

    def is_set(self): return self.event.is_set()

    def request_stop(self,signum=None,frame=None):
        self.signal_number=signum; self.event.set()

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT,signal.SIGTERM):
                self.handlers[sig]=signal.getsignal(sig)
                signal.signal(sig,self.request_stop)
        return self

    def __exit__(self,*args):
        for sig,handler in self.handlers.items(): signal.signal(sig,handler)


def check_stop(event):
    if event is not None and event.is_set(): raise WorkInterrupted('已请求停止，丢弃未完成的工作单元')
