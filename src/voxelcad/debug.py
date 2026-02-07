import sys, logging, traceback, os, psutil
from inspect import currentframe, getframeinfo

# Optional super_utils integration for structured profiling
try:
    from super_utils import TIMING_START, TIMING_END, TIMING_EXPORT_JSON
    from super_utils import MEMORY_SNAPSHOT as _SU_MEMORY_SNAPSHOT
    HAS_SUPER_UTILS = True
except ImportError:
    HAS_SUPER_UTILS = False
    def TIMING_START(label): pass
    def TIMING_END(label): pass
    def TIMING_EXPORT_JSON(output_path, **kwargs): return output_path
    def _SU_MEMORY_SNAPSHOT(): return 0


def create_logger(name, level=logging.DEBUG):
    #REF: https://docs.python.org/3/howto/logging.html#advanced-logging-tutorial
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger

def DEBUG_TAG(frame, msg = None):
    info = getframeinfo(frame)
    tag = f"*** DEBUG ***\n*** \tL#{info.lineno} in '{info.filename}'"
    if msg is not None:
        tag += f": {msg}"
    print('*'*40)
    print(tag)
    print('*'*40)


def DEBUG_PRINT_EXCEPTION():
    exc = traceback.format_exc()
    tag = f"*** EXCEPTION ***\n*** \n\t{exc}"
    print('*'*40)
    print(tag)
    print('*'*40)


def DEBUG_EMBED(local_ns, global_ns = None, exit = False):
    import IPython
    #merge global and local namespaces
    if global_ns is None:
        global_ns = {}
    user_ns = global_ns
    user_ns.update(local_ns)
    IPython.embed(user_ns=user_ns)
    if exit:
        sys.exit()

def MEMORY_USAGE(offset=0.0,show=False):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss
    drss = rss - offset
    if show:
        if offset == 0:
            print(f"TOTAL MEMORY USED: {rss/2**30:0.2} GB")  
        else:
            print(f"DELTA MEMORY USED: {drss/2**30:0.2} GB")
    return drss
    