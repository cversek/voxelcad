import sys, logging, traceback
from inspect import currentframe, getframeinfo

def DEBUG_TAG(frame, msg = None):
    info = getframeinfo(frame)
    tag = f"*** DEBUG ***\n*** \tL#{info.lineno} in '{info.filename}'"
    if msg is not None:
        tag += f": {msg}"
    print('*'*40)
    print(tag);
    print('*'*40)


def DEBUG_PRINT_EXCEPTION():
    exc = traceback.format_exc()
    tag = f"*** EXCEPTION ***\n*** \n\t{exc}"
    print('*'*40)
    print(tag);
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