# -*- coding: utf-8 -*-
"""
@author: Neo
@software: PyCharm
@file: MyErrores.py
@time: 2023-07-27 10:25
说明:
"""

import multiprocessing
import traceback
import subprocess
import time
import os


class ParamsError(Exception):
    pass


def throw_exception(name):
    print('子进程%s发生异常,进程号为%s' % (name, os.getpid()))
    cmd = 'taskkill /im ' + str(os.getpid()) + ' /F'
    res = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    print(res.stdout.read())
    print(res.stderr.read())
    time.sleep(2)


def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)


class LogExceptions(object):
    def __init__(self, func):
        self.__callable = func
        return

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result
