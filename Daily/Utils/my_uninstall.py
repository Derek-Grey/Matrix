#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/2 18:24
# @Author  : Neo
# @File    : my_uninstall.py
# @Software: PyCharm
# 说明:

import os


def check_dependents(module_name):
    """
    查看一个模块的依赖
    :param module_name: 模块名
    :return: module_name 所依赖的模块的列表
    """
    with os.popen('pip show %s' % module_name) as p:
        dependents = p.readlines()
        if not len(dependents):
            return None
        dependents = dependents[-1]
        dependents = dependents.split(':')[-1].replace(' ', '').strip()
        if dependents:
            dependents = dependents.split(',')
        else:
            dependents = None
        return dependents


def remove(module_name):
    """
    递归地卸载一个模块
    :param module_name: 模块名称
    :return: None
    """
    dependents = check_dependents(module_name)
    if dependents:
        for package in dependents:
            remove(package)
    os.system('pip uninstall -y %s' % module_name)


if __name__ == '__main__':
    pkg_name = input('请输入要卸载的第三方模块包: ')
    remove(pkg_name)
