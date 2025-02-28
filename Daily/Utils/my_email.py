#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 19:04
# @Author  : Neo
# @File    : my_email.py
# @Software: PyCharm
# 说明:

from smtplib import SMTP_SSL, SMTPException
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from loguru import logger


# smtplib模块主要负责发送邮件：是一个发送邮件的动作，连接邮箱服务器，登录邮箱，发送邮件（有发件人，收信人，邮件内容）。
# email模块主要负责构造邮件：指的是邮箱页面显示的一些构造，如发件人，收件人，主题，正文，附件等。

def my_email(title: str = None, content: str = None):
    sender_ = 'z.yong.cool@163.com'  # 发件人邮箱
    receiver = ['yongzhu@treadlefountain.com', '18979277371@189.cn']  # 收件人邮箱
    # 初始化一个邮件主体
    msg = MIMEMultipart()
    msg["Subject"] = Header(title, 'utf-8')
    msg["From"] = sender_
    msg['To'] = ";".join(receiver)
    msg.attach(MIMEText(content, 'plain', 'utf-8'))

    host_server = 'smtp.163.com'
    pwd = 'LITRMQVMOIMPWCFI'
    try:
        smtp = SMTP_SSL(host_server)  # ssl登录
        # smtp.set_debuglevel(1)  # 0是关闭，1是开启debug
        smtp.ehlo(host_server)  # 跟服务器打招呼，告诉它我们准备连接，最好加上这行代码
        smtp.login(sender_, pwd)
        smtp.sendmail(sender_, receiver, msg.as_string())
        smtp.quit()
    except SMTPException:
        logger.warning(f'邮件发送失败！！！')


if __name__ == '__main__':
    pass
