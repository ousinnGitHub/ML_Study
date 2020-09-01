#!/bin/bash
#log path is /var/log/openstack-kilo
#if commod execute sucessed,it will return 0 else return 1
# Copyright 2014 Intel Corporation, All Rights Reserved.
function log_info ()
{
if [  -d /home/wang/work/log  ]
then
    mkdir -p /home/wang/work/log 
fi

DATE_N=`date "+%Y-%m-%d %H:%M:%S"`
USER_N=`whoami`
echo "${DATE_N} ${USER_N} execute $0 [INFO] $@" >>/home/wang/work/log/fun2.log #执行成功日志打印路径

}

function log_error ()
{
DATE_N=`date "+%Y-%m-%d %H:%M:%S"`
USER_N=`whoami`
echo -e "error ${DATE_N} ${USER_N} execute $0 [ERROR] $@ "  >>/home/wang/work/log/fun2.log #执行失败日志打印路径

}

function fn_log ()  {
if [  $? -eq 0  ]
then
    log_info "$@ sucessed."
    echo -e "exec $@ sucessed. "
else
    log_error "$@ failed."
    echo -e "exec $@ failed. "
    exit 1
fi
}
trap 'fn_log "DO NOT SEND CTR + C WHEN EXECUTE SCRIPT !!!! "'  2
