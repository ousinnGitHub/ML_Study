#!/bin/bash
source /home/wang/work/printlog.sh

echo 'exec log_info printlog'
log_info

psql -U postgres -d mydb -c "select mycount1()"
echo 'call plsql'

echo 'exec log_error printlog'
log_error
echo 'exec over'
