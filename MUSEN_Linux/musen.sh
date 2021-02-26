#!/bin/sh
appname=`basename $0 | sed s,\.sh$,,`

dirname=`dirname $0`
tmp="${dirname#?}"

if [ "${dirname%$tmp}" != "/" ]; then
    dirname=$PWD/$dirname
fi
export LD_LIBRARY_PATH="$dirname"/lib/:"$dirname":$LD_LIBRARY_PATH
export QT_PLUGIN_PATH="$dirname"/plugins/:$QT_PLUGIN_PATH
export QTDIR="$dirname"
export QT_QPA_PLATFORM_PLUGIN_PATH="$dirname"/plugins/platforms:$QT_QPA_PLATFORM_PLUGIN_PATH
"$dirname/bin/$appname" "$@"
