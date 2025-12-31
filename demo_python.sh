#!/bin/bash
# 替代原始demo.sh脚本，使用Python而不是MATLAB

echo "Please wait.."
set -e
today="$(date '+%d_%m_%Y_%T')"

pathInput=${1%/}
pathOutput=${2%/}
modelfile="$3"

path_downscaled="$pathOutput/temp"

if [ ! -d $path_downscaled ]; then
  mkdir $path_downscaled
fi

# 创建日志文件
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>logs/log_$today.out 2>&1

# 缩小图像尺寸（使用Python替代MATLAB）
python3 resize_im.py "$pathInput" "$path_downscaled"

# 去雾处理
sh convertHazy2GT.sh "$path_downscaled" "$modelfile"

# 放大图像尺寸（使用Python替代MATLAB）
python3 laplacian.py "$path_downscaled" "$pathInput" "$pathOutput"

# 清理临时目录
if [ -d $path_downscaled ]; then
  rm -rf $path_downscaled
fi

echo "处理完成！"
