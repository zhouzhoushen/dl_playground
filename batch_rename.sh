#!/bin/bash

# 项目目录路径
PROJECT_DIR="."

# 遍历项目目录中的所有文件和文件夹
find "$PROJECT_DIR" -depth -name '*-*' | while read -r file; do
    # 获取新的文件名
    new_file=$(echo "$file" | sed 's/-/_/g')
    
    # 检查新文件名是否与原文件名相同
    if [ "$file" != "$new_file" ]; then
        # 重命名文件
        mv "$file" "$new_file"
        echo "Renamed '$file' to '$new_file'"
    fi
done