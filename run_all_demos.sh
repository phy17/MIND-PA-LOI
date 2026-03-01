#!/bin/bash

# 定义颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MIND 项目 Mac (M-Series) 全 Demo 运行脚本 ===${NC}"

# ... (前置环境检查和安装步骤跳过，假设已安装好，只做运行时修正) ...

# 激活环境
source venv_mind_cpu/bin/activate

# 批量修改所有配置文件
echo -e "${GREEN}正在修改所有配置文件的 GPU 设置...${NC}"
python -c "
import json
import glob
import os

files = glob.glob('configs/demo_*.json')
for file_path in files:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 强制关闭 GPU
        data['use_cuda'] = False
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f'✅ {os.path.basename(file_path)}: use_cuda = False')
    except Exception as e:
        print(f'❌ {os.path.basename(file_path)}: 失败 {e}')
"

# 询问用户要跑哪个
echo "----------------------------------------"
echo "发现以下 Demo:"
ls configs/demo_*.json | xargs -n 1 basename
echo "----------------------------------------"
echo "你要运行哪一个？(输入数字 1, 2, 3 或 all 跑全部)"
read -p "请输入: " CHOICE

if [ "$CHOICE" == "all" ]; then
    for i in {1..4}; do
        echo -e "${GREEN}>>> 正在运行 Demo $i ...${NC}"
        python run_sim.py --config configs/demo_$i.json
    done
else
    echo -e "${GREEN}>>> 正在运行 Demo $CHOICE ...${NC}"
    python run_sim.py --config configs/demo_$CHOICE.json
fi

echo "----------------------------------------"
echo "运行结束！请检查 outputs/ 目录。"
