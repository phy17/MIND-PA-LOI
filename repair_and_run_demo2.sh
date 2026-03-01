#!/bin/bash

# 定义颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== MIND 项目: 环境修复 & 运行 Demo 2 ===${NC}"
echo "检测到项目位置变更，正在重建虚拟环境..."

# 1. 检查 Python 3.10
PYTHON_EXEC=""
if [ -x "/opt/homebrew/bin/python3.10" ]; then
    PYTHON_EXEC="/opt/homebrew/bin/python3.10"
elif command -v python3.10 &> /dev/null; then
    PYTHON_EXEC="python3.10"
else
    echo -e "${RED}错误：未找到 Python 3.10。${NC}"
    exit 1
fi

# 2. 重建环境 (必须步骤)
rm -rf venv_mind_cpu
$PYTHON_EXEC -m venv venv_mind_cpu
source venv_mind_cpu/bin/activate

# 3. 快速安装依赖 (使用缓存，应该很快)
echo -e "${GREEN}正在重新关联依赖...${NC}"
pip install --upgrade pip setuptools wheel
export CFLAGS="-Wno-nullability-completeness"
pip install Theano-PyMC==1.1.2
pip install "numpy==1.23.5" "scipy==1.10.1" "opencv-python==4.9.0.80" \
    torch torchvision torchaudio \
    matplotlib pandas networkx joblib tqdm click av av2 argcomplete \
    colorlog distlib filelock fonttools fsspec Jinja2 kiwisolver llvmlite \
    markdown-it-py MarkupSafe mdurl mpmath nox numba packaging \
    pillow platformdirs pyarrow Pygments pyparsing pyproj python-dateutil \
    pytz rich shapely six sympy tomli typing_extensions tzdata virtualenv

# 4. 重新注入源码补丁 (因为 av2 是新装的)
echo -e "${GREEN}重新应用 av2/visualization 补丁...${NC}"
python -c "
import os, site
site_packages = site.getsitepackages()[0]
target_file = os.path.join(site_packages, 'av2', 'utils', 'dilation_utils.py')
if os.path.exists(target_file):
    with open(target_file, 'r') as f: content = f.read()
    if 'from cv2.typing import MatLike' in content:
        with open(target_file, 'w') as f:
            f.write(content.replace('from cv2.typing import MatLike', 'from typing import Any; MatLike = Any'))
        print('✅ av2 修复完成')

# 修复 visualization.py (渲染崩溃)
vis_file = 'common/visualization.py'
if os.path.exists(vis_file):
    with open(vis_file, 'r') as f: content = f.read()
    if 'Patch for Shapely 2.0' not in content:
        # 重复之前的修复逻辑
        old_val = '] for i in range(len(t))\\n    ]'
        new_val = '] for i in range(len(t))\\n    ]\\n    if len(points) > 0 and points[0] != points[-1]: points.append(points[0]) # Patch for Shapely 2.0'
        # 简单替换尝试
        if old_val in content:
            with open(vis_file, 'w') as f: f.write(content.replace(old_val, new_val))
            print('✅ visualization.py 修复完成')
"

# 5. 修改 Demo 2 配置
echo -e "${GREEN}配置 Demo 2 (关闭 CUDA)...${NC}"
python -c "
import json
try:
    with open('configs/demo_2.json', 'r') as f: data = json.load(f)
    data['use_cuda'] = False
    with open('configs/demo_2.json', 'w') as f: json.dump(data, f, indent=4)
    print('Demo 2 配置修改成功')
except Exception as e: print(e)
"

# 6. 运行
echo -e "${GREEN}>>> 启动 Demo 2 <<<${NC}"
python run_sim.py --config configs/demo_2.json
