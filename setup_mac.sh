#!/bin/bash

# 定义颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MIND 项目 Mac (M-Series) 终极修复版 V6 (含渲染修复) ===${NC}"

# 1. 检查 Python 3.10
PYTHON_EXEC=""
if [ -x "/opt/homebrew/bin/python3.10" ]; then
    PYTHON_EXEC="/opt/homebrew/bin/python3.10"
elif command -v python3.10 &> /dev/null; then
    PYTHON_EXEC="python3.10"
else
    echo -e "${RED}未找到 Python 3.10。请确保已运行 'brew install python@3.10'。${NC}"
    exit 1
fi

# 2. 清理环境 (渲染报错不需要清理旧环境，但为了保险起见还是清理)
# echo -e "${GREEN}[2/9] 清理旧环境...${NC}"
# rm -rf venv_mind_cpu  # 这次就不删了，节省时间，直接在原环境打补丁
# rm -rf ~/.theano

# 3. 激活/创建虚拟环境
echo -e "${GREEN}[3/9] 激活虚拟环境...${NC}"
if [ ! -d "venv_mind_cpu" ]; then
    $PYTHON_EXEC -m venv venv_mind_cpu
fi
source venv_mind_cpu/bin/activate

# 4. 生成 .theanorc (同 V4)
echo -e "${GREEN}[4/9] 确认 ~/.theanorc 配置...${NC}"
cat > ~/.theanorc <<EOL
[global]
device = cpu
floatX = float32
cxx = g++

[blas]
ldflags = -Wl,-framework,Accelerate
EOL

# 5. 安装依赖 (同 V4，确保依赖齐全)
echo -e "${GREEN}[5/9] 检查依赖...${NC}"
export CFLAGS="-Wno-nullability-completeness"
pip install Theano-PyMC==1.1.2
pip install "numpy==1.23.5" "scipy==1.10.1" "opencv-python==4.9.0.80" \
    torch torchvision torchaudio \
    matplotlib pandas networkx joblib tqdm click av av2 argcomplete \
    colorlog distlib filelock fonttools fsspec Jinja2 kiwisolver llvmlite \
    markdown-it-py MarkupSafe mdurl mpmath nox numba packaging \
    pillow platformdirs pyarrow Pygments pyparsing pyproj python-dateutil \
    pytz rich shapely six sympy tomli typing_extensions tzdata virtualenv

# 6. Patch av2 源码 (同 V3， av2 修复)
echo -e "${GREEN}[6/9] 检查 av2 源码补丁...${NC}"
python -c "
import os
import site

site_packages = site.getsitepackages()[0]
target_file = os.path.join(site_packages, 'av2', 'utils', 'dilation_utils.py')

if os.path.exists(target_file):
    with open(target_file, 'r') as f:
        content = f.read()
    
    old_str = 'from cv2.typing import MatLike'
    new_str = 'from typing import Any; MatLike = Any # Patched by MIND setup script'
    
    if old_str in content:
        new_content = content.replace(old_str, new_str)
        with open(target_file, 'w') as f:
            f.write(new_content)
        print('✅ av2 源码修复成功！')
"

# 7. 【关键修复】Patch visualization.py (Shapely 2.0 兼容性)
echo -e "${GREEN}[7/9] 正在修复 visualization.py (渲染崩溃问题)...${NC}"
python -c "
import os

target_file = 'common/visualization.py'

if os.path.exists(target_file):
    with open(target_file, 'r') as f:
        content = f.read()
    
    # 问题代码: points = [[x, y] for i in range(len(t))]
    # Shapely 2.0 要求 LinearRing 必须闭合，即最后一个点 = 第一个点
    
    old_code = '''    points = [
        [
            circle.center[0] + circle.radius * np.cos(t[i]),
            circle.center[1] + circle.radius * np.sin(t[i])
        ] for i in range(len(t))
    ]'''
    
    new_code = '''    points = [
        [
            circle.center[0] + circle.radius * np.cos(t[i]),
            circle.center[1] + circle.radius * np.sin(t[i])
        ] for i in range(len(t))
    ]
    # Patch for Shapely 2.0: Ensure LinearRing is closed
    if len(points) > 0 and points[0] != points[-1]:
        points.append(points[0])'''
    
    if 'Patch for Shapely 2.0' not in content:
        if old_code in content:
            new_content = content.replace(old_code, new_code)
            with open(target_file, 'w') as f:
                f.write(new_content)
            print('✅ visualization.py 修复成功！已强制闭合多边形点集。')
        else:
            print('⚠️ 未找到目标代码块，可能格式不匹配，尝试模糊替换...')
            # 备用替换方案，匹配关键行
            pivot = '] for i in range(len(t))'
            if pivot in content and 'points.append(points[0])' not in content:
                # 找到该行并追加闭合逻辑
                lines = content.splitlines()
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if pivot in line and line.strip().endswith(']'):
                         new_lines.append('    # Patch for Shapely 2.0: Ensure LinearRing is closed')
                         new_lines.append('    if len(points) > 0 and points[0] != points[-1]: points.append(points[0])')
                
                with open(target_file, 'w') as f:
                    f.write('\n'.join(new_lines))
                print('✅ visualization.py (备用方案) 修复成功！')
    else:
        print('ℹ️ visualization.py 似乎已经修复过。')

else:
    print(f'❌ 未找到文件: {target_file}')
"

# 8. 修改配置 (同 V4)
echo -e "${GREEN}[8/9] 确认配置文件...${NC}"
python -c "
import json
import os
try:
    with open('configs/demo_1.json', 'r') as f:
        data = json.load(f)
    data['use_cuda'] = False
    with open('configs/demo_1.json', 'w') as f:
        json.dump(data, f, indent=4)
except: pass
"

# 9. 运行仿真
echo -e "${GREEN}[9/9] 开始运行仿真...${NC}"
echo "----------------------------------------"
python run_sim.py --config configs/demo_1.json
