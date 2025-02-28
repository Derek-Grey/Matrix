# %%
import subprocess
import os

def install_packages(packages, wheel_dir):
    # 确保目标目录存在
    if not os.path.exists(wheel_dir):
        os.makedirs(wheel_dir)

    
    
    for package in packages:
        subprocess.run(['pip', 'download', package, '-d', wheel_dir])
        # subprocess.run(['pip', 'install', f'{wheel_dir}/{package}*.whl'])

if __name__ == "__main__":
    # 要安装的包列表
    packages_to_install = ['Cython']
    
    # 目标路径
    destination_path = r'E:\whls'
    
    # 调用函数安装包
    install_packages(packages_to_install, destination_path)