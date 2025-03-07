import pkgutil
import importlib
from os.path import join as join
import os


def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            m = importlib.import_module(current_module + '.' + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break
    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr

def get_next_time_number(snapshot_path_parent):
    if not os.path.exists(snapshot_path_parent):
        return f"{1:02}"
    try:
        # 获取目录下所有的内容
        contents = os.listdir(snapshot_path_parent)
        
        # 过滤出按数字命名的文件夹
        folder_numbers = [
            int(folder_name) for folder_name in contents 
            if folder_name.isdigit() and os.path.isdir(os.path.join(snapshot_path_parent, folder_name))
        ]
        
        # 如果没有数字文件夹，返回01作为下一个编号
        if not folder_numbers:
            return f"{1:02}"
        
        # 返回最大编号 + 1
        return f"{max(folder_numbers)+1:02}"
    except Exception as e:
        print(f"Error: {e}")
        return None

# 测试
if __name__ == '__main__':
    print(get_next_time_number("/home/zhang03/zxc/STFT_3DDL/model/STFT_3Ddl_das1k/pi_unet_ss64_train"))