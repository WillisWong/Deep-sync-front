# 导入 alive-progress 库
from alive_progress import alive_bar
import time

# 使用 with 语句创建一个进度条
with alive_bar(100) as bar: # 给 alive_bar 传入进度条总数目（这里是 100）
    for item in range(100):
        # 等待 1s
        time.sleep(.1)
        #更新进度条，进度 +1
        bar()