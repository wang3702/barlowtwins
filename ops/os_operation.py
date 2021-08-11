# /*******************************************************************************
# *  Author : Xiao Wang
# *  Email  : wang3702@purdue.edu xiaowang20140001@gmail.com
# *******************************************************************************/
import os
import time
import random
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        sleep_time = random.random()
        time.sleep(sleep_time)#in order to avoid same directory collision#gen file before submitting jobs
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                return False
        print(path + " created")
        return True
    else:
        print (path+' existed')
        return False
def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text