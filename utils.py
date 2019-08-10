import ctypes
import time



def ring():
    player = ctypes.windll.kernel32

    for i in range(10):
        time.sleep(1)
        player.Beep(1000,200)