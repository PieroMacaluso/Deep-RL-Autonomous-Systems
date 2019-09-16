import time
import sys

from termcolor import colored


class Log(object):
    
    def __init__(self, folder):
        self.log_folder = folder + "logfile.log"

    def info(self,message):
        msg_type = "INFO"
        data = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime())
        print(colored(f"{data} [{msg_type}] - {message}", "white"))
        with open(self.log_folder, "a+") as out:
            out.write(f"{data} [{msg_type}] - {message}\n")
    
    def error(self,message):
        msg_type = "ERROR"
        data = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime())
        print(colored(f"{data} [{msg_type}] - {message}", "red"))
        with open(self.log_folder, "a+") as out:
            out.write(f"{data} [{msg_type}] - {message}\n")

    def debug(self,message):
        gettrace = getattr(sys, 'gettrace', None)
    
        msg_type = "DEBUG"
        data = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime())
        if gettrace is not None and gettrace():
            print(colored(f"{data} [{msg_type}] - {message}", "green"))
        with open(self.log_folder, "a+") as out:
            out.write(f"{data} [{msg_type}] - {message}\n")

    def important(self,message):
        msg_type = "INFO"
        data = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime())
        print(colored(f"{data} [{msg_type}] - {message}", "cyan"))
        with open(self.log_folder, "a+") as out:
            out.write(f"{data} [{msg_type}] - {message}\n")
