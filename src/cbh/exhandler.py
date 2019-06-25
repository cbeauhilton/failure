
import traceback 
import sys
import os
from datetime import datetime

try:
    os.remove("exceptions.txt")
except OSError:
    pass

def exhandler(ex):
    exfile = open("exceptions.txt","a")
    exfile.write('#' * 80) # a line of 80 `#` to make a clear delineation bw errors
    exfile.write("\n \n") # a blank line - like hitting enter/return twice
    exfile.write(str(datetime.now()))
    exfile.write("\n \n") # a blank line - like hitting enter/return twice
    exfile.write(f"Error on line {sys.exc_info()[-1].tb_lineno}:") # line number
    exfile.write(f"\n {str(ex)}") # short version of the error
    exfile.write("\n \n")
    exfile.write(traceback.format_exc()) # long version of the error
    exfile.write("\n")
    exfile.write('#' * 80)
    exfile.write("\n \n")
    exfile.close()
    print(f"{sys.exc_info()[-1].tb_lineno}:")
    print(ex, "\n")
#     print(traceback.format_exc())