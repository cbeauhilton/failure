
import traceback 
import sys
import os
from datetime import datetime
from cbh import config

if not os.path.exists(config.EXCEPT_DIR):
    os.makedirs(config.EXCEPT_DIR)

try:
    os.remove("exceptions.txt")
except OSError:
    pass

def exhandler(ex, module):
    expath = config.EXCEPT_DIR/f"{module}.exceptions.txt"
    # expath = config.REPORTS_DIR/f"{os.path.basename(__file__)}.exceptions.txt"
    exfile = open(expath,"a")
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

