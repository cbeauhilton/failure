import cowsay

###############################################################################
#     ____  ________  ____
#    / __ \/ ___/ _ \/ __ \
#   / /_/ / /  /  __/ /_/ /
#  / .___/_/   \___/ .___/
# /_/             /_/
###############################################################################

def prep():
    import a01cleanccf
    import a01cleancmml
    import a01cleanicus
    import a01cleanmds
    import a01cleanmpn
    import a02combine


###############################################################################
#    __             _
#   / /__________ _(_)___
#  / __/ ___/ __ `/ / __ \
# / /_/ /  / /_/ / / / / /
# \__/_/   \__,_/_/_/ /_/
###############################################################################

def train():
    # import a03train
    # import a04metrics
    # import a05shap
    # import a02prettify
    import a06forceplotmerge


###############################################################################
#                                __
#    ________  ____  ____  _____/ /_
#   / ___/ _ \/ __ \/ __ \/ ___/ __/
#  / /  /  __/ /_/ / /_/ / /  / /_
# /_/   \___/ .___/\____/_/   \__/
#          /_/
###############################################################################

def report():
    import c00eda
    import b00abstractconf
    import b01abstractmanu
    import b02methods
    import b03results
    import b04figure01
    import b04figure02
    import b04figure03
    import b05table01
    import b05table02



# prep()
train()
report()

print("#" * 80)
cowsay.tux("finis")
print("#" * 80)