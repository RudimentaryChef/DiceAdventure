import os

objc_disable_initialize_fork_safety = os.getenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY")

if objc_disable_initialize_fork_safety == "yes":
    print("yes")
else:
    print("no")