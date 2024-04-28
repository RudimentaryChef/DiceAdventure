import os

objc_disable_initialize_fork_safety = os.getenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY")
print(objc_disable_initialize_fork_safety)
