
def fetch_env(env_name, system):
    """return path to env"""
    file_name = None
    if env_name == "walker":
        if system == "linux":
            file_name = "../envs/walker_linux/walker.x86_64"
        if system == "window":
            file_name = "../envs/walker_window/Unity Environment.exe"
    if file_name is None:
        raise ValueError("env path does not exist")
    else:
        return file_name
