class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def print_yellow(str):
    print(f'{bcolors.WARNING}{str}{bcolors.ENDC}')
def print_green(str):
    print(f'{bcolors.OKGREEN}{str}{bcolors.ENDC}')
def print_on_start_on_end(func):
    import time
    def inner1(*args, **kwargs):
        begin = time.time()
        print_yellow(f'Starting execution of {func.__name__}')
        rets = func(*args, **kwargs)
        print_green(f'Execution completed of {func.__name__}')
        end = time.time()
        print(f'Total time taken in {func.__name__}: {end - begin} s')
        return rets
    return inner1
def execute_command_line(command):
    from os import system
    print_yellow(command)
    system(command)