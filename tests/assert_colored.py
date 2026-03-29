class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'

def assert_colored(expr, msg_pass, msg_fail):
    if expr:
        print(f"{Color.GREEN}PASS{Color.END}: {msg_pass}")
    else:
        print(f"{Color.RED}FAIL{Color.END}: {msg_fail}")
    assert expr, msg_fail
