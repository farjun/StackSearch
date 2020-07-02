import sys
import os


def run_read(*args, **kwargs):
    import subprocess
    p = subprocess.Popen(["java", "MyClass"], stdin=subprocess.PIPE)
    p.stdin.writelines(
        args + [
            "line 1",
            "line 2",
            "done"
        ])
    p.stdin.flush()


def run_write(*args, **kwargs):
    pass


COMMANDS = ["read", "write"]
functions = [run_read, run_write]

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: python_side command args...")
        exit(2)

    command = argv[1]
    if command not in COMMANDS:
        valid_commands = ",".join(COMMANDS)
        print(f"Invalid command:{command}, expected: {valid_commands}")
        exit(2)

    functions[COMMANDS.index(command)](*argv[2:])
