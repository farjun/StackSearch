# import select
# import sys
# import os
# import subprocess
# import time
# from typing import Union
#
#
# class Logger():
#
#     def __init__(self):
#         self.log = open("log.txt", "w")
#
#     def write(self, s):
#         self.log.write(s)
#
#     def close(self):
#         self.log.close()
#
#
#
#
# class FDListener:
#     def __init__(self, fd_in: Union[int, str], fd_out: Union[int, str]):
#         self.fd_in = fd_in
#         self.fd_out = fd_out
#         # self.in_io = os.fdopen(int(self.fd_in), "r+")
#         # self.out_io =os.fdopen(int(self.fd_out), "r+")
#         self.in_io = sys.stdin
#         self.out_io = sys.stdout
#         LOGGER.write("Start\n")
#
#     def listen(self, sleep_between_reads_secs=1, terminate_after=60):
#
#         count = 0
#         to_terminate = False
#         LOGGER.write("Start listen")
#         while count < terminate_after and not to_terminate:
#             line = None
#             if self.in_io.isatty():
#                 line = self.in_io.readline()
#             if line:
#                 print(f"got line: {line}")
#                 LOGGER.write(f"got line: {line}")
#
#                 to_terminate = line.strip().lower() == "done"
#             else:
#                 print(f"no new input on sec: {count * sleep_between_reads_secs}")
#             time.sleep(sleep_between_reads_secs)
#             count += 1
#         LOGGER.write("Done listen")
#
#
# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print("Usage: python_side -|number")
#         exit(2)
#
#     LOGGER = Logger()
#     FDListener("-", "-").listen()
#     LOGGER.close()


import sys
s = sys.stdin.readline().strip()
while s not in ['break', 'quit']:
    sys.stdout.write(s.upper() + '\n')
    sys.stdout.flush()
    s = sys.stdin.readline().strip()