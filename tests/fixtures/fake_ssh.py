"""Stand-in for the ssh client in tests: runs the remote command with the local POSIX sh.

Called as ``fake_ssh.py -T -o ... TARGET -- COMMAND``; stdin/stdout/stderr are inherited, so
closing the caller's stdin reaches the remote wrapper exactly as an SSH channel would.
"""
import shutil
import subprocess
import sys

args = sys.argv[1:]
command = args[args.index("--") + 1]
assert command.startswith("sh -c "), command
sh = shutil.which("sh")
sys.exit(subprocess.call([sh, "-c", command]))
