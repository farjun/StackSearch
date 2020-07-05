import sys


def model(word: str):
    base = ord('a')
    histogram = [0] * (ord('z') - base + 1)
    word = word.lower()
    for c in word.lower():
        histogram[ord(c) - base] = word.count(c)
    return histogram


def tag(word):
    return model(word)


class Listener:
    TAG_COMMAND = "tag:"

    def listen(self):
        s = sys.stdin.readline().strip()
        while s not in ['|']:
            s = s.lower()

            if s.startswith(Listener.TAG_COMMAND):
                s_tag = tag(s[len(Listener.TAG_COMMAND):])
                sys.stdout.write(str(s_tag))
                sys.stdout.flush()

            s = sys.stdin.readline().strip()


def resolve_stdin():
    backup = sys.stdin
    if len(sys.argv) == 2:
        # I Will allow a input file and not just input from command line.
        sys.stdin = open(sys.argv[1])
    return backup


def restore_stdin(stdin):
    if stdin != sys.stdin:
        sys.stdin.close()
        sys.stdin = stdin


if __name__ == '__main__':
    backup = resolve_stdin()
    Listener().listen()
    restore_stdin(backup)
