
class Log:

    def __init__(self, name) -> None:
        self.f = open(name, "a")

    def __lshift__(self, txt):
        self.f.write(txt)
        self.f.flush()

    def print(self, txt):
        self << txt

    def println(self, txt):
        self << (txt + "\n")

    def __del__(self):
        self.f.flush()
        self.f.close()
