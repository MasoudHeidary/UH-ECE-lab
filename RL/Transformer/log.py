import time
import sys 


class Log:

    def __init__(self, name, terminal=True) -> None:
        self.f = open(name, "a")
        self.terminal = terminal

    def __lshift__(self, txt):
        txt = f"[{time.asctime()}] >> " + txt
        if self.terminal:
            print(txt, end='')
        self.f.write(txt)
        self.f.flush()

    def print(self, txt=""):
        self << txt

    def println(self, txt=""):
        self << (txt + "\n")

    def __del__(self):
        self.f.flush()
        self.f.close()



class Progress:
    def __init__(self, bars=False, labels=False, char_ln=25):
        if (bars and labels) and (len(labels) != bars):
            raise IndexError("label length and bars is different")

        if labels:
            self.bars = len(labels)
        else:
            self.bars = bars
        
        self.labels = labels
        self.char_ln = char_ln

        self.per_lst = [0 for _ in range(self.bars)]
        self.keep = False
        # self.__print_all_bars()

    @staticmethod
    def move_cursor_up(n):
        sys.stdout.write(f"\033[{n}F")
        sys.stdout.flush()
        return n
    
    @staticmethod
    def move_cursor_down(n):
        sys.stdout.write(f"\033[{n}E")
        sys.stdout.flush()
        return n
    
    def __update_one_line(self, index):
        row = self.bars - index
        self.move_cursor_up(row)
        self.__print_bar(index)
        self.move_cursor_down(row)

    def __print_bar(self, index):
        try:
            label = f"{self.labels[index]}\t"
        except:
            label = "unname"

        bar_fill = int(self.per_lst[index] * self.char_ln)
        bar = "█" * bar_fill + "-" * (self.char_ln-bar_fill)

        per = f"{self.per_lst[index] * 100 : .2f}"

        print(f"{label}|{bar}|{per}%")

    def __print_all_bars(self):
        for index in range(self.bars):
            self.__print_bar(index)
    
    def label_to_index(self, label):
        if not self.labels:
            raise LookupError("labels is not defined")
        for index, l in enumerate(self.labels):
            if l == label:
                return index
        raise LookupError("label not found")

    def keep_line(self):
        if not self.keep:
            print('PROGRESS BAR TMP')
            self.keep = True
    
    def update(self, index_label, per):
        if isinstance(index_label, str):
            index = self.label_to_index(index_label)
        else:
            index = index_label

        per = per
        if (per < 0) or (per > 1):
            raise ValueError(f"invalid value: percentage should be in range of 0-1")
        self.per_lst[index] = per
        self.__update_one_line(index)

