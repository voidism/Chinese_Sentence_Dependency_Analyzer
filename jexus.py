import time


class Clock:
    def __init__(self, epoch, update_rate=50, flush_rate=1, title=""):
        self.title = title
        self.start_time = time.time()
        self.iteration = epoch
        self.rate = update_rate
        self.rem_time = 0
        self.pass_time = 0
        self.info_dict = {}
        self.last_txt_len = 0
        self.idx = 0
        self.flush_rate = flush_rate

    def set_start(self):
        self.start_time = time.time()

    def set_total(self, epoch):
        self.iteration = epoch

    def info2str(self, info):
        info_str = ''
        for i in info.keys():
            txt = str(round(info[i], 4))
            info_str += ' ' + str(i) + ': '
            info_str += txt
            info_str += ' '*(6-len(txt))
        return info_str

    def flush(self, info={}, enter=False):
        print_txt = "\tETA: "+str(round(self.rem_time, 0)) + \
            " s" + self.info2str(info)
        if self.idx == 0 and self.title != "":
            self.last_txt_len = len(print_txt)
            print("\n<=== ["+self.title+"] ===>")
        elif self.idx % self.rate == 1:
            self.pass_time = time.time() - self.start_time
            self.rem_time = self.pass_time * \
                (self.iteration - self.idx) / self.idx
        
        if self.idx % self.flush_rate != 0:
            pass
        else:
            print(
                chr(13) + "|" + "=" * (50 * self.idx // self.iteration
                                    ) + ">" + " " * (50 * (self.iteration - self.idx) // self.iteration
                                                        ) + "| " + str(
                    round(100 * self.idx / self.iteration, 1)) + "%",
                #"\tave cost: "+str(round(cost, 2)) if cost != 0 else "",
                print_txt+' '*(self.last_txt_len - len(print_txt)),
                sep=' ', end='', flush=True)
            self.last_txt_len = len(print_txt)
            if self.idx == self.iteration-1:
                print("")
            if enter:
                print("")
        self.idx += 1
