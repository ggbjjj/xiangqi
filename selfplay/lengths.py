"""Actual generated plies per finished game; padding and warm-up are excluded."""
from collections import Counter


class PlyHistogram:
    def __init__(self,width=32):
        self.width=width; self.counts=Counter(); self.total=0; self.steps=0
        self.minimum=None; self.maximum=None

    def add(self,plies,count=1):
        if type(plies) is not int or plies<1 or type(count) is not int or count<0:
            raise ValueError('棋局步数必须为正整数，计数必须为非负整数')
        if not count: return
        self.counts[(plies-1)//self.width]+=count
        self.total+=count; self.steps+=plies*count
        self.minimum=plies if self.minimum is None else min(self.minimum,plies)
        self.maximum=plies if self.maximum is None else max(self.maximum,plies)

    def snapshot(self):
        bins=[]
        for i in range(max(self.counts,default=-1)+1):
            lower=i*self.width+1; upper=(i+1)*self.width; count=self.counts[i]
            bins.append({'lower':lower,'upper':upper,'label':f'{lower}–{upper}',
                         'count':count,'percent':100*count/self.total if self.total else 0})
        return {'bin_width':self.width,'total':self.total,'steps':self.steps,
                'min':self.minimum,'max':self.maximum,'mean':self.steps/self.total if self.total else None,
                'bins':bins}
