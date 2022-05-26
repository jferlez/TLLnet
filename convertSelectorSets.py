import TLLnet
import sys

if __name__=='__main__':

    if len(sys.argv) != 2:
        raise ValueError()
    
    tll = TLLnet.TLLnet.fromTLLFormat(sys.argv[1])
    # print(tll.selectorSets)
    # print([[[type(el) for el in s] for s in out] for out in tll.selectorSets])
    tll.toPythonIntSelectorSets()
    # print(tll.selectorSets)
    # print([[[type(el) for el in s] for s in out] for out in tll.selectorSets])
    tll.save(fname=sys.argv[1])