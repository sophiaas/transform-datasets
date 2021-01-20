import re
import numpy as np
import pandas as pd

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def read_event_txt(filename, savename=None):
    events = pd.read_csv(filename, header=None)
    events = list(events[0])
    events_parsed = []
    for e in events:
        t, x, y, p = e.split(' ')
        p = int(p)
        if p == 0:
            p = -1
        events_parsed.append([int(x), int(y), float(t), p])

    events = np.array(events_parsed)
    if savename is not None:
        np.save(savename, events)
    return events