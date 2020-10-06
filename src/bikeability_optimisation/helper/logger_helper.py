"""
This module includes all necessary functions for the logging functionality.
"""
import time
import numpy as np
from datetime import datetime


def get_duration(st, et):
    """
    Returns the duration between start time and end time.
    :param st: start time
    :type st: timestamp
    :param et: end time
    :type et: timestamp
    :return: days, hours, minutes, seconds
    :rtype: int, int, int, int
    """
    sd = datetime.fromtimestamp(st)
    ed = datetime.fromtimestamp(et)
    td = abs(ed - sd)
    return int(td.days), td.seconds//3600, td.seconds//60 % 60, td.seconds % 60


def log_to_file(file, txt, stamptime=time.localtime(), start=None,
                end=None, stamp=True, difference=False):
    """
    Logs to file with stamp time, duration and message.
    :param file: Path of the logfile
    :type file: str
    :param txt: Message to log
    :type txt: str
    :param stamptime: Time of log
    :type stamptime: time.struct_time
    :param start: Start time to calculate a duration
    :type start: float or time.struct_time
    :param end: End time to calculate a duration
    :type end: float or time.struct_time
    :param stamp: Log stamp time
    :type stamp: bool
    :param difference: Log duration
    :type difference: bool
    :return: None
    """
    if difference and not stamp:
        dur = get_duration(start, end)
        print('{0:} after {1:d}d{2:02d}h{3:02d}m{4:02d}s.'
              .format(txt, dur[0], dur[1], dur[2], dur[3]))
        with open(file, 'a+') as logfile:
            logfile.write('{0:} after {1:d}d{2:02d}h{3:02d}m{4:02d}s.\n'
                          .format(txt, dur[0], dur[1], dur[2], dur[3]))
    elif not difference and stamp:
        print('{}: {}.'.format(time.strftime('%d %b %Y %H:%M:%S', stamptime),
                               txt))
        with open(file, 'a+') as logfile:
            logfile.write('{}: {}.\n'.format(time.strftime('%d %b %Y %H:%M:%S',
                                                           stamptime), txt))
    elif difference and stamp:
        dur = get_duration(start, end)
        print('{0:}: {1:} after {2:d}d{3:02d}h{4:02d}m{5:02d}s.'
              .format(time.strftime('%d %b %Y %H:%M:%S', stamptime), txt,
                      dur[0], dur[1], dur[2], dur[3]))
        with open(file, 'a+') as logfile:
            logfile.write('{0:}: {1:} after {2:d}d{3:02d}h{4:02d}m{5:02d}s.\n'
                          .format(time.strftime('%d %b %Y %H:%M:%S',
                                                stamptime),
                                  txt, dur[0], dur[1], dur[2], dur[3]))
    else:
        print('{}: {}.'.format(time.strftime('%d %b %Y %H:%M:%S', start), txt))
        with open(file, 'a+') as logfile:
            logfile.write('{}: {}.\n'.format(time.strftime('%d %b %Y %H:%M:%S',
                                                           start), txt))
