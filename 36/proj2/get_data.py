
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib.request
from PIL import Image
import hashlib


act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']





def timeout(func, args=(), kwargs={}, timeout_duration=500, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.request.FancyURLopener()


def ModifyImage(folder, filename, Corners):
    try:
        im = imread(folder+filename)
    except IOError:
        print(filename+" is corrupted")
        return
    im = im[int(Corners[1]):int(Corners[3]), int(Corners[0]):int(Corners[2])]
    # im = imresize(im, (227, 227))
    im = Image.fromarray(im)
    im.save('cropped/'+filename)
    print("Modified" + filename)

#Note: you need to create the uncropped folder first in order
#for this to work

for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("facescrub_actors.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
            if not os.path.isfile("uncropped/"+filename):
                continue
            if (hashlib.sha256(open("uncropped/"+filename, "rb").read()).hexdigest() == line.split()[6]):
                ModifyImage("uncropped/", filename, line.split()[5].split(','))
            # print line.split()[4]
            # print line.split()[6]
            # ModifyImage("uncropped/", filename, line.split()[5].split(','))
            print(filename)
            i += 1


    for line in open("facescrub_actresses.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
            if not os.path.isfile("uncropped/"+filename):
                continue
            if (hashlib.sha256(open("uncropped/"+filename, "rb").read()).hexdigest() == line.split()[6]):
                ModifyImage("uncropped/", filename, line.split()[5].split(','))
            # print line.split()[4]
            # print line.split()[6]
            # ModifyImage("uncropped/", filename, line.split()[5].split(','))
            print(filename)

            i += 1
