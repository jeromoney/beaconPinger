# User turns on beacon to be fingerprinted
# Identifies peak frequency ~ 2000 hz
# Identifies period ~ 1 sec

DEBUG = True

import pickle
import pyaudio
import wave
import os
from numpy import arange
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
import numpy as np
import scipy.signal as signal

import matplotlib.pyplot as plt


from time import time


def numpy_ewma_vectorized(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


class Beacon(object):
    def __init__(self):
        self.id = str(int(time()*1000))
        self.name = None
        self.frequency = None
        self.period = None # periods per 1000 secs
        self.RECORDING_DIR = 'soundRecordings'
        self.FRQ_NEIGHBORHOOD = 2000 # hz (peak signal should be nearby)


    def recordsignal(self):
        # Records signal for a fixed time and stores as a sound file
        RECORD_SECONDS = 2 # seconds of recording time
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100 # Increasing number improves performance, slows processing
        CHUNK = 1024


        audio = pyaudio.PyAudio()

        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        print "recording..."
        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print "finished recording"
        print os.getcwd()

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open(self.RECORDING_DIR + '/' + self.id + '.wav', 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

    def getpeakFrq(self,t0,t1):
        #optimization constants
        DURATION = 15 # seconds. Listen to first 10 seconds.

        #get data
        fs, data = wavfile.read(self.RECORDING_DIR + '/' + self.id + ".wav")

        data = data.T[0]
        data = data[t0*fs:t1 * fs] #listen to first 10 seconds

        f, Period_spec = signal.periodogram(data, fs, 'flattop', scaling='spectrum')

        # search between boundaries for peak
        peak = np.argmax(Period_spec)
        self.frequency = f[peak]


    def getperiod(self):

        #optimization constants
        DOWNSAMPLE = 5
        LISTEN = 15 #seconds

        fs, data = wavfile.read(self.RECORDING_DIR + '/' + self.id + ".wav")
        data = data.T[0]
        data = data[:fs*LISTEN]

        data = abs(data)


        #downsample data
        data = data[::DOWNSAMPLE]
        if True:
            res = []

            maxCorrel = -1
            maxIndex = -1
            for i in range(fs/DOWNSAMPLE * 6 / 10 , fs/DOWNSAMPLE * 13 / 10):
                corr = np.corrcoef(data, np.roll(data, i))[1, 0]
                if maxCorrel < corr:
                    maxCorrel = corr
                    maxIndex = i
                res.append(corr)

            #plt.plot(res, 'r')
            #plt.show()
            self.period = int((float(maxIndex * DOWNSAMPLE) / fs) * 1000)


        else:
            maxSound = max(data)

            i = 0
            results = []

            # naive algo to find period of signal
            # listens for threshold to be exceeded
            # makes a note of time and then jumps ahead past signal
            while i < len(data):
                if data[i] > maxSound * 25 / 100:
                    results.append(i)
                    i += fs * 6 / 10
                else:
                    i += 1

            old = float(results[0]) / fs
            periods = []
            for i in results[1:]:
                new = float(i) / fs
                time = new - old
                old = new
                periods.append(int(time * 1000))

            self.period = int(sum(periods) / len(periods))

    def findmatch(self, beacons):

        minDist = 20
        match = -1
        for beacon in beacons:
            dist = (self.period - beacon.period)**2 + (self.frequency - beacon.frequency)**2
            if dist < minDist and self.id != beacon.id:
                minDist = min(minDist, dist)
                match = beacon
        return match


if __name__ == '__main__':
    recordings = os.listdir('/home/justin/PycharmProjects/beaconPinger/recordedBacons/')

    ids = {}

    for i in range(0,30,5):
        for recording in recordings:
            newBeacon = Beacon()
            #newBeacon.recordsignal()
            newBeacon.id = recording[:-4]
            if newBeacon.id not in ids.keys():
                ids[newBeacon.id] = []

            newBeacon.getpeakFrq(i,i+5)
            ids[newBeacon.id].append(newBeacon.frequency)


        #newBeacon.getperiod()
        #
        #
        # if os.path.isfile("save.p"):
        #     beacons = pickle.load( open( "save.p", "rb" ) )
        #     beacons.append(newBeacon)
        #     pickle.dump(beacons,open("save.p","wb"))
        #
        # else:
        #     beacons = [newBeacon]
        #     pickle.dump(beacons,open("save.p","wb"))
        #
        # matchBeacon = newBeacon.findmatch(beacons)
        #
        # if matchBeacon == -1:
        #     print "No match found"
        # else:
        #     print "Match is " + matchBeacon.id


        #beacons.append(newBeacon)
        #pickle.dump(beacons, open("save.p", "wb"))


        #print("{0},{1}".format(newBeacon.id[:6],str(newBeacon.frequency)))
        #print("{0};{1}".format(newBeacon.id,str(newBeacon.period)))

    for id in ids.keys():
        print ids[id]