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


class Beacon(object):
    def __init__(self):
        self.id = str(int(time()*1000))
        self.name = None
        self.frequency = None
        self.period = None # secs
        self.rise = None # time signal is sent ~0.2s
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
        LISTEN = 30 #seconds
        SMOOTH_FQ = 4000 #hz smoothes signal to frequency

        fs, data = wavfile.read(self.RECORDING_DIR + '/' + self.id + ".wav")
        data = data.T[0]
        data = abs(data)

        # average data over a half of period (4000 hz)
        smple_period = int(fs / SMOOTH_FQ)

        data = [sum(data[i: i + smple_period]) / SMOOTH_FQ for i in range(0, len(data), smple_period)]

        i = 0
        smple = 100
        avgSnd = sum(data[:smple])
        #Ensure that we start counting before beep, during quiet time
        while avgSnd > smple * max(data) / 3 and i + smple < len(data):
            avgSnd += data[i + smple] - data[i]
            i += 1

        ups = []
        downs = []
        half_lmt = max(data) / 2

        while i + 1 < len(data):
            if data[i] < half_lmt <= data[i+1]:
                ups.append(float(i) / fs)
                i += 30
            elif data[i] > half_lmt >= data[i+1]:
                downs.append(float(i) / fs)
                i += 30

            i += 1

        self.period = ((ups[-1] - ups[0]) / (len(ups) - 1) + (downs[-1] - downs[0]) / (len(downs) - 1)) * smple_period / 2

        #make sure we don't end up with mismatched ups and downs
        downs = [i for i in downs if i > ups[0]]
        ups = [i for i in ups if i < downs[-1]]
        self.rise = sum([downs[i] - ups[i] for i in range(len(downs))]) / (len(downs) - 1)


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

    for recording in recordings:
        i = 0
        newBeacon = Beacon()
        #newBeacon.recordsignal()
        newBeacon.id = recording[:-4]

        newBeacon.getpeakFrq(i,i+5)
        newBeacon.getperiod()

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

        print("{0}\t{1}\t{2}".format(newBeacon.id[:6],str(newBeacon.frequency),str(newBeacon.period), str(newBeacon.rise)))
