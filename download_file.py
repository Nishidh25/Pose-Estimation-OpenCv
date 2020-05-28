# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:41:18 2020

@author: Nishidh Shekhawat
"""
import urllib.request 
from tqdm import tqdm
import os 

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download(url,filename):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
        urllib.request.urlretrieve(url, filename, reporthook=t.update_to)

def downloadfile(url,filename):
    if os.path.exists(filename):
        if input("File allready exists do you want to overrite ? [y/n] : ") == "y":
            download(url,filename)
        else : print("Cancelling download operation")
    else : download(url,filename)