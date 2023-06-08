import requests
import re
import pandas as pd
from multiprocessing import Pool
import string
import numpy as np

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}
def getrandstring():
  letters = string.ascii_letters + string.digits
  letters = list(letters)
  letters = ''.join(np.random.choice(letters,2))
  return letters

def get_image_urls(query_key,nres=50,iter=50):
    """
        Get all image url from google image search
        Args:
            query_key: search term as of what is input to search box.
        Returns:
            (list): list of url for respective images.
 
    """
    urllist = []
    i = 0
    query_key = query_key.replace(' ','+')#replace space in query space with +
    query = query_key
    while (len(urllist)<nres)|(i>iter):
      print('complete: %.2f%%'%(len(urllist)/nres*100))
      tgt_url = 'https://www.google.co.th/search?q={}&tbm=isch&tbs=sbd:0'.format(query)#last part is the sort by relv
      r = requests.get(tgt_url, headers = headers)
      urllist = urllist + [n for n in re.findall('"(https[a-zA-Z0-9_./:-]+.(?:jpg|jpeg|png))",', r.text)]
      urllist = list(set(urllist))
      randomstr = getrandstring()
      query = query_key + randomstr
    print('completed')
    return urllist