import collections
import math
import sys
import twitter
import pandas as pd
import time

c_k = 'oPbv9rsGgmsh3E0vfN4apXJC5'
c_sk = 'E2VplBQ0noObKdlIhR8j151rMx8uWM3XSKtUGkWj9mDuWC6sG5'
a_t_k = '816115737177784320-22CHSu85ucFYuKFGIQOQ01ie2lIRDw7'
a_t_sk = 'cjK45x0yVpqunT6L2ZJDvt47HEyymdmvXy8n3A6AoSH0i'

raw_data_loc = 'WH_dataset_withText_save'

api = twitter.Api(consumer_key=c_k,
                  consumer_secret=c_sk,
                  access_token_key=a_t_k,
                  access_token_secret=a_t_sk,
                  sleep_on_rate_limit=True)

tID_raw = pd.read_csv(raw_data_loc)
tID_text = tID_raw.values
print(tID_text)

tID_text_toReplace = tID_text.copy()
i = 0
for ind, id, label in tID_text:
    if label != 'racism' and label != 0 and label != 1:
        print(id)
        if str(id) != 'nan' and '#' not in id and id != 'NA' and id != 'NAr' and ' ' not in id:
            id = int(id)
            tweet = ''
            try:
                tweet = api.GetStatus(id).text
            except twitter.error.TwitterError as e:
                print(str(e))
                tweet = 'error in fetching'
            tID_text_toReplace[i, 1] = tweet
            tID_text_toReplace[i, 2] = 1 if label == 'sexism' else 0
            pd.DataFrame(tID_text_toReplace).to_csv('WH_dataset_withText')
    elif label != 0 or label != 1:
        tID_text_toReplace[i, 2] = 'NA'
    i += 1
#    time.sleep(0.5)
    print(i)



print("Test status: " + str(api.GetStatus('572342978255048705').text))