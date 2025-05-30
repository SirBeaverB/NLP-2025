import json
import requests
import getpass
from urllib.parse import urljoin

# Your QA Function
from run_qa import evaluate

# 提交地址
base_url = 'http://10.47.36.145:18081'

def input_idx():
    idx = input('idx: ')
    # maybe some restrictions
    return idx

def input_passwd():
    passwd = getpass.getpass('passwd for final submission (None for debug mode): ')
    if passwd == '':
        print('=== DEBUG MODE ===')
    return passwd

def login(idx, passwd):
    url = urljoin(base_url, 'login')
    r = requests.post(url, data={'idx': idx, 'passwd': passwd})
    r_dct = eval(r.text)
    queries = r_dct['queries']
    if r_dct['mode'] == 'illegal':
        raise ValueError('illegal password!')
    print(f'{len(queries)} queries.')
    return queries

def send_ans(idx, passwd, answers):
    url = urljoin(base_url, 'em')
    r = requests.post(url, data={'idx': idx, 'passwd': passwd, 'answers': json.dumps(answers)})
    print(r.text)
    r_dct = eval(r.text)
    if r_dct['mode'] == 'illegal':
        raise ValueError('illegal password!')
    return r_dct['mode'], r_dct['em']

def main():
    idx = input_idx()
    passwd = input_passwd()
    queries = login(idx, passwd)
    tot_answers = evaluate(queries)
    mode, em = send_ans(idx, passwd, tot_answers)
    print(f'EM: [{em}], [{mode}] mode')

if __name__ == '__main__':
    main()