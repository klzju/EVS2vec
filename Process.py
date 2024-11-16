import matplotlib.pyplot as plt
import seaborn as sns
import socket
import dpkt
import numpy as np
import pickle
import json

def ParseTraffic(PcapFile,delta=1.):
    Sequence=[]
    with open(PcapFile,'rb') as f:
        t = 0 # 分段时间
        t0 = 0 # 记录初始时间
        payload_len = 0 #记录负载长度
        packets=dpkt.pcap.Reader(f)
        for ts,buf in packets:
            t0=ts if t0==0 else t0
            # 前进到当前时间段，保证ts-t0<t+delta
            while ts-t0>t+delta:
                Sequence.append(0)
                t += delta
            # 判断是否新增记录，当ts-t0>t+delta
            if ts-t0>t:
                # print(t,payload_len)
                Sequence.append(payload_len)
                payload_len=0
                t+=delta
            # 解析数据包
            eth = dpkt.ethernet.Ethernet(buf)
            if eth.type == dpkt.ethernet.ETH_TYPE_IP:
                ip = eth.data
                if ip.p == 6:  # TCP
                    tcp = ip.data
                    payload_len+=len(tcp.data)
                # elif ip.p == 17:  # UDP
                #     udp = ip.Traffic_Bilibili
                #     payload_len+=len(udp.Traffic_Bilibili)
                else:
                    pass
    return Sequence

def draw_traffic(Sequence):
    figsize = (int(len(Sequence) / 5), 5)
    plt.figure(figsize=figsize)
    plt.bar(range(len(Sequence)), Sequence, width=0.09)
    plt.xlim([0, len(Sequence) + 1])
    plt.xticks(range(0,len(Sequence),25),range(0,len(Sequence),25))
    # plt.ylim([0, 50000])

import random
import os
import pickle

if __name__ == '__main__':
    rootdir = 'Traffic_Bilibili'
    for title in os.listdir(rootdir):
        DATA = []
        title_path = os.path.join(rootdir,title)
        print(title)
        # for stream in random.sample(os.listdir(title_path),k=3):
        #     pcap_path = os.path.join(title_path,stream)
        #     seq = ParseTraffic(pcap_path)
        #     print(len(seq),seq[-1])
        #     draw_traffic(seq)
        #     plt.show()
        for i,stream in enumerate(os.listdir(title_path)):
            print(i,stream)
            pcap_path = os.path.join(title_path, stream)
            try:
                seq = ParseTraffic(pcap_path)
                DATA.append(seq)
            except:
                print(pcap_path)
                continue

        with open(f'data/{title}.pkl', 'wb') as f:
            pickle.dump(DATA,f)

        del DATA