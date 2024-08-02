##########################################################################################
# Description : Designed to monitor the moments when cigarettes are bought from various cashier points of sale (POS). The script continuously checks specified URLs for the presence of cigarette purchase entries and logs the relevant details into a file.
# Author     : Daniela Rigoli
# Creation    : 2023
#
##########################################################################################


import urllib.request
import time

word = "CIG "
cashiers = [2,3,4,5,6,7,8,9,11]

while True:
    for cashier in cashiers:
        url = str('https://<URL>/pdv/3/'+ str(cashier) +'/')
        print(url)
        webUrl=urllib.request.urlopen(url)

        content=webUrl.read()
        if word in str(content):
            file = open('outputA.log', 'a')
            arr = content.split()
            file.write('caixa ' +  str(cashier))
            file.write('\tdata ' + str(arr[0]))
            file.write('\thora ' + str(arr[1]))
            pos = str(content).find(word)
            file.write('\t**produto ' + str(content[pos-20:pos +50]) + '**\t')
            file.write('\n')
            file.close()
            time.sleep(1)
