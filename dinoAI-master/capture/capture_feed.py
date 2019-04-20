from selenium import webdriver
import selenium.webdriver.common.keys as Keys
import os
import cv2
from mss import mss
import numpy as np
import keyboard
import time
lastsave = 0

def preprocessing(img):
    img = img[::,75:615]
    img = cv2.Canny(img, threshold1=100, threshold2=200)
    return img

# captures dinosaur run game, designed for my personal computer (adjust coordinates resepctively)
def start(driver,location):
    page = driver.find_element_by_class_name('offline')
    dino = driver.find_element_by_class_name("runner-container")
    sct = mss()
    print(location)
    x =location['x']
    y =location['y']
    print(x)
    print(y)
    coordinates = {
        'top': int(x),
        'left': int(y),  
        'width': 500,
        'height': 400,
    }
    last = 0
    with open('actions.csv', 'w') as csv:
        x = 0
        if not os.path.exists(r'./images'):
            os.mkdir(r'./images')
        while True:
            #print(dino.location)
            img = preprocessing(np.array(sct.grab(coordinates)))
            global lastsave
            if keyboard.is_pressed('up arrow')and time.time()-lastsave>0.5:
                lastsave =time.time() 
                cv2.imwrite('./images/frame_{0}.jpg'.format(x), img)
                print('jump write')
                csv.write('1\n')
                x += 1
                last=1
            elif keyboard.is_pressed('down arrow')and time.time()-lastsave>0.5:
                lastsave =time.time()
                cv2.imwrite('./images/frame_{0}.jpg'.format(x), img)
                print('duck')
                csv.write('2\n')
                x += 1
                last=2
            #if keyboard.is_pressed('q'):
             #   return
            elif keyboard.is_pressed('t')and time.time()-lastsave>0.5:
                lastsave =time.time()
                cv2.imwrite('./images/frame_{0}.jpg'.format(x), img)
                print('nothing')
                csv.write('0\n')
                x += 1
                last =0 
            # break the video feed
            elif keyboard.is_pressed('q'):
                csv.close()
                cv2.destroyAllWindows()
                break
                return
