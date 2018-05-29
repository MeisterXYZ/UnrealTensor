from pymouse import PyMouseEvent
import time

started= False
#positions = []
clickLines = 0
file = open('run1.txt','w') 

class ListenInterrupt(Exception):
    pass

class Click(PyMouseEvent):
    def __init__(self):
        PyMouseEvent.__init__(self)
        return

    def click(self, x, y, button, press):
        if not started:
            started = True
            start = time.time()
        #if button == 1:
        if press:
            file.write('%f, %d, %d, 1, ' % (time.time()-start,x, y))
            print('%f, %d, %d, 1, ' % (time.time()-start,x, y))
            clickLines = clickLines +1
        if clickLines >= 50:
            raise ListenInterrupt("50 Clicks - End of Log.")
        return
    
    def move(self, x, y):
        file.write('%f, %d, %d, 0, ' % (time.time()-start,x, y))
        print('%f, %d, %d, 0, ' % (time.time()-start,x, y))
        #return x, y

C = Click()

try:
    C.run()
except ListenInterrupt as e:
    file.close() 
    print(e.args[0])
