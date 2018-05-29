from pymouse import PyMouseEvent
import time

file = open('run1.txt','w') 

class ListenInterrupt(Exception):
    pass

class Click(PyMouseEvent):
    clickLines = 0
    start = 0
    started= False

    def __init__(self):
        PyMouseEvent.__init__(self)

    def click(self, x, y, button, press):
        if press:
            if not self.started:
                self.started = True
                self.start = time.time()
            #if button == 1:
            print('%f, %d, %d, 1, ' % (time.time()-self.start,x, y))
            file.write('%f, %d, %d, 1, \n' % (time.time()-self.start,x, y))
            self.clickLines = self.clickLines +1
        if self.clickLines >= 50:
            raise ListenInterrupt("50 Clicks - End of Log.")
    
    def move(self, x, y):
        if self.started:
            pass
            print('%f, %d, %d, 0, ' % (time.time()-self.start,x, y))
            file.write('%f, %d, %d, 0, \n' % (time.time()-self.start,x, y))

C = Click()

try:
    C.run()
except ListenInterrupt as e:
    print(e.args[0])
    file.close() 
