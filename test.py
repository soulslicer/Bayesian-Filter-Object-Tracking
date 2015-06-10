from Algorithms import *
import time
import random

field = GaussianField(height=580, width=780, scale=0.5, memory=5)

var = 1
while(1):
    var = var + 1
    start_time = time.time()

    # # Noise
    field.add_object(scaleX=50, scaleY=50, Theta=0, x=490-var+random.randrange(-200, 200, 1), y=290+random.randrange(-200, 200, 1), sensor=0)
    field.add_object(scaleX=50, scaleY=50, Theta=0, x=490-var+random.randrange(-200, 200, 1), y=290+random.randrange(-200, 200, 1), sensor=0)
    field.add_object(scaleX=50, scaleY=50, Theta=0, x=490-var+random.randrange(-200, 200, 1), y=290+random.randrange(-200, 200, 1), sensor=0)

    print var
    if var > 55 and var < 90:
        print "S1 LOSS"
        pass
    else:
        field.add_object(scaleX=20, scaleY=20, Theta=0, x=490-var, y=290, sensor=0)

    if var > 35 and var < 45:
        print "S2 LOSS"
        pass
    else:
        field.add_object(scaleX=20, scaleY=200, Theta=0, x=490-var, y=290, sensor=1)

    field.cycle()
    F = field.compute()

    cv2.imshow("a",F)
    cv2.waitKey(15)
    time.sleep(0.08)

