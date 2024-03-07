import cv2
import RPi.GPIO as GPIO
import time
import os
import numpy as np
import h5py
import tflite_runtime.interpreter as tflite

class PYBOT:
    
    ############################################
    #               MANEUVERING                #
    ############################################
    
    def pinInit(self):
        GPIO.setmode(GPIO.BCM)
        for pin in self.motor_pins.values():
            GPIO.setup(pin, GPIO.OUT)
            
        self.motor_left  = GPIO.PWM(self.motor_pins['Lpwm'], self.pwmFreq)
        self.motor_right = GPIO.PWM(self.motor_pins['Rpwm'], self.pwmFreq)
        self.motor_left.start(0)
        self.motor_right.start(0)
        
    def setForward(self, pin_in1=0, pin_in2=0) :
        GPIO.output(pin_in1, 0)
        GPIO.output(pin_in2, 1)
        
    def drive(self, motorSpeed=[]):
        self.setForward(self.motor_pins['Lin1'], self.motor_pins['Lin2'])
        self.setForward(self.motor_pins['Rin1'], self.motor_pins['Rin2'])
        self.motor_left.ChangeDutyCycle (abs(motorSpeed[0]))
        self.motor_right.ChangeDutyCycle(abs(motorSpeed[1]))
        
    def saveLookUpTable(self):
        with h5py.File(name=self.LookUpTableDir+'/lookUpTable.h5', mode='w') as f:
            f.create_dataset('time',    data=self.LookUpTable['time'][:])
            f.create_dataset('Lmotor',  data=self.LookUpTable['Lmotor'][:])
            f.create_dataset('Rmotor',  data=self.LookUpTable['Rmotor'][:])
        f.close()
        
        print('-'*50)
        print(f"Lookup table saved to {self.LookUpTableDir+'/lookUpTable.h5'}")
        lookUpLen = len(self.LookUpTable['time'][:])
        print(f'LUT Length : {lookUpLen}')
        print('-'*50)

    def loadLookUpTable(self):
        time    = []
        Lmotor  = []
        Rmotor  = []
        
        with h5py.File(name=self.LookUpTableDir+'/lookUpTable.h5', mode='r') as f:
            time    = f['time'][:]
            Lmotor  = f['Lmotor'][:]
            Rmotor  = f['Rmotor'][:]
            
        f.close()
        
        print('-'*50)
        print(f"Lookup table loaded from {self.LookUpTableDir+'/lookUpTable.h5'}")
        print(f'LUT Length : {len(time)}')
        print('-'*50)
        
        return time, Lmotor, Rmotor

    ############################################
    #                 CAPTURING                #
    ############################################
    
    def getTrialDirectory (self):
        mainFolder      = self.frameDir
        
        folder          = os.listdir(mainFolder)[0]
        folderName      = folder[:5]
        
        newFolderIndex  = int(folder[-1])
        newFolderName   = folderName+'_'+str(newFolderIndex+1)
        newFolderDir    = mainFolder+'/'+newFolderName
        
        os.mkdir(newFolderDir)
        
        return newFolderDir
    
    def realTimeMovingCapture(self, capture=True, maneuver=True, lookUpTable='None'):
        self.idx = 0
        remainSpeed = False
        firstRun = True
        saveFrameDirectory = " "
        motorSpeed = np.zeros(2)
        
        if (lookUpTable == 'load'):
            lookUpTime, lookUpLmotor, lookUpRmotor = self.loadLookUpTable()
        
        while(True):
            _, raw = self.cam.read()
            
            self.showImages(raw)
            
            waitKey = cv2.waitKey(1)

            if (capture):
                if (self.idx % int(self.capturePeriod*100) == 0):
                    if (remainSpeed):
                        speedMapping = -1
                    else:
                        speedMapping = int(motorSpeed[0] - motorSpeed[1])
                    self.setLabel(manual=False, value=speedMapping)
                    
                    if (firstRun):
                        saveFrameDirectory = self.getTrialDirectory()
                        firstRun = False
                    else:
                        pass
                    
                    self.saveFrame(saveFrameDirectory, self.idx, self.img)
                            
            if (waitKey == ord('q')):
                print('Labeling Done..')
                
                if lookUpTable == 'save':
                    self.saveLookUpTable()
                    print("capture and save frame finished")
                break
            
            elif (waitKey == self.keyboardArrow['up']):
                motorSpeed[0] = self.defaultMotorSpeed
                motorSpeed[1] = self.defaultMotorSpeed
                remainSpeed = False
                
            elif (waitKey == self.keyboardArrow['down']):
                remainSpeed = True
                
            elif (waitKey == self.keyboardArrow['left']):
                motorSpeed[0] = 0
                motorSpeed[1] = self.defaultMotorSpeed
                remainSpeed = False
                
            elif (waitKey == self.keyboardArrow['right']):
                motorSpeed[0] = self.defaultMotorSpeed
                motorSpeed[1] = 0
                remainSpeed = False
                
            if (maneuver):
                if (lookUpTable == 'load'):
                    if (self.idx >= len(lookUpLmotor)):
                        print('-'*50)
                        print(f'Maneuvering Done with {len(lookUpLmotor)} elements !!')
                        print('-'*50)
                        break
                    else:
                        print(f'Motor Speed (Lookup) : {lookUpLmotor[self.idx]} {lookUpRmotor[self.idx]}')
                        self.drive(np.array([lookUpLmotor[self.idx], lookUpRmotor[self.idx]]))
                    
                else :
                    print(f'Motor Speed (manual) : {motorSpeed}')
                    self.drive(motorSpeed)
            
            if (lookUpTable == 'save'):
                self.LookUpTable['time'].append(self.idx)
                self.LookUpTable['Lmotor'].append(motorSpeed[0])
                self.LookUpTable['Rmotor'].append(motorSpeed[1])
                    
            self.idx += 1
        
        if (lookUpTable == 'save'):
            self.saveLookUpTable()
            
    def capture(self):
        self.idx = 0
        while (self.idx < self.loadVideoParam['totalFrame']):
            
            leftFrame = self.loadVideoParam['totalFrame'] - self.idx
            print(f'#[{self.idx}] Proceeded, Frame [{leftFrame}] left')
            
            if (self.initialFrameSkip):
                print(f'Skip starting frame..')
                self.frameSkip(735)
                self.initialFrameSkip = False
            
            _, raw = self.cam.read()
                        
            self.showImages(raw, self.idx)
            waitKey = cv2.waitKey()

            if (waitKey == ord('s')):
                self.setLabel()
                self.saveFrame(self.idx, self.img)

            elif (waitKey == ord('n')):
                self.frameSkip(self.defaultFrameSkip)
            
            elif (waitKey == ord('r')):
                print('Preprocess Configuration..')
                self.preprocessingConf()

            elif (waitKey == ord('q')):
                print('Labeling Done..')
                break
            
            self.idx += 1

        self.cam.release()
        cv2.destroyAllWindows()

    def frameSkip(self, number):
        for _ in range(number-1):
            self.cam.read()
            self.idx += 1
            
    def saveFrame(self, directory, idx, img):
        fileName = f'/frame_{idx}_captured_{self.userLabel}.png'
        cv2.imwrite(directory+fileName, img)
        
        file_size = os.path.getsize(directory+fileName)
        
        print(f'{fileName} saved in {directory}, File Size: {file_size} bytes')
        
    def setLabel(self, manual=True, value=0):
        if (manual):
            userEntered = input('Enter the label for this frame, in range of [-90, 90] : ').strip()
            self.userLabel = int(userEntered)
        else:
            self.userLabel = value
            
    def preprocessImg(self, raw = [], inverted = True):
        
        if (inverted):
            image  =   cv2.bitwise_not(raw)
        else:
            image  =   raw
        
        image_hsv           =   cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound         =   np.array([0, 0, 200], dtype=np.uint8)
        upper_bound         =   np.array([180, 30, 255], dtype=np.uint8)
        mask                =   cv2.inRange(image_hsv, lower_bound, upper_bound)
        masked              =   cv2.bitwise_and(image, image, mask=mask)
        masked_rgb          =   cv2.cvtColor(masked, cv2.COLOR_HSV2RGB)
        masked_yuv          =   cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2YUV)

        
        mask_yuv_resized = cv2.resize(image,(200,66))
        mask_yuv_resized = mask_yuv_resized.astype(np.float32)
        mask_yuv_resized /=255.
        
        mask_yuv_resized = np.expand_dims(mask_yuv_resized, axis =0)
        
        crop_start = 35
        cropped_image = masked_yuv[crop_start:,:,:]
        
        cropped_image = cropped_image.astype(np.float32)
        cropped_image /=255.

        cropped_image = np.expand_dims(cropped_image, axis=0)
        return cropped_image       
#        return mask_yuv_resized
    
    def perspectiveTransform(self, raw):
        h, w, _ = raw.shape
        
        pointsBefore = np.float32([self.preprocessingParam['pt_topL'], self.preprocessingParam['pt_botL'],
                                   self.preprocessingParam['pt_topR'], self.preprocessingParam['pt_botR']])
        
        pointsAfter = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
        
        transform = cv2.getPerspectiveTransform(pointsBefore,
                                                pointsAfter)
        
        lane = cv2.warpPerspective(raw, transform, (w, h))
        
        inverseTransform = cv2.getPerspectiveTransform(pointsAfter,
                                                       pointsBefore)
        
        maxWidth = max(self.preprocessingParam['pt_topL'][0], self.preprocessingParam['pt_topR'][0], self.preprocessingParam['pt_botL'][0], self.preprocessingParam['pt_botR'][0])
        maxHeight = max(self.preprocessingParam['pt_topL'][1], self.preprocessingParam['pt_topR'][1], self.preprocessingParam['pt_botL'][1], self.preprocessingParam['pt_botR'][1])
        
        overlapped = cv2.warpPerspective(lane, inverseTransform, (maxWidth, maxHeight))
        
        return lane
            
    def decodeCommand(self, string="None"):
        command_start = string.index('(')
        command = string[:command_start].strip()
        if (command == 'done'):
            param = 0
            pass
        else :
            param_string = string[command_start:]
            param = eval(param_string)
        return command, param
        
    def preprocessingConf(self):
        while (True):
            userEntered = input("Enter Command and Parameters (e.g., 'g(3,3)') : ")
            command, param = self.decodeCommand(userEntered)
                        
            if (command == 'g'):
                self.gaussianFilterSize = param
                print(f'Gaussian Filter : {param}')
                print('-'*50)
                break
            
            elif (command == 'h'):
                self.heightCropSize = param
                print(f'Crop Height : {param}')
                print('-'*50)
                break
            
            elif (command == 'w'):
                self.widthCropSize = param
                print(f'Crop Width : {param}')
                print('-'*50)
                break
            
            elif (command == 'r'):
                self.resizedFrame = param
                print(f'Resize : {param}')
                print('-'*50)
                break
            
            else:
                pass

    def showImages(self, rawFrame):
        self.img = self.preprocessImg(rawFrame)
        #cv2.imshow('raw', rawFrame
        print(f'Image Shape : {self.img.shape}') ##
        cv2.imshow('processed', self.img)

    def showDirectory(self):
        print(f'Main Directory \t\t: {self.dir}')
        print(f'Frame will be saved in \t: {self.frameDir}')
        print('-'*50)

    def setSourceVideo(self, fileName, showInfo=False):
        fileName += '.mp4'
        videoDir = self.dir+'/'+fileName
        self.cam = cv2.VideoCapture(videoDir)
        self.loadVideoParam['width']        = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.loadVideoParam['height']       = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.loadVideoParam['totalFrame']   = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.loadVideoParam['fps']          = self.cam.get(cv2.CAP_PROP_FPS)
        self.loadVideoParam['length']       = self.loadVideoParam['totalFrame']/self.loadVideoParam['fps']
        print(f'Source Video \t\t: {videoDir}')
        print('-'*50)
        if (showInfo):
            print(f"Length  \t\t: {self.loadVideoParam['length']} seconds")
            print(f"Size    \t\t: {self.loadVideoParam['height']} * {self.loadVideoParam['width']} pixels")
            print(f"Fps     \t\t: {self.loadVideoParam['fps']} frame per second")
            print(f"Total   \t\t: {self.loadVideoParam['totalFrame']} frames")
            print('-'*50)
                
    ############################################
    #                 INFERENCING              #
    ############################################

    def setInterpreter(self):
        self.interpreter     = tflite.Interpreter(model_path=self.modelDir+'/tfLiteModel_final2.tflite')
        self.interpreter.allocate_tensors()
        input_details   = self.interpreter.get_input_details()
        output_details  = self.interpreter.get_output_details()
        
        self.tfLite_input_index     = input_details[0]['index']
        self.tfLite_output_index    = output_details[0]['index']
        
        print(self.tfLite_input_index)
        print(self.tfLite_output_index)
        
    def predictSteeringAngle(self, frame):
        preprocessed = self.preprocessImg(frame)
        
        print(self.tfLite_input_index)
        
        self.interpreter.set_tensor(self.tfLite_input_index,
                                    preprocessed.reshape(1,66,200,3))
        self.interpreter.invoke()
        
        predicted = self.interpreter.get_tensor(self.tfLite_output_index)
        print(f'Predicted : {predicted}')
        print('-'*50)
        return int(predicted)

    def driveFromPrediction(self, predicted):
        steering_angle = int(predicted)
        mapped_speed = self.mapSterringToSpeed(steering, angle)
        motor_speed = self.decodeMotorSpeed(mapped_speed)
        
        self.drive(motorSpeed = motor_speed)
        
        
    def mapSterringToSpeed(self, steering_angle):
        mapped_speed = int((steering_angle + 90)/180*100)
        return mapped_speed
    def decodeMotorSpeed(self, mappedSpeed):
        motorSpeed = np.zeros(2)
        
        if (mappedSpeed > 0):
            # Right Steering
            motorSpeed[0] = mappedSpeed
            motorSpeed[1] = 0
        else :
            # Left Steering
            motorSpeed[1] = mappedSpeed
            motorSpeed[0] = 0
            
        return motorSpeed
    
    def realTimeManeuvering(self):
        motorSpeed = np.zeros(2)
        
        while(True):
            _, raw = self.cam.read()
            
            self.showImages(raw)
            waitKey = cv2.waitKey(1)
            
            if (waitKey == ord('q')):
                print('-'*50)
                print("Pybot Terminated..")
                print('-'*50)
                
            else:
                predicted = self.predictSteeringAngle(self.img)
                if (predicted == -1):
                    pass
                    # Remain Speed
                else :
                    motorSpeed = self.decodeMotorSpeed(predicted)
                
                self.drive(motorSpeed=motorSpeed)
            

    ############################################
    #                 INITIALIZING             #
    ############################################
        
    def __init__ (self, directory="None"):
        self.cam                            = cv2.VideoCapture(-1)
        self.frameH                         = 480
        self.frameW                         = 640
        self.cam.set(3, self.frameW)
        self.cam.set(4, self.frameH)
        self.dir                            = directory
        
        self.frameDir                       = self.dir+'/data'+'/frame'
        self.LookUpTableDir                 = self.dir+'/data'+'/LookUpTable'
        self.modelDir                       = self.dir+'/data'+'/model'
        
        self.img                            = np.zeros(shape=(self.frameH, self.frameW, 3))
        self.userLabel                      = 0
        self.defaultFrameSkip               = 20
        
        self.loadVideoParam                 = {}
        self.loadVideoParam['width']        = 0
        self.loadVideoParam['height']       = 0
        self.loadVideoParam['totalFrame']   = 0
        self.loadVideoParam['fps']          = 0
        self.loadVideoParam['length']       = 0
        
        self.preprocessingParam             = {}
        self.preprocessingParam['gaussian'] = False
        self.preprocessingParam['yuv']      = False
        self.preprocessingParam['crop']     = False
        self.preprocessingParam['resize']   = False
        self.preprocessingParam['pt']       = False
        
        self.preprocessingParam['pt_topL']  = np.array([])
        self.preprocessingParam['pt_topR']  = np.array([])
        self.preprocessingParam['pt_botL']  = np.array([])
        self.preprocessingParam['pt_botR']  = np.array([])
        
        self.gaussianFilterSize             = []
        self.heightCropSize                 = []
        self.widthCropSize                  = []
        self.resizedFrame                   = []
        self.idx                            = 0
        self.initialFrameSkip               = True
        
        self.pwmFreq                        = 500
        self.motor_pins                     = {}
        self.motor_pins['Lpwm']             = 18
        self.motor_pins['Lin1']             = 22
        self.motor_pins['Lin2']             = 27
        self.motor_pins['Rpwm']             = 23
        self.motor_pins['Rin1']             = 25
        self.motor_pins['Rin2']             = 24
        
        self.keyboardArrow = {}
        self.keyboardArrow['up']            = 82
        self.keyboardArrow['down']          = 84
        self.keyboardArrow['left']          = 81
        self.keyboardArrow['right']         = 83
        self.defaultMotorSpeed              = 70
        self.capturePeriod                  = 0.1
        
        self.LookUpTable                    = {}
        self.LookUpTable['time']            = []
        self.LookUpTable['Lmotor']          = []
        self.LookUpTable['Rmotor']          = []

        self.model                          = []
        self.interpreter                    = []
        self.tfLite_input_index             = []
        self.tfLite_output_index            = []
        
        self.pinInit()
        
        print('='*50)
        print("\t\tPybot Initialized")
        print('='*50)

def main():
    
    directory = "/home/phj07022/AI_CAR/video/train_new"
    
    bot = PYBOT(directory=directory)
    
    bot.setInterpreter()   
    bot.realTimeManeuvering()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
    GPIO.cleanup()

