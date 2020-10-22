import os
import sys
import glob
import time
import psutil
import platform as platform_
import argparse
import threading
from abc import abstractmethod

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import cv2
import numpy as np
import yaml

from openvino.inference_engine import IECore, StatusCode, WaitMode

inf_count = 0
disp_res = [1920,1080]

canvas = np.zeros([0], dtype=np.uint8)
abort_flag = False

framebuf_lock = threading.Lock()   # Lock object to control exclusive access from rendering and displaying (OGL)

class FullScreenCanvas:
    def __init__(self, winname='noname', shape=(1080, 1920, 3), full_screen=True):
        global canvas
        self.shape   = shape
        self.winname = winname
        canvas  = np.zeros((self.shape), dtype=np.uint8)

    def __del__(self):
        return

    def update(self):
        return

    def getROI(self, x0, y0, x1, y1):
        global canvas
        return canvas[y0:y1, x0:x1, :]

    def setROI(self, img, x0, y0, x1, y1):
        global canvas
        canvas[y0:y1, x0:x1, :] = img
    
    def displayOcvImage(self, img, x, y):
        h,w,_ = img.shape
        self.setROI(img, x, y, x+w, y+h)



class BenchmarkCanvas(FullScreenCanvas):
    def __init__(self, display_resolution=[1920,1080], full_screen=True):
        super().__init__('Benchmark', (display_resolution[1], display_resolution[0], 3), full_screen=full_screen)
        self.disp_res = display_resolution

        # Grid area to display inference result
        self.grid_col = 15
        self.grid_row = 7
        self.grid_area = [(0,0), (int(self.shape[1]), int(self.shape[0]*3/4))]
        self.grid_width  = int((self.grid_area[1][0]-self.grid_area[0][0])/self.grid_col)
        self.grid_height = int((self.grid_area[1][1]-self.grid_area[0][1])/self.grid_row)
        self.idx = 0    # current display pane index
        self.marker_img = np.full((self.grid_height, self.grid_width, 3), (64,64,64), dtype=np.uint8)

        # Status area
        self.sts_area = [ (0, int(self.shape[0]*3/4)), (self.shape[1]-1, self.shape[0]-1) ]

        # Calculate status grid size
        self.sts_grid_size = int(self.disp_res[0] / 80)


    def calcPaneCoord(self, paneIdx):
        col =  paneIdx  % self.grid_col
        row = (paneIdx // self.grid_col) % self.grid_row
        x0 = int(col * self.grid_width  + self.grid_area[0][0])
        y0 = int(row * self.grid_height + self.grid_area[0][1])
        x1 = int(x0 + self.grid_width)
        y1 = int(y0 + self.grid_height)
        return x0, y0, x1, y1

    def displayPane(self, ocvimg, idx=-1):
        if idx == -1:
            idx = self.idx
        self.idx = idx + 1
        x0, y0, x1, y1 = self.calcPaneCoord(idx)
        ocvimg = cv2.resize(ocvimg, (self.grid_width-2, self.grid_height-2), interpolation=cv2.INTER_NEAREST) # INTER_NEAREST for speed
        self.setROI(ocvimg, x0, y0, x1-2, y1-2)     # -2 to keep border lines

    def markCurrentPane(self, idx=-1):
        if idx == -1:
            idx = self.idx
        x0, y0, x1, y1 = self.calcPaneCoord(idx)
        self.setROI(self.marker_img, x0, y0, x1, y1)

    def displayLogo(self):
        global canvas
        stsY = self.grid_height * self.grid_row
        gs = self.sts_grid_size

        logo1 = os.path.join('logo', 'logo-classicblue-3000px.png')
        logo2 = os.path.join('logo', 'int-openvino-wht-3000.png')

        if os.path.isdir('logo'):
            tmpimg = cv2.imread(logo1)
            h = tmpimg.shape[0]
            tmpimg = cv2.resize(tmpimg, None, fx=(gs*4)/h, fy=(gs*4)/h, interpolation=cv2.INTER_LINEAR)    # Logo height = 3*gs
            tmpimg = cv2.cvtColor(tmpimg, cv2.COLOR_BGR2RGB)
            self.displayOcvImage(tmpimg, gs*26, stsY+gs*7)

            tmpimg = cv2.imread(logo2, cv2.IMREAD_UNCHANGED)
            b,g,r,alpha = cv2.split(tmpimg)
            tmpimg = cv2.merge([alpha,alpha,alpha])
            h = tmpimg.shape[0]
            tmpimg = cv2.resize(tmpimg, None, fx=(gs*4)/h, fy=(gs*4)/h, interpolation=cv2.INTER_LINEAR) 
            self.displayOcvImage(tmpimg, gs*32, stsY+gs*7)
        else:
            cv2.putText(canvas, 'OpenVINO', (gs*32, stsY+gs*9), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 5)

    def displayModel(self, modelName, device, batch, skip_count):
        global canvas
        _, name = os.path.split(modelName)
        name,_ = os.path.splitext(name)
        stsY = self.grid_height * self.grid_row
        gs = self.sts_grid_size
        ts = self.disp_res[0] / 960         # text size
        tt = self.disp_res[0] / 960         # text thickness
        tt = int(max(tt,1))
        txt = 'model: {} ({})'.format(name, device)
        cv2.putText(canvas, txt, (gs*1, stsY+gs* 8), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)
        txt = 'batch: {}, skip_frame {}'.format(batch, skip_count)
        cv2.putText(canvas, txt, (gs*1, stsY+gs*10), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)

    # elapse = sec
    def dispProgressBar(self, curItr, ttlItr, elapse, max_fps=100):
        global canvas
        def progressBar(img, x0, y0, x1, y1, val, color):
            val = min(100, val)
            xx = int(((x1-x0)*val)/100)+x0
            cv2.rectangle(img, (x0,y0), (xx,y1), color, -1)
            cv2.rectangle(img, (xx,y0), (x1,y1), (32,32,32), -1)
        img = canvas

        stsY  = self.grid_height * self.grid_row
        gs = self.sts_grid_size             # status pane grid size (dot)
        ts = self.disp_res[0] / 960         # text size
        tt = self.disp_res[0] / 960         # text thickness
        tt = int(max(tt,1))
        # erase numbers on the right
        cv2.rectangle(img, (gs*66, stsY), (self.disp_res[0]-1, self.disp_res[1]-1), (0,0,0), -1)

        cv2.putText(img, 'Progress:', (gs* 1, stsY+gs*2), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)
        progressBar(img, gs*8, stsY+gs*1, gs*64, stsY+gs*3, (curItr*100)/ttlItr, ( 32,255,255))
        cv2.putText(img, '{}/{}'.format(curItr,ttlItr)          , (gs*66, stsY+gs*2), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)

        try:
            fps = (curItr/elapse)
        except ZeroDivisionError:
            fps = 0
        cv2.putText(img, 'FPS:'     , (gs*1, stsY+gs*5), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)
        progressBar(img, gs*8, stsY+gs*4, gs*64, stsY+gs*6, (fps*100)/max_fps, (  0,255,128))
        cv2.putText(img, '{:5.2f} inf/sec'.format(fps), (gs*66, stsY+gs*5), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)

        cv2.putText(img, 'Time: {:5.1f} sec'.format(elapse), (gs*66, stsY+gs*8), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)


class benchmark():
    def __init__(self, model, device='CPU', nireq=4, config=None):
        global disp_res
        self.config = config
        self.read_labels()
        base, ext = os.path.splitext(model)
        self.ie = IECore()
        print('reading the model...', end='', flush=True)
        self.net = self.ie.read_network(base+'.xml', base+'.bin')
        print('done')
        if 'batch' in self.config['model_config']:
            self.batch = self.config['model_config']['batch']
        else:
            self.batch = 1
        self.net.batch_size = self.batch
        self.inputBlobName  = next(iter(self.net.input_info))
        self.outputBlobName = next(iter(self.net.outputs)) 
        self.inputShape  = self.net.input_info  [self.inputBlobName ].tensor_desc.dims
        self.outputShape = self.net.outputs     [self.outputBlobName].shape

        # Setup network configuration parameters
        print('*** SET CONFIGURATION') 
        network_cfg = self.config['plugin_config']
        for device in network_cfg:
            cfg_items = network_cfg[device]
            for cfg in cfg_items:
                self.ie.set_config(cfg, device)
                print('   ', cfg, device)

        print('loading the model to the plugin...', end='', flush=True)
        self.exenet = self.ie.load_network(self.net, device, num_requests=nireq)
        print('done')
        self.nireq = nireq

        disp_res = [ int(i) for i in self.config['display_resolution'].split('x') ]  # [1920,1080]
        self.canvas = BenchmarkCanvas(display_resolution=disp_res, full_screen=self.config['full_screen'])
        self.skip_count = self.config['display_skip_count']
        self.canvas.displayLogo()
        self.canvas.displayModel(model, device, self.batch, self.skip_count)
        self.infer_slot = [ [False, 0] for i in range(self.nireq) ]   # [Inuse flag, ocvimg index]
        self.draw_requests = []
        self.draw_requests_lock = threading.Lock()

    def read_labels(self):
        if 'label_file' in self.config['model_config']:
            label_file = self.config['model_config']['label_file']
            with open(label_file, 'rt') as f:
                self.labels = [ line.rstrip('\n').split(',')[0] for line in f ]
        else:
            self.labels = None


    def preprocessImages(self, files):
        print('preprocessing image files...', end='', flush=True)
        self.blobImages = []
        self.ocvImages = []
        for f in files:
            ocvimg = cv2.imread(f)
            ocvimg = cv2.cvtColor(ocvimg, cv2.COLOR_BGR2RGB)    # Assuming to use OpenCL to display the frame buffer (RGB)
            # preprocess for inference
            blobimg = cv2.resize(ocvimg, (self.inputShape[-1], self.inputShape[-2]), interpolation=cv2.INTER_LINEAR)
            blobimg = blobimg.transpose((2,0,1))
            blobimg = blobimg.reshape(self.inputShape[1:])
            self.blobImages.append(blobimg)
            # scaling for image to display in the panes
            ocvimg = cv2.resize(ocvimg, (self.canvas.grid_width-2, self.canvas.grid_height-2), interpolation=cv2.INTER_LINEAR)
            self.ocvImages.append(ocvimg)
        print('done')


    def run(self, niter=10, nireq=4, files=None, max_fps=100):
        global abort_flag, framebuf_lock
        print('*** CURRENT CONFIGURATION')
        met_keys = self.exenet.get_metric('SUPPORTED_METRICS')
        cfg_keys = self.exenet.get_metric('SUPPORTED_CONFIG_KEYS')
        for key in cfg_keys:
            print('   ', key, self.exenet.get_config(key))

        niter = (niter//self.batch)*self.batch + (self.batch if niter % self.batch else 0)  # tweak number of iteration for batch inferencing
        self.inf_count = 0

        framebuf_lock.acquire()
        self.canvas.dispProgressBar(curItr=0, ttlItr=niter, elapse=0, max_fps=max_fps)
        framebuf_lock.release()
        time.sleep(1)

        # Do inference
        inf_kicked = 0
        inf_done   = 0
        start = time.perf_counter()
        while inf_done < niter:
            # get idle infer request slot
            self.exenet.wait(num_requests=1, timeout=WaitMode.RESULT_READY)
            request_id = self.exenet.get_idle_request_id()
            infreq = self.exenet.requests[request_id]

            # if slot has been already in use, process the infer result
            if self.infer_slot[request_id][0] == True:
                inf_done += self.batch
                ocvIdx = self.infer_slot[request_id][1]   # OCV image index
                res = infreq.output_blobs[self.outputBlobName].buffer[0].ravel()
                self.infer_slot[request_id] = [False, 0]
            else:
                ocvIdx = -1

            # kick inference
            dataIdx = inf_kicked % len(self.blobImages)
            self.infer_slot[request_id] = [True, dataIdx]
            infreq.async_infer(inputs={ self.inputBlobName : self.blobImages[dataIdx] } )
            inf_kicked += 1

            # deferred postprocess & rendering
            if ocvIdx != -1:
                if ocvIdx % self.skip_count == 0:
                    ocvimg = self.ocvImages[ocvIdx].copy()
                    idx = (res.argsort())   #[::-1]
                    txt = self.labels[idx[-1]]
                    cv2.putText(ocvimg, txt, (0, ocvimg.shape[-2]//2), cv2.FONT_HERSHEY_PLAIN, 2, (  0,  0,  0), 3 )
                    cv2.putText(ocvimg, txt, (0, ocvimg.shape[-2]//2), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2 )
                    self.canvas.displayPane(ocvimg)

                    if ocvIdx % (self.skip_count*5) == 0:
                        framebuf_lock.acquire()
                        self.canvas.dispProgressBar(curItr=inf_done, ttlItr=niter, elapse=time.perf_counter()-start, max_fps=max_fps)
                        framebuf_lock.release()
                        self.canvas.markCurrentPane()

            if abort_flag == True:
                break

        end = time.perf_counter()

        if abort_flag == False:
            # Display the rsult
            print('Time: {:8.2f} sec, Throughput: {:8.2f} inf/sec'.format(end-start, niter/(end-start)))
            framebuf_lock.acquire()
            self.canvas.dispProgressBar(curItr=niter, ttlItr=niter, elapse=end-start, max_fps=max_fps)
            framebuf_lock.release()
            glutPostRedisplay()
            time.sleep(5)
        else:
            print('Program aborted')

        abort_flag = True




def draw():
    global canvas, framebuf_lock
    h, w = canvas.shape[:2]
    framebuf_lock.acquire()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, canvas)
    framebuf_lock.release()

    #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #glColor3f(1.0, 1.0, 1.0)

    #glEnable(GL_TEXTURE_2D)
    # Set texture map method
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # draw a square
    glBegin(GL_QUADS) 
    glTexCoord2d(0.0, 1.0)
    glVertex3d(-1.0, -1.0,  0.0)
    glTexCoord2d(1.0, 1.0)
    glVertex3d( 1.0, -1.0,  0.0)
    glTexCoord2d(1.0, 0.0)
    glVertex3d( 1.0,  1.0,  0.0)
    glTexCoord2d(0.0, 0.0)
    glVertex3d(-1.0,  1.0,  0.0)
    glEnd()
    glFlush();
    glutSwapBuffers()

def init():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    # Enable texture map
    glEnable(GL_TEXTURE_2D)
    # Set texture map method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

def idle():
    return

def timer(val):
    global abort_flag
    if abort_flag == True:
        sys.exit(0)
    glutPostRedisplay()
    glutTimerFunc(33*1, timer, 0)

def reshape(w, h):
    global disp_res
    glViewport(0, 0, w, h)
    glLoadIdentity()
    glOrtho(-w / disp_res[0], w / disp_res[0], -h / disp_res[1], h / disp_res[1], -1.0, 1.0)

def keyboard(key, x, y):
    global abort_flag
    key = key.decode('utf-8')
    if key == 'q':
        abort_flag = True



def main():
    global abort_flag, disp_res

    abort_flag = False

    # set process priority
    proc = psutil.Process(os.getpid())
    if platform_.system() == 'Windows':
        proc.nice(psutil.HIGH_PRIORITY_CLASS) # HIGH_PRIORITY_CLASS, ABOVE_NORMAL_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS, BELOW_NORMAL_PRIORITY_CLASS, IDLE_PRIORITY_CLASS
    else:
        proc.nice(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='default.yml', type=str, help='Input configuration file (YAML)')
    args = parser.parse_args()

    # Read YAML configuration file
    with open(args.config, 'rt') as f:
        config = yaml.safe_load(f)
    #for key, val in config.items():
    #    print(key, val)

    image_src = config['image_source_dir']
    files = glob.glob(os.path.join(image_src, '*.'+config['image_data_extension']))
    if len(files)==0:
        print('ERROR: No input images are found. Please check \'image_source_dir\' setting in the YAML configuration file.')
        return 1

    model = config['xml_model_path']
    if not os.path.isfile(model):
        print('ERROR: Model file is not found. ({})'.format(model))
        return 1
    model_type = config['model_config']['type']
    bm = benchmark(model, device=config['target_device'], config=config)

    bm.preprocessImages(files)
    th = threading.Thread(target=bm.run, kwargs={
        'niter'  :config['iteration'], 
        'nireq'  :config['num_requests'], 
        'max_fps':config['fps_max_value']})
    th.setDaemon(True)
    th.start()

    glutInitWindowPosition(0, 0)
    glutInitWindowSize(*disp_res)
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE )
    if config['full_screen'] == True:
        glutEnterGameMode()
    else:
        glutCreateWindow('performance demo')
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    init()
    glutIdleFunc(None)
    glutTimerFunc(33, timer, 0)
    glutMainLoop()

    return 0

if __name__ == '__main__':
    sys.exit(main())
