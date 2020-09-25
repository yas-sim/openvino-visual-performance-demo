import os
import sys
import glob
import time
import argparse

import cv2
import numpy as np
import yaml

from openvino.inference_engine import IECore

inf_count = 0

class FullScreenCanvas:
    def __init__(self, winname='noname', shape=(1080, 1920, 3), full_screen=True):
        self.shape   = shape
        self.winname = winname
        self.canvas  = np.zeros((self.shape), dtype=np.uint8)
        if full_screen:
            cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def __del__(self):
        cv2.destroyAllWindows()

    def update(self):
        cv2.imshow(self.winname, self.canvas)

    def getROI(self, x0, y0, x1, y1):
        return self.canvas[y0:y1, x0:x1, :]

    def setROI(self, img, x0, y0, x1, y1):
        self.canvas[y0:y1, x0:x1, :] = img
    
    def displayOcvImage(self, img, x, y):
        h,w,_ = img.shape
        self.setROI(img, x, y, x+w, y+h)


class BenchmarkCanvas(FullScreenCanvas):
    def __init__(self, display_resolution=[1920,1080], full_screen=True):
        super().__init__('Benchmark', (display_resolution[1], display_resolution[0], 3), full_screen=full_screen)
        self.disp_res = display_resolution

        # Grid area to display inference result
        self.grid_col = 6
        self.grid_row = 3
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
        x1 -= 2
        y1 -= 2
        img = cv2.resize(ocvimg, (self.grid_width-2, self.grid_height-2))
        self.setROI(img, x0, y0, x1, y1)

    def markCurrentPane(self, idx=-1):
        if idx == -1:
            idx = self.idx
        x0, y0, x1, y1 = self.calcPaneCoord(idx)
        self.setROI(self.marker_img, x0, y0, x1, y1)

    def displayLogo(self):
        stsY = self.grid_height * self.grid_row
        gs = self.sts_grid_size

        tmpimg = cv2.imread(os.path.join('logo', 'logo-classicblue-3000px.png'))
        h = tmpimg.shape[0]
        tmpimg = cv2.resize(tmpimg, None, fx=(gs*4)/h, fy=(gs*4)/h)    # Logo height = 3*gs
        self.displayOcvImage(tmpimg, gs*26, stsY+gs*7)

        tmpimg = cv2.imread(os.path.join('logo', 'int-openvino-wht-3000.png'), cv2.IMREAD_UNCHANGED)
        b,g,r,alpha = cv2.split(tmpimg)
        tmpimg = cv2.merge([alpha,alpha,alpha])
        h = tmpimg.shape[0]
        tmpimg = cv2.resize(tmpimg, None, fx=(gs*4)/h, fy=(gs*4)/h) 
        self.displayOcvImage(tmpimg, gs*32, stsY+gs*7)

    def displayModel(self, modelName):
        _, name = os.path.split(modelName)
        name,_ = os.path.splitext(name)
        stsY = self.grid_height * self.grid_row
        gs = self.sts_grid_size
        ts = self.disp_res[0] / 960         # text size
        tt = self.disp_res[0] / 960         # text thickness
        tt = int(max(tt,1))
        cv2.putText(self.canvas, 'model: {}'.format(name), (gs*1, stsY+gs*8), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)

    # elapse = sec
    def dispProgressBar(self, curItr, ttlItr, elapse, max_fps=100):
        def progressBar(img, x0, y0, x1, y1, val, color):
            xx = int(((x1-x0)*val)/100)+x0
            cv2.rectangle(img, (x0,y0), (xx,y1), color, -1)
            cv2.rectangle(img, (xx,y0), (x1,y1), (64,64,64), -1)
        img = self.canvas

        stsY  = self.grid_height * self.grid_row
        gs = self.sts_grid_size             # status pane grid size (dot)
        ts = self.disp_res[0] / 960         # text size
        tt = self.disp_res[0] / 960         # text thickness
        tt = int(max(tt,1))
        # erase numbers on the right
        cv2.rectangle(img, (gs*66, stsY), (self.disp_res[0]-1, self.disp_res[1]-1), (0,0,0), -1)

        cv2.putText(img, 'Progress:', (gs* 1, stsY+gs*2), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)
        progressBar(img, gs*8, stsY+gs*1, gs*65, stsY+gs*3, (curItr*100)/ttlItr, (255,0,64))
        cv2.putText(img, '{}/{}'.format(curItr,ttlItr)          , (gs*66, stsY+gs*2), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)

        cv2.putText(img, 'FPS:'     , (gs*1, stsY+gs*5), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)
        progressBar(img, gs*8, stsY+gs*4, gs*65, stsY+gs*6, (curItr*100/elapse)/max_fps, (128,255,0))
        cv2.putText(img, '{:5.2f} inf/sec'.format(curItr/elapse), (gs*66, stsY+gs*5), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)

        cv2.putText(img, 'Time: {:5.1f}'.format(elapse), (gs*66, stsY+gs*8), cv2.FONT_HERSHEY_PLAIN, ts, (255,255,255), tt)


class benchmark():
    def __init__(self, model, device='CPU', nireq=4, config=None):
        self.config = config
        with open(config['label_file'], 'rt') as f:
            self.labels = [ line.rstrip('\n') for line in f ]

        base, ext = os.path.splitext(model)
        self.ie = IECore()
        self.net = self.ie.read_network(base+'.xml', base+'.bin')
        self.inputBlobName  = next(iter(self.net.input_info))
        self.outputBlobName = next(iter(self.net.outputs)) 
        self.inputShape  = self.net.input_info  [self.inputBlobName ].tensor_desc.dims
        self.outputShape = self.net.outputs     [self.outputBlobName].shape

        # Setup network configuration parameters
        print('*** SET CONFIGURATION') 
        network_cfg = config['plugin_config']
        if device in network_cfg:
            cfg_items = network_cfg[device]
            for cfg in cfg_items:
                self.ie.set_config(cfg, device)
                print('   ', cfg, device)

        self.exenet = self.ie.load_network(self.net, device, num_requests=nireq)
        self.nireq = nireq
        self.inf_count = 0

        disp_res = [ int(i) for i in config['display_resolution'].split('x') ]  # [1920,1080]
        self.canvas = BenchmarkCanvas(display_resolution=disp_res, full_screen=config['full_screen'])
        self.inf_slot = [ None for i in range(self.nireq) ]
        self.inf_slot_inuse = [ False for i in range(self.nireq) ]
        self.skip_count = config['display_skip_count']
        self.canvas.displayLogo()
        self.canvas.displayModel(model)

    def preprocessImages(self, files):
        self.blobImages = []
        self.ocvImages = []
        for f in files:
            img = cv2.imread(f)
            img = cv2.resize(img, (self.inputShape[-1], self.inputShape[-2]))
            blobimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            blobimg = blobimg.transpose((2,0,1))
            blobimg = blobimg.reshape(self.inputShape)
            self.ocvImages.append(img)
            self.blobImages.append(blobimg)


    def callback(self, status, pydata):
        self.inf_count += 1
        if self.inf_count % self.skip_count == 0:
            ireq = self.exenet.requests[pydata]
            ocvimg = self.inf_slot[pydata]
            res = ireq.output_blobs[self.outputBlobName].buffer.ravel()
            idx = (res.argsort())[::-1]
            txt = self.labels[idx[0]]
            cv2.putText(ocvimg, txt, (0, ocvimg.shape[-2]//2), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 5 )
            cv2.putText(ocvimg, txt, (0, ocvimg.shape[-2]//2), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2 )
            self.canvas.displayPane(ocvimg)
        self.inf_slot_inuse[pydata] = False


    def run(self, niter=10, nireq=4, files=None, max_fps=100):

        print('*** CURRENT CONFIGURATION')
        met_keys = self.exenet.get_metric('SUPPORTED_METRICS')
        cfg_keys = self.exenet.get_metric('SUPPORTED_CONFIG_KEYS')
        for key in cfg_keys:
            print('   ', key, self.exenet.get_config(key))

        self.inf_count = 0
        start = time.perf_counter()
        # Do inference
        for i in range(niter):
            request_id = self.exenet.get_idle_request_id()
            if request_id == -1:
                self.exenet.wait(num_requests=1, timeout=-1)
                request_id = self.exenet.get_idle_request_id()
            while self.inf_slot_inuse[request_id] == True:
                pass
            self.inf_slot_inuse[request_id] = True
            infreq = self.exenet.requests[request_id]
            dataIdx = i % len(self.blobImages)
            self.inf_slot[request_id] = self.ocvImages[dataIdx]
            infreq.set_completion_callback(self.callback, request_id)
            infreq.async_infer(inputs={ self.inputBlobName : self.blobImages[dataIdx] } )

            if i % self.skip_count == 0:
                self.canvas.dispProgressBar(curItr=i, ttlItr=niter, elapse=time.perf_counter()-start, max_fps=max_fps)
                self.canvas.markCurrentPane()
                cv2.imshow(self.canvas.winname, self.canvas.canvas)
                key = cv2.waitKey(1)
                if key == 27:
                    break
        # Wait for completion of all infer requests
        while self.inf_count < niter and key != 27:
            pass
        end = time.perf_counter()
        print('Time: {:8.2f} sec, Throughput: {:8.2f} inf/sec'.format(end-start, niter/(end-start)))
        self.canvas.dispProgressBar(curItr=niter, ttlItr=niter, elapse=end-start)
        cv2.imshow(self.canvas.winname, self.canvas.canvas)
        cv2.waitKey(5 * 1000)    # wait for 5 sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='default.yml', type=str, help='Input configuration file (YAML)')
    args = parser.parse_args()

    # Read YAML configuration file
    with open(args.config, 'rt') as f:
        config = yaml.safe_load(f)
    #print(config)

    image_src = config['image_source_dir']
    files = glob.glob(os.path.join(image_src, '*.'+config['image_data_extension']))

    model = config['xml_model_path']
    bm = benchmark(model, device=config['target_device'], config=config)
    bm.preprocessImages(files)
    bm.run(niter=config['iteration'], nireq=config['num_requests'], max_fps=config['fps_max_value'])

if __name__ == '__main__':
    main()
