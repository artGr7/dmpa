import sys
import time
import numpy as np
import cv2

def draw(vis,x0,y0,x1,y1):
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2, 1)
        return True


class MedianFlowTracker(object):
    def __init__(self):
        self.lk_params = dict(winSize  = (11, 11),
                              maxLevel = 3,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

        self._atan2 = np.vectorize(np.math.atan2)

    def track(self, bb, prev, curr):

        self._n_samples = 100
        self._fb_max_dist = 1
        self._ds_factor = 0.95
        self._min_n_points = 10

        p0 = np.empty((self._n_samples, 2))
        p0[:, 0] = np.random.randint(bb[0], bb[2] + bb[0], self._n_samples)
        p0[:, 1] = np.random.randint(bb[1], bb[3] + bb[1], self._n_samples)

        p0 = p0.astype(np.float32)

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **self.lk_params)
        indx = np.where(st == 1)[0]
        p0 = p0[indx, :]
        p1 = p1[indx, :]
        p0r, st, err = cv2.calcOpticalFlowPyrLK(curr, prev, p1, None, **self.lk_params)

        if err is None:
            return None

        fb_dist = np.abs(p0 - p0r).max(axis=1)
        good = fb_dist < self._fb_max_dist

        err = err[good].flatten()
        if len(err) < self._min_n_points:
            return None

        indx = np.argsort(err)
        half_indx = indx[:len(indx) // 2]
        p0 = (p0[good])[half_indx]
        p1 = (p1[good])[half_indx]

        dx = np.median(p1[:, 0] - p0[:, 0])
        dy = np.median(p1[:, 1] - p0[:, 1])

        i, j = np.triu_indices(len(p0), k=1)
        pdiff0 = p0[i] - p0[j]
        pdiff1 = p1[i] - p1[j]

        p0_dist = np.sum(pdiff0 ** 2, axis=1)
        p1_dist = np.sum(pdiff1 ** 2, axis=1)
        ds = np.sqrt(np.median(p1_dist / (p0_dist + 2**-23)))
        ds = (1.0 - self._ds_factor) + self._ds_factor * ds

        dx_scale = (ds - 1.0) * 0.5 * (bb[3] - bb[1])
        dy_scale = (ds - 1.0) * 0.5 * (bb[2] - bb[0])
        bb_curr = (int(bb[0] + dx - dx_scale + 0.5),
                   int(bb[1] + dy - dy_scale + 0.5),
                   int(bb[2] + dx + dx_scale + 0.5),
                   int(bb[3] + dy + dy_scale + 0.5))
        
        if bb_curr[0] >= (bb_curr[0]+bb_curr[2]) or bb_curr[1] >= (bb_curr[1] + bb_curr[3]):
             return None

        bb_curr = (min(max(0, bb_curr[0]), curr.shape[1]),
                   min(max(0, bb_curr[1]), curr.shape[0]),
                   min(max(0, bb_curr[2]), curr.shape[1]),
                   min(max(0, bb_curr[3]), curr.shape[0]))

        return bb_curr

def run():
        prev, curr = None, None
        ret, frame = video.read()

        frame_height, frame_width = frame.shape[:2]
        frame = cv2.resize(frame, [frame_width//2, frame_height//2])

        output_video_path = f"tracked__{current_dir}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width // 2, frame_height // 2))

        if not ret:
            raise IOError('can\'t read the frame')
        bbox = cv2.selectROI(frame, False)
        tracker = MedianFlowTracker()
        while True:
            ret, frame = video.read()
            if not ret:
                end_time = time.time()
                if video.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
                    print(f"Время работы метода MedianFlow: {end_time - start_time:.5f} секунд")

                break
            frame = cv2.resize(frame, [frame_width//2, frame_height//2])

            prev, curr = curr, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev is not None and bbox is not None:
                bb = tracker.track(bbox, prev, curr)
                if bb is not None:
                    bbox = bb
            draw(frame,bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3])

            cv2.imshow("Tracking", frame)
            output.write(frame)

            ch = cv2.waitKey(1)
            if ch == 27 or ch in (ord('q'), ord('Q')):
                break
        video.release()
        cv2.destroyAllWindows()

map_dirs = {0:'example1.mp4',1:'example2.mp4',2:'example3.mp4',3:'example4.MP4',4:'example5.mp4'}
dir_num = int(sys.argv[1])
current_dir = map_dirs[dir_num]
video = cv2.VideoCapture(current_dir)
start_time = time.time()


run()
