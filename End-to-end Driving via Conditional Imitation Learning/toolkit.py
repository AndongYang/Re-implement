# coding = utf-8
import os
import random
import sys

import torch

from net import Net

try:
    sys.path.append("../CARLA_0.8.2/PythonClient/")
    from carla.agent import Agent
    from carla.carla_server_pb2 import Control
    from carla.sensor import Camera
    from carla.settings import CarlaSettings
    from carla.driving_benchmark.experiment import Experiment
    from carla.driving_benchmark.experiment_suites.experiment_suite import ExperimentSuite
except ImportError:
    raise RuntimeError('Cannot import carla abstract class')




def get_h5_list(file_dir):   
    all_file = []   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.h5':  
                all_file.append(os.path.join(root, file))           
                return all_file

#官网提供了测试代码的tensorflow版本https://github.com/carla-simulator/imitation-learning
##继承Agent抽象类并实现，之后将其传给run_driving_benchmark
class ImitationLearningAgent(Agent):
    def __init__(self, city_name,
                 avoid_stopping=True,
                 models_save="./models_save/checkpoint_best.pth",
                 image_cut=[115, 510],
                 gpu='1'):

        super().__init__()

        #指定使用的显卡
        GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICE']=gpu

        #与网络定义的输入大小一样
        self._image_size = (88, 200, 3)
        self._avoid_stopping = avoid_stopping

        self.model = Net()
        if torch.cuda.is_available():
            self.model.cuda()

        #恢复checkpoint
        os.makedirs('./models_save/', exist_ok=True)
        if os.path.isfile(models_save):
            checkpoint = torch.load(models_save)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("Loaded net")
        else:
            print("Load net failed!")
        
        self.model.eval()

        self._image_cut = image_cut

    def run_step(self, measurements, sensor_data, directions, target):

        control = self._compute_action(
            sensor_data['CameraRGB'].data,
            measurements.player_measurements.forward_speed,
            directions)

        return control
    
    def _compute_action(self, rgb_image, speed, direction=None):

        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])

        image_input = image_input.astype(np.float32)

        #为了节约空间之后调用浅拷贝变为tensor，所以这里要自己处理（H,W,C）->(N,C,H,W)
        image_input = np.expand_dims(
            np.transpose(image_input, (2, 0, 1)),
            axis=0)
        #标准化
        image_input = np.multiply(image_input, 1.0 / 255.0)

        #这里速度表示与数据集里的不统一，这里的25就是数据集中的90km/h
        speed = np.array([[speed]]).astype(np.float32) / 25.0
        direction = int(direction-2)

        steer, acc, brake = self._control_function(image_input,
                                                   speed,
                                                   direction)

        #这里开始代码与官方示例完全一样
        # This a bit biased, but is to avoid fake breaking

        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        if np.abs(steer) > 0.15:
            acc = acc * 0.4

        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0

        return control

    def _control_function(self, image_input, speed, control_input):
        img_ts = torch.from_numpy(image_input).cuda()
        speed_ts = torch.from_numpy(speed).cuda()

        with torch.no_grad():
            branches, pred_speed = self.model(img_ts, speed_ts)

        pred_result = branches[0][
            3*control_input:3*(control_input+1)].cpu().numpy()

        predicted_steers = (pred_result[0])

        predicted_acc = (pred_result[1])

        predicted_brake = (pred_result[2])

        #车速过慢而预测速度较大时，将车辆加速，可以避免车辆停止
        if self._avoid_stopping:
            predicted_speed = pred_speed.squeeze().item()
            real_speed = speed * 25.0

            real_predicted = predicted_speed * 25.0
            if real_speed < 2.0 and real_predicted > 3.0:

                predicted_acc = 1 * (5.6 / 25.0 - speed) + predicted_acc

                predicted_brake = 0.0

                predicted_acc = predicted_acc

        return predicted_steers, predicted_acc, predicted_brake


#继承ExperimentSuite抽象类并实现，之后将其传给run_driving_benchmark
class VrgTransferSuite(ExperimentSuite):

    def __init__(self, city_name, weathers):
        self._train_weathers = []
        self._test_weathers = weathers
        super(VrgTransferSuite, self).__init__(city_name)

    @property
    def train_weathers(self):
        return self._train_weathers

    @property
    def test_weathers(self):
        return self._test_weathers

    def _poses_town01(self):
        """
        Each matrix is a new task. We have all the four tasks
        """

        def _poses_straight():
            return [[36, 40], [39, 35], [110, 114], [7, 3], [0, 4],
                    [68, 50], [61, 59], [47, 64], [147, 90], [33, 87],
                    [26, 19], [80, 76], [45, 49], [55, 44], [29, 107],
                    [95, 104], [84, 34], [53, 67], [22, 17], [91, 148],
                    [20, 107], [78, 70], [95, 102], [68, 44], [45, 69]]

        def _poses_one_curve():
            return [[138, 17], [47, 16], [26, 9], [42, 49], [140, 124],
                    [85, 98], [65, 133], [137, 51], [76, 66], [46, 39],
                    [40, 60], [0, 29], [4, 129], [121, 140], [2, 129],
                    [78, 44], [68, 85], [41, 102], [95, 70], [68, 129],
                    [84, 69], [47, 79], [110, 15], [130, 17], [0, 17]]

        def _poses_navigation():
            return [[105, 29], [27, 130], [102, 87], [132, 27], [24, 44],
                    [96, 26], [34, 67], [28, 1], [140, 134], [105, 9],
                    [148, 129], [65, 18], [21, 16], [147, 97], [42, 51],
                    [30, 41], [18, 107], [69, 45], [102, 95], [18, 145],
                    [111, 64], [79, 45], [84, 69], [73, 31], [37, 81]]

        #test： return[[36,40]]
        
        return [_poses_straight(),
                _poses_one_curve(),
                _poses_navigation(),
                _poses_navigation()]

    def _poses_town02(self):

        def _poses_straight():
            return [[38, 34], [4, 2], [12, 10], [62, 55], [43, 47],
                    [64, 66], [78, 76], [59, 57], [61, 18], [35, 39],
                    [12, 8], [0, 18], [75, 68], [54, 60], [45, 49],
                    [46, 42], [53, 46], [80, 29], [65, 63], [0, 81],
                    [54, 63], [51, 42], [16, 19], [17, 26], [77, 68]]

        def _poses_one_curve():
            return [[37, 76], [8, 24], [60, 69], [38, 10], [21, 1],
                    [58, 71], [74, 32], [44, 0], [71, 16], [14, 24],
                    [34, 11], [43, 14], [75, 16], [80, 21], [3, 23],
                    [75, 59], [50, 47], [11, 19], [77, 34], [79, 25],
                    [40, 63], [58, 76], [79, 55], [16, 61], [27, 11]]

        def _poses_navigation():
            return [[19, 66], [79, 14], [19, 57], [23, 1],
                    [53, 76], [42, 13], [31, 71], [33, 5],
                    [54, 30], [10, 61], [66, 3], [27, 12],
                    [79, 19], [2, 29], [16, 14], [5, 57],
                    [70, 73], [46, 67], [57, 50], [61, 49], [21, 12],
                    [51, 81], [77, 68], [56, 65], [43, 54]]

        return [_poses_straight(),
                _poses_one_curve(),
                _poses_navigation(),
                _poses_navigation()
                ]

    def build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.
        """

        # We set the camera
        # This single RGB camera is used on every experiment

        camera = Camera('CameraRGB')
        #FOV is the horizontal field of view of the camera.
        camera.set(FOV=100)
        camera.set_image_size(800, 600)
        camera.set_position(2.0, 0.0, 1.4)
        camera.set_rotation(-15.0, 0, 0)

        if self._city_name == 'Town01':
            poses_tasks = self._poses_town01()
            vehicles_tasks = [0, 0, 0, 20]
            pedestrians_tasks = [0, 0, 0, 50]
        else:
            poses_tasks = self._poses_town02()
            vehicles_tasks = [0, 0, 0, 15]
            pedestrians_tasks = [0, 0, 0, 50]

        experiments_vector = []

        for weather in self.weathers:

            for iteration in range(len(poses_tasks)):
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                conditions = CarlaSettings()
                conditions.set(
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather
                )
                # Add all the cameras that were set for this experiments

                conditions.add_sensor(camera)

                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Task=iteration,
                    Repetitions=1
                )
                experiments_vector.append(experiment)

        return experiments_vector

