# coding = utf-8

import argparse
import time
import datetime
import logging

from net import Net
from data_loader import CARLA_Data

from toolkit import ImitationLearning
from toolkit import VrgTransferSuite

try:
    sys.path.append("../CARLA_0.8.2/PythonClient/")
    # from carla import carla_server_pb2 as carla_protocol
    from carla.driving_benchmark import run_driving_benchmark
    # from carla.driving_benchmark.experiment_suites import CoRL2017
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

    argparser = argparse.ArgumentParser(description="test log")
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--city-name',
        metavar='C',
        default='Town01',
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log-name',
        metavar='T',
        default='test_log',
        help='The name of the log file to be created by the scripts'
    )

    argparser.add_argument(
        '--avoid-stopping',
        default=True,
        action='store_false',
        help=' Uses the speed prediction branch to avoid unwanted agent stops'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the given log name'
    )
    argparser.add_argument(
        '--weathers',
        nargs='+',
        type=int,
        default=[1],
        help='weather list 1:clear 3:wet, 6:rain 8:sunset'
    )
    argparser.add_argument(
        '--model-path',
        metavar='P',
        default='./models_save/checkpoint_best.pth',
        type=str,
        help='torch imitation learning model path (relative in model dir)'
    )
    argparser.add_argument(
        '--visualize',
        default=False,
        action='store_true',
        help='visualize the image and transfered image through tensorflow'
    )
    parser.add_argument('--gpu', default='1', type=str, help='GPU id to use.')

if __name__ == '__main__': 
    #获得参数，初始化日志
    global args
    args = parser.parse_args()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    #获取数据
    data = CARLA_Data(
        data_path = args.train_data_dir
        train_eval_flag = 'eval'
        batch_size = args.batch_size
    )
    data_loader = data.get_data_load()

    #指定使用的显卡
    GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICE']=args.gpu

  
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(filename="./test_log.log", level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    agent = ImitationLearningAgent(args.city_name,
                              args.avoid_stopping,
                              args.model_path,
                              args.gpu
                              )
    logging.info('Build agent success')

    # experiment_suites = CoRL2017(args.city_name)
    experiment_suites = VrgTransferSuite(args.city_name, args.weathers)
    logging.info('Build suites success')

    # Now actually run the driving_benchmark
    run_driving_benchmark(agent, experiment_suites, args.city_name,
                          args.log_name, args.continue_experiment,
                          args.host, args.port)

    



