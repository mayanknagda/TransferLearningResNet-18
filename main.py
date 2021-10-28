## ML Pipeline
# collect data
# process data
# prepare data
# model selection
# training
# testing
# process data
# visualization
# logging

import data
import model
import train
import utils
import argparse

def main(args):
    utils.seed_experiment(args.seed)
    args = data.get_data(args)
    args.model = model.ResNet(args)
    args = utils.get_optim(args)
    args = utils.get_loss_fn(args)
    args = train.fit(args)
    args = utils.test(args)
    utils.logoutput(args)
    print('Finished Training, Please check output folder and the Jupyter Notebook')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Please input your model arguments.')
    parser.add_argument('epochs', help='Insert total number of epochs.', type=int, default=10)
    parser.add_argument('batch_size', help='Insert number of batch_size.', type=int, default=128)
    parser.add_argument('lr', help='Insert your learning rate.', type=float, default=1e-1)
    parser.add_argument('seed', help='Insert your random seed.', type=int, default=5)
    parser.add_argument('config_type', help='Insert your config type here.', type=str, default='a')
    parser.add_argument('optim_type', help='Which optimizer you wanna use? options are (SGD / Adam)', type=str, default='SGD')
    parser.add_argument('data_location', help='Location of your data.', type=str, default='/Users/mayanknagda/Documents/Work/Self/projects/dataset/')
    parser.add_argument('out_features_size', help='Out features size', type=int, default=100)
    parser.add_argument('device', help='Device on which to train the model', type=str, default='cuda:0')
    parser.add_argument('output_dir', help='Name of output folder for this particular run', type=str, default='folder1')
    #args = parser.parse_args()
    args = argparse.Namespace(epochs = 10,
                              batch_size = 64,
                              lr = 1e-1,
                              seed = 5,
                              config_type = 'e',
                              optim_type = 'SGD',
                              data_location = '/Users/mayanknagda/Documents/Work/Self/projects/pretrained_transformer_test_task',
                              out_features_size = 100,
                              device='cuda:0',
                              output_dir = 'config_e')
    main(args)