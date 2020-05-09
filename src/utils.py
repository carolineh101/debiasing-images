import torch

def getDevice(gpu_id=None):

    #CPU
    device = 'cpu'

    #GPU
    if torch.cuda.is_available():

        device = 'cuda'

        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        cuda_str = 'Using CUDA '
        for i in range(0, ng):

            if i == gpu_id:
                device += ':' + str(i)

            if i == 1:
                cuda_str = ' ' * len(cuda_str)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                (cuda_str, i, x[i].name, x[i].total_memory / c))
            
    print('Device:', device)
    return device


def calculateAccuracy(outputs, targets, threshold=0.5):
    """
        Calculates the average accuracy.
          outputs: Tensor
          targets: Tensor
          threshold: float
    """

    preds = torch.sigmoid(outputs) > threshold

    average_accuracy = (preds == targets).sum() * 1.0 / (targets.size(0) * targets.size(1))

    return average_accuracy

def calculateEqualityGap(outputs, targets):
    return 0.0

def calculateParityGap(outputs, targets):
    return 0.0

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count