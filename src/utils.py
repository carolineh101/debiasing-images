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