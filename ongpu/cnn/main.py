from cnn import *
from layers import *
import cupy as cp
import numpy as np
import pickle
import cProfile

dogs = 0

if __name__ == "__main__" and dogs == 1:
    
    c = CNN()

    c.add_conv_layer(size=3, amount=16, channels=3)
    
    first_res_block = []
    first_res_block.append(ReluLayer())
    first_res_block.append(BN_layer((16, 224, 224)))
    first_res_block.append(ConvLayer(amount=16, channels=16))
    
    c.add_res_block(*first_res_block)
    del first_res_block

    c.add_relu_layer()
    c.add_bn_layer((16, 224, 224))
    c.add_pool_layer()
    c.add_conv_layer(amount=32, channels=16)

    second_res_block = []
    second_res_block.append(ReluLayer())
    second_res_block.append(BN_layer((32, 112, 112)))
    second_res_block.append(ConvLayer(amount=32, channels=32))

    c.add_res_block(*second_res_block)
    del second_res_block

    c.add_relu_layer()
    c.add_bn_layer((32, 112, 112))
    c.add_pool_layer()
    c.add_conv_layer(amount=64, channels=32)

    """
    c.add_relu_layer()
    c.add_bn_layer((64, 56, 56))
    c.add_conv_layer(amount=64, channels=64)
    """

    third_res_block = []
    third_res_block.append(ReluLayer())
    third_res_block.append(BN_layer((64, 56, 56)))
    third_res_block.append(ConvLayer(amount=64, channels=64))
    third_res_block.append(ReluLayer())
    third_res_block.append(BN_layer((64, 56, 56)))
    third_res_block.append(ConvLayer(amount=64, channels=64))
    third_res_block.append(BN_layer((64, 56, 56)))
    third_res_block.append(ConvLayer(amount=64, channels=64))
    third_res_block.append(ReluLayer())
    third_res_block.append(BN_layer((64, 56, 56)))
    third_res_block.append(ConvLayer(amount=64, channels=64))

    c.add_res_block(*third_res_block)
    del third_res_block

    c.add_relu_layer()
    c.add_bn_layer((64, 56, 56))
    c.add_pool_layer()
    c.add_conv_layer(amount=128, channels=64)
    
    """
    c.add_relu_layer()
    c.add_bn_layer((128, 28, 28))
    c.add_conv_layer(amount=128, channels=128)
    """


    fourth_res_block = []
    fourth_res_block.append(ReluLayer())
    fourth_res_block.append(BN_layer((128, 28, 28)))
    fourth_res_block.append(ConvLayer(amount=128, channels=128))
    fourth_res_block.append(ReluLayer())
    fourth_res_block.append(BN_layer((128, 28, 28)))
    fourth_res_block.append(ConvLayer(amount=128, channels=128))
    fourth_res_block.append(ReluLayer())
    fourth_res_block.append(BN_layer((128, 28, 28)))
    fourth_res_block.append(ConvLayer(amount=128, channels=128))

    c.add_res_block(*fourth_res_block)
    del fourth_res_block

    c.add_relu_layer()
    c.add_bn_layer((128, 28, 28))
    c.add_conv_layer(amount=128, channels=128)

    fourth_res_block = []
    fourth_res_block.append(ReluLayer())
    fourth_res_block.append(BN_layer((128, 28, 28)))
    fourth_res_block.append(ConvLayer(amount=128, channels=128))
    fourth_res_block.append(ReluLayer())
    fourth_res_block.append(BN_layer((128, 28, 28)))
    fourth_res_block.append(ConvLayer(amount=128, channels=128))
    fourth_res_block.append(ReluLayer())
    fourth_res_block.append(BN_layer((128, 28, 28)))
    fourth_res_block.append(ConvLayer(amount=128, channels=128))

    c.add_res_block(*fourth_res_block)
    del fourth_res_block

    c.add_relu_layer()
    c.add_bn_layer((128, 28, 28))
    c.add_pool_layer()
    c.add_conv_layer(amount=256, channels=128)
    

    """
    c.add_relu_layer()
    c.add_bn_layer((256, 14, 14))
    c.add_conv_layer(amount=256, channels=256)
    """

    fifth_res_block = []
    fifth_res_block.append(ReluLayer())
    fifth_res_block.append(BN_layer((256, 14, 14)))
    fifth_res_block.append(ConvLayer(amount=256, channels=256))
    fifth_res_block.append(ReluLayer())
    fifth_res_block.append(BN_layer((256, 14, 14)))
    fifth_res_block.append(ConvLayer(amount=256, channels=256))    
    fifth_res_block.append(ReluLayer())
    fifth_res_block.append(BN_layer((256, 14, 14)))
    fifth_res_block.append(ConvLayer(amount=256, channels=256))

    c.add_res_block(*fifth_res_block)
    del fifth_res_block

    c.add_relu_layer()
    c.add_bn_layer((256, 14, 14))
    c.add_conv_layer(amount=256, channels=256)

    fifth_res_block = []
    fifth_res_block.append(ReluLayer())
    fifth_res_block.append(BN_layer((256, 14, 14)))
    fifth_res_block.append(ConvLayer(amount=256, channels=256))
    fifth_res_block.append(ReluLayer())
    fifth_res_block.append(BN_layer((256, 14, 14)))
    fifth_res_block.append(ConvLayer(amount=256, channels=256))    
    fifth_res_block.append(ReluLayer())
    fifth_res_block.append(BN_layer((256, 14, 14)))
    fifth_res_block.append(ConvLayer(amount=256, channels=256))

    c.add_res_block(*fifth_res_block)
    del fifth_res_block

    c.add_relu_layer()
    c.add_bn_layer((256, 14, 14))
    c.add_pool_layer()
    c.add_conv_layer(amount=512, channels=256)
    
    """
    c.add_relu_layer()
    c.add_bn_layer((512, 7, 7))
    c.add_conv_layer(amount=512, channels=512)
    """

    sixth_res_block = []
    sixth_res_block.append(ReluLayer())
    sixth_res_block.append(BN_layer((512, 7, 7)))
    sixth_res_block.append(ConvLayer(amount=512, channels=512))
    sixth_res_block.append(ReluLayer())
    sixth_res_block.append(BN_layer((512, 7, 7)))
    sixth_res_block.append(ConvLayer(amount=512, channels=512))
    sixth_res_block.append(ReluLayer())
    sixth_res_block.append(BN_layer((512, 7, 7)))
    sixth_res_block.append(ConvLayer(amount=512, channels=512))

    c.add_res_block(*sixth_res_block)
    del sixth_res_block

    c.add_relu_layer()
    c.add_bn_layer((512, 7, 7))
    c.add_conv_layer(size=3, amount=512, pad=0, channels=512)

    res_block = []
    res_block.append(ReluLayer())
    res_block.append(BN_layer((512, 5, 5)))
    res_block.append(ConvLayer(amount=512, channels=512))
    res_block.append(ReluLayer())
    res_block.append(BN_layer((512, 5, 5)))
    res_block.append(ConvLayer(amount=512, channels=512))
    res_block.append(ReluLayer())
    res_block.append(BN_layer((512, 5, 5)))
    res_block.append(ConvLayer(amount=512, channels=512))

    del res_block

    c.add_relu_layer()
    c.add_bn_layer((512, 5, 5))
    c.add_conv_layer(size=3, amount=512, channels=512)
 
    res_block = []
    res_block.append(ReluLayer())
    res_block.append(BN_layer((512, 5, 5)))
    res_block.append(ConvLayer(amount=512, channels=512))
    res_block.append(ReluLayer())
    res_block.append(BN_layer((512, 5, 5)))
    res_block.append(ConvLayer(amount=512, channels=512))
    res_block.append(ReluLayer())
    res_block.append(BN_layer((512, 5, 5)))
    res_block.append(ConvLayer(amount=512, channels=512))

    c.add_relu_layer()
    c.add_bn_layer((512, 5, 5))
    c.add_conv_layer(size=3, pad=0, amount=512, channels=512)   

    c.add_fc_layer(512*3*3, 2056, 1)
    c.add_relu_layer()
    c.add_bn_layer((2056,))


    c.add_fc_layer(2056, 1024)
    c.add_relu_layer()
    c.add_bn_layer((1024,))


    c.add_fc_layer(1024, 7, 0)
    c.add_softmax_layer()

    dataset_path = "/home/yaniv/dog-breed/"
    x, y = get_dogs(dataset_path)
    xb = x.reshape((-1, 32, 3, 224, 224))
    yb = y.reshape((-1, 32, 7))

    n = int(xb.shape[0] * 0.7)
    (tx, ty), (vx, vy) = (xb[:n], yb[:n]), (xb[n:], yb[n:])

    #cProfile.run('c.sgd(1, tx, ty, vx, vy, e0=1e-3, wd=1e-8, k=2500)')
    try:
        with open('file', 'w') as f:
            cProfile.run('j, jv = c.adam_momentum(10, tx[::], ty[::], vx[::], vy[::], e=1e-4, wd=1e-8, k=32)')

            #j, jv = c.adam_momentum(10, tx[::], ty[::], vx[::], vy[::], e=1e-4, wd=0, k=32)
    except KeyboardInterrupt:
        with open('model-12-01.pickle', 'wb') as f:
            pickle.dump(c, f)
        raise

    with open('model-12-01.pickle', 'wb') as f:
        pickle.dump(c, f)
    
    y = cp.load(dataset_path + 'all-labels-grouped-7.npy')
    
    print('training test: ')
    c.test(x[:2048].reshape((-1 , 32, 3, 224, 224)), y[:2048].reshape((-1, 32)))
 
    print('validation test: ')
    c.test(x[-2048:].reshape((-1 , 32, 3, 224, 224)), y[-2048:].reshape((-1, 32)))

    fig, axs = plt.subplots(2)
    axs[0].plot(range(len(j)), j)
    axs[1].plot(range(len(jv)), jv)

    plt.show()

if __name__ == '__main__' and dogs==0:
    dataset_path = '/home/yaniv/drawings/'
    c = CNN()

    c.add_conv_layer(amount=16, channels=1)
    c.add_relu_layer()
    c.add_bn_layer((16, 28, 28))

    c.add_conv_layer(amount=16, channels=16)
    c.add_relu_layer()
    c.add_bn_layer((16, 28, 28))

    c.add_pool_layer()

    c.add_conv_layer(amount=32, channels=16)
    c.add_relu_layer()
    c.add_bn_layer((32, 14, 14))

    c.add_conv_layer(amount=32, channels=32)
    c.add_relu_layer()
    c.add_bn_layer((32, 14, 14))

    c.add_pool_layer()

    c.add_conv_layer(amount=64, channels=32)
    c.add_relu_layer()
    c.add_bn_layer((64, 7, 7))

    c.add_conv_layer(amount=64, channels=64)
    c.add_relu_layer()
    c.add_bn_layer((64, 7, 7))

    c.add_fc_layer(3136, 3136, 1)
    c.add_relu_layer()
    c.add_bn_layer((3136,))

    c.add_fc_layer(3136, 2048)
    c.add_relu_layer()
    c.add_bn_layer((2048,))

    c.add_fc_layer(2048, 2048)
    c.add_relu_layer()
    c.add_bn_layer((2048,))

    c.add_fc_layer(2048, 1024)
    c.add_relu_layer()
    c.add_bn_layer((1024,))

    c.add_fc_layer(1024, 9)
    c.add_softmax_layer()

    x, y = np.load(dataset_path + 'x.npy').reshape(-1, 1, 28, 28), np.load(dataset_path + 'y.npy')
    print(x.shape)
    print(y.shape)

    n = int(0.9 * x.shape[0])
    tx, ty, vx, vy = x[:n], y[:n], x[n:], y[n:]

    tx = tx.reshape((-1, 1024, 1, 28, 28))
    vx = vx.reshape((-1, 1024, 1, 28, 28))
    
    ty = get_one_hot(ty, 9)
    vy = get_one_hot(vy, 9)

    ty = ty.reshape((-1, 1024, 9))
    vy = vy.reshape((-1, 1024, 9))

    print(vx.shape)
    print(vy.shape)

    try:
        cProfile.run('j, jv = c.adam_momentum(60, tx, ty, vx, vy, e=1e-4, wd=1e-9, k=1024)')
    except KeyboardInterrupt:
        with open('google-model.pickle', 'wb') as f:
            pickle.dump(c, f)
        raise

    with open('google-model.pickle', 'wb') as f:
        pickle.dump(c, f)
    
    y = np.load(dataset_path + 'y.npy')
    ty, vy = y[:n], y[n:]
    
    print('trainig testL: ')
    c.test(tx[:2048].reshape((-1, 64, 1, 28, 28)), ty[:2048].reshape((-1, 64)))

    print('validation testL: ')
    c.test(vx[:2048].reshape((-1, 64, 1, 28, 28)), vy[:2048].reshape((-1, 64)))

    fig, axs = plt.subplots(2)
    axs[0].plot(range(len(j)), j)
    axs[1].plot(range(len(jv)), jv)

    plt.show()