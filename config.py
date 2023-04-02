##################################################
# Training Config
##################################################
GPU = '0'                   # GPU
workers = 0                 # number of Dataloader workers
epochs = 40                # number of epochs
#batch_size = 12             # batch size
batch_size=6
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
#image_size = (448, 448)   # size of training images
image_size = (224,224)
#net = 'inception_mixed_6e'  # feature extractor
#net='resnet50'
#net='convnext50'
#net='vgg19_bn'
net='iResnet50'
#net='xception'
#net='conv_res'
#net='pyconv'
#net='fcanet'
cutmix_alpha=1.3
num_attentions = 32         # number of attention maps
beta = 5e-2                 # param for update feature centers
cal_lambda=0.1
##################################################
# Dataset/Path Config
##################################################
tag = 'bird'                # 'aircraft', 'bird', 'car', or 'dog'

# saving directory of .ckpt models
save_dir = './FGVC/CUB-200-2011/ckpt/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
#ckpt = 'C:/Users/ZYYZNB/bishe_3/FGVC/CUB-200-2011/ckpt/model.ckpt'
ckpt=False
# ckpt = save_dir + model_name

##################################################
# Eval Config
##################################################
visualize = True
eval_ckpt = save_dir + model_name
eval_savepath = './FGVC/CUB-200-2011/visualize/'

eval_ckpt1='model_1.ckpt'
eval_ckpt2='model_2.ckpt'
eval_ckpt3='model_3.ckpt'