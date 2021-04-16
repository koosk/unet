from model import *
from data_mine import *
import argparse

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=str, required=True)
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--val', type=str, default=None)
parser.add_argument('--test', type=str, default=None)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--batch', type=int, default=2)
args = parser.parse_args()

# if model exists skip training and only predict test images
if os.path.isfile(args.model + '.json') and os.path.isfile(args.model + '_weights.h5'):
    # load json and create model
    json_file = open(args.model + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(args.model + '_weights.h5')
    print("Loaded model from disk")

else:
    # do training first
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    # myGene = trainGenerator(args.batch,args.train,'images','unet_masks',data_gen_args,save_to_dir = None,target_size = (256,256),image_color_mode = "rgb")

    myGene = trainGenerator(args.batch, args.train, 'images', 'masks_single_rgb', data_gen_args,
                            save_to_dir=None, target_size=(256, 256), image_color_mode="rgb")

    model = unet(input_size=(256, 256, 3))
    model_checkpoint = ModelCheckpoint(args.model + '.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=args.steps, epochs=args.epochs, callbacks=[model_checkpoint])

    # save model as json
    model_json = model.to_json()
    with open(args.model + '.json', 'w') as f:
        f.write(model_json)

    # save weights too
    model.save_weights(args.model + '_weights.h5')

# start prediction either after training or model loading
predictNSaveImage(model, args.test, args.results, target_size=(256, 256))

'''
imageFiles = [f for f in os.listdir(os.path.join(args.test,'images')) if os.path.isfile(os.path.join(args.test,'images',f))]
imcount = len(imageFiles)
print('Found ',str(imcount),' images in folder ',os.path.join(args.test,'images'))
testGene = testGeneratorCustom(args.test,imageFiles,num_image=imcount) #testGenerator(args.test)
results = model.predict_generator(testGene,imcount,verbose=1) #,30,verbose=1
saveResult(args.results,results)
'''
'''
testGene = testGenerator(args.test) #testGenerator(args.test)
results = model.predict_generator(testGene,12,verbose=1) #,30,verbose=1
saveResult(args.results,results)
'''
